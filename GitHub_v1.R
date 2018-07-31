rm(list = ls())
#setwd("~/Dropbox/Redhat")


library(data.table)
library(dplyr)
library(stringr)
library(h2o)

act_train = fread("act_train.csv", header = T, sep = ",")
act_test = fread("act_test.csv", header = T, sep = ",")
act_test$outcome = -1 # dummy variable
act = rbind(act_train, act_test)
people = fread("people.csv", header = T, sep = ",")

### Cleaning
# merge two data frames by people_id
total <- merge(people, act, by="people_id", suffixes = c("_people", "_action"))
features_to_drop = c("people_id", "activity_id", "group_1", "date_people")
total = total %>% 
  select(which(names(total) %in% features_to_drop == FALSE)) %>%
  select(-c(char_1_action:char_9_action)) %>%
  mutate_at(vars(c(char_1_people, char_2_people:char_37, activity_category:outcome)), funs(as.factor(.)))

# transform some cols into integer values & do imputation if possible
#total$group_1 = as.integer(gsub("group ", "", total$group_1))
total$char_10_action = as.integer(gsub("type ", "", total$char_10_action))
ind_NA_char_10_action = which(is.na(total$char_10_action))
total$char_10_action[ind_NA_char_10_action] = round(median(total$char_10_action, na.rm = T))
total$char_10_action = log(total$char_10_action)

# spread date_action into three day, month and year cols
total$date_action = as.Date(total$date_action)
total = data.frame(total, year = as.numeric(format(total$date_action, format = "%Y")),
                 month = as.numeric(format(total$date_action, format = "%m")),
                 day = as.numeric(format(total$date_action, format = "%d")))
# remove date_action from the total data frame
total$date_action = NULL


### train/test split
train.index = which(total$outcome == -1)
test.data = total[train.index, ]
test.data$outcome = NULL
train.data = total[-train.index, ]


### model fitting using h2o.ai
# library(h2o)
# initialize h2o and establish connection
h2o.init(ip = 'localhost', port = 54321, nthreads= -1,
         max_mem_size = '8g')

# convert to h2o data frame
train.data.h2o = as.h2o(train.data)
test.data.h2o = as.h2o(test.data)

# model fitting using RF
set.seed(1234)
model_rf.h2o = h2o.randomForest(y = "outcome", training_frame = train.data.h2o, nfolds = 2, ntrees = 50)

###  predict rf model via h2o
perf_rf.h2o = h2o.performance(model_rf.h2o, newdata = test.data.h2o)
plot(perf_rf.h2o)
h2o.confusionMatrix(model_rf.h2o, test.data.h2o, metrics = "accuracy")
pred_rf.h2o = as.data.frame(h2o.predict(object = model_rf.h2o, newdata = test.data.h2o, metrics = "accuracy"))
#perf <- h2o.performance(model_rf.h2o, newdata = train.data.h2o)
submitted_file = data.frame(activity_id = act_test$activity_id, outcome = as.integer(as.character(pred_rf.h2o$predict)))
submitted_file = submitted_file[order(submitted_file$activity_id),]
write.table(submitted_file, file = "submission.csv", sep = ",", col.names = T, row.names = F)
