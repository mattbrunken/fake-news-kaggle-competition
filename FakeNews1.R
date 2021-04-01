## libraries
library(tidyverse)
library(xgboost)
library(caret)

## read in Dr. Heaton's cleaned up fake news dataset

fake_news <- vroom::vroom('/Users/matthewbrunken/Winter2021/Kaggle/competition4/CleanFakeNews.csv')


## split into train and test

train <- fake_news %>% filter(Set == 'train')
test <- fake_news %>% filter(Set == 'test')

## extract x and y to make data suitable (matrix form) for XGboost modeling

train_y <- data.matrix(train[,'isFake'])
train_x <- data.matrix(train[,-c(1,2)])
train_x <- train_x[,-2]

test_y <- data.matrix(test[,'isFake'])
test_x <- data.matrix(test[,-c(1,2)])


## convert to xgboost matrix data type
xgb_train = xgb.DMatrix(data = train_x, label = train_y)
xgb_test = xgb.DMatrix(data = test_x, label = test_y)


## fit XgBoost model
xgbc = xgboost(data = xgb_train, max.depth = 3, nrounds = 50)

## predict new values
preds = predict(xgbc, xgb_test)

## select cutoff for binary classification
preds_binary <- ifelse(preds > 0.5, 1, 0)

## put predictions in test dataframe
test$isFake <- preds_binary

## extract for submission
submission <- test[,c('Id', 'isFake')] 
names(submission) <- c('id', 'label')
  
## write out to .csv for submission1
write.csv(submission, '/Users/matthewbrunken/Winter2021/Kaggle/competition4/submission1.csv',
          row.names = FALSE)








##################################################################3

## SUBMISSION 2: LOGISITIC REGRESSION

# CUT DATA IN HALF TO MAKE IT RUN FASTER

train_slim <- train[,-c(1,4)]
train_slim <- train_slim[,1:(.1*ncol(train) %>% round(digits = -1) # take 20% of the variables
)]


# try to fit a logisitic regression with 10% of the explanatory variables
classifier <- glm(isFake ~ .,
                  family= 'binomial' ,data = train_slim)

# use model to predict on the test set
preds = predict(classifier, test)

# convert logits to 1s and 0s
preds2 <- ifelse(preds > .5, 1, 0)

# fill the isFake Variable with the predictions
test$isFake <- preds2

# create submission dataframe
submission2 <- test[,c('Id', 'isFake')]
names(submission2) <- c('id', 'label')

# create submission file
write.csv(submission2, '/Users/matthewbrunken/Winter2021/Kaggle/competition4/submission2.csv',
          row.names = FALSE)















nb_mod <- e1071::naiveBayes(isFake ~ . , train_clean, laplace = 3)

preds <- predict(nb_mod, test %>% pull('isFake'),
                 threshold = 0.0001)
























## SUBMISSION 3: XGBOOST WITH MORE PARAMATER TUNING

## SET UP TRAIN AND TEST MATRIX

x_train <- train_x

x_pred <- test_x

y_train <- train_y %>% 
  factor(labels = c("yes", "no")) ## XGboost doesn't like 1 vs 0

## set up xgboost
xgb_grid <- base::expand.grid(
   list(
    nrounds = c(100, 200),
    max_depth = c(10, 15, 20), # maximum depth of a tree
    colsample_bytree = seq(0.5), # subsample ratio of columns when construction each tree
    eta = 0.1, # learning rate
    gamma = 0, # minimum loss reduction
    min_child_weight = 1,  # minimum sum of instance weight (hessian) needed ina child
    subsample = 1 # subsample ratio of the training instances
))

# pack the training control parameters
xgb_trcontrol = trainControl(
  method = "cv",
  number = 5,
  verboseIter = TRUE,
  returnData = FALSE,
  returnResamp = "all",                                                        # save losses across all models
  classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
  summaryFunction = twoClassSummary,
  allowParallel = TRUE
)

# build model
xgb_train_1 = train(
  x = x_train,
  y = y_train,
  trControl = xgb_trcontrol,
  tuneGrid = xgb_grid,
  method = "xgbTree"
)

