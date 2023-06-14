library(tidyverse)
library(grid)
library(knitr)
library(dplyr)
library(ggplot2)
library(gridExtra)
library(readr)

#Import data:

math_grade2 <- read.csv("student-mat.csv")
head(math_grade2)
dim(math_grade2)

#The data set has 395 observations and 33 columns.

#Missing values:

nrow(math_grade2[!complete.cases(math_grade2), ])

#There is no missing value.

#We are going to predict the students final math grades based using the remaining 32 predictors. 

colnames(math_grade2)

#Check the data types of all columns

sapply(math_grade2, class)

#The variables having "character" as their data types are school, sex, address, famsize, Pstatus, Mjob, Fjob, reason, guardian, schoolsup, famsup, paid, activities, nursery, higher, internet, romantic

#Encode all categorical variables from their original data type of character. 

math_grade2$school = factor(math_grade2$school)
math_grade2$sex = factor(math_grade2$sex)
math_grade2$address = factor(math_grade2$address)
math_grade2$famsize = factor(math_grade2$famsize)
math_grade2$Pstatus = factor(math_grade2$Pstatus)
math_grade2$Mjob = factor(math_grade2$Mjob)
math_grade2$Fjob = factor(math_grade2$Fjob)
math_grade2$reason = factor(math_grade2$reason)
math_grade2$guardian = factor(math_grade2$guardian)
math_grade2$schoolsup = factor(math_grade2$schoolsup)
math_grade2$famsup = factor(math_grade2$famsup)
math_grade2$paid = factor(math_grade2$paid)
math_grade2$activities = factor(math_grade2$activities)
math_grade2$nursery = factor(math_grade2$nursery)
math_grade2$higher = factor(math_grade2$higher)
math_grade2$internet = factor(math_grade2$internet)
math_grade2$romantic = factor(math_grade2$romantic)


#Check data types

unique(sapply(math_grade2, class))

#Data plot for the final grade to be predicted: 
#G1 - first period grade (numeric: from 0 to 20)
#G2 - second period grade (numeric: from 0 to 20)
#G3 - final grade (numeric: from 0 to 20, output target)

hist(math_grade2$G1)
hist(math_grade2$G2)
hist(math_grade2$G3)

#The three distributions are normal as expected. However, there are nearly 40 students gained 0. This might because they did not attend the exam.

#Plot gender (Female students are a little more than male students):

ggplot(data = math_grade2, aes(x = sex)) + geom_bar()

#Plot address (most students live in urban):

ggplot(data = math_grade2, aes(x = address)) + geom_bar()

#Plot mother job:

ggplot(math_grade2, aes(x=Mjob)) + geom_bar()

#Plot father job:

ggplot(math_grade2, aes(x=Fjob)) + geom_bar()


# Elastic net

set.seed(1)
rsq_list = c()
rmse_list = c()
# create a list to record alpha
al = c()
for (d in 0:10){
  # setting alpha to be 0.1, 0.2 ... to 1.0
  al_pha = d/10
  al = append(al,al_pha)
  r_model = glmnet(x = data.matrix(grade_trn[, 1:32]), y = grade_trn[,33], alpha = al_pha)
  cv.fit = cv.glmnet(x = data.matrix(math_grade2[, 1:32]), y = math_grade2[,33], 
                     nfolds = 10, alpha = al_pha)
  
  best_lambda <- cv.fit$lambda.min
  y_predicted <- predict(r_model, s = best_lambda, newx = data.matrix(grade_tst[, 1:32]))
  tss = sum((grade_tst[,33] - mean(grade_tst[,33]))^2)
  sse = sum((y_predicted - grade_tst[,33])^2)
  #find R-Squared
  rsq <- 1 - sse/tss
  rsq_list = append(rsq_list, rsq)
  #find RMSE
  rmse = sqrt(mean((grade_tst$G3 - y_predicted)^2))
  rmse_list = append(rmse_list, rmse)
}

#plot alpha vs Rsquared for the coresponding alpha
plot(al, rsq_list,
     xlab = "alpha",
     ylab = "area under curve",
     type="l",
     col="blue",
)
#plot alpha vs rmse for the coresponding alpha
plot(al, rmse_list,
     xlab = "alpha",
     ylab = "area under curve",
     type="l",
     col="red",
)

which.max(rsq_list)
which.min(rmse_list)
# alpha = 0.9 maximize the R-squared, and minimized rmse
rsq_list[which.max(rsq_list)]
rmse_list[which.min(rmse_list)]

# rebuild model when alpha = 0.9
r_model = glmnet(x = data.matrix(grade_trn[, 1:32]), y = grade_trn[,33], alpha = 0.9)
cv.fit = cv.glmnet(x = data.matrix(math_grade2[, 1:32]), y = math_grade2[,33], 
                   nfolds = 10, alpha = 0.9)
plot(cv.fit)
best_lambda <- cv.fit$lambda.min
best_model = glmnet(x = data.matrix(grade_trn[, 1:32]), y = grade_trn[,33], alpha = 0.9, lambda = best_lambda)
coef(best_model)

#The R-squared is higher than the one(0.8709272) we got in ridge regression. 
#Up to now, the elastic net when alpha = 0.9 performs the best, having R-squared 0.9061616 and rmse 1.257866.
#The coefficients selected by the this model (elastic net when alpha = 0.9) are studytime, failures, activities, romantic, famrel, Walc, health, absences, G1, G2.

#Split data to training and tesing (set 80% training data and 20% testing data), then we use the all predictors to predict G3 by constructing a linear regression:

library(caret)
set.seed(1)
trn_idx = createDataPartition(math_grade2$G3, p = 0.80, list = FALSE)
grade_trn = math_grade2[trn_idx, ]
grade_tst = math_grade2[-trn_idx,]
model1 = lm(G3~.,data = grade_trn)
summary(model1)

pred_test = predict(model1, grade_tst)
data_prediction = (data.frame((pred_test), (grade_tst$G3),(abs(pred_test - grade_tst$G3))))
colnames(data_prediction) <- c("Predicted G3","Real G3","Difference")
head(data_prediction,10)
max(data_prediction$Difference)
mean(data_prediction$Difference)
dim(data_prediction)

# calculate RMSE
sqrt(mean((grade_tst$G3 - pred_test)^2))


#We can see that the maximum prediction error among the 77 testing data points is 4.15 of a 20 points scale; the mean of the prediction error is 1.24 which is quite acceptable.

#The R-squared is 0.839, indicating the linear regression model for predicting students final math grade is reasonable. RMSE is 1.583396

#Later we will also perform Lasso regression and principle components analysis to do variable selection. We will also do testings like F-tests to explore the significance of the predictors. Finally, we will compare models based on the RSME.

#According to the p-value, having a high quality of family relationships, having more absences (a little strange), getting high grades in G1 and G2 will generate positive impact to final grades, while having larger age and having extra activities will negatively affect the final grade. 

# Ridge regression:

set.seed(123)
library(glmnet)
r_model = glmnet(x = data.matrix(grade_trn[, 1:32]), y = grade_trn[,33], alpha = 0)
plot(r_model)

# use k-fold cross validation to identify the lambda value that produces the lowest test mean squared error (MSE)
cv.fit = cv.glmnet(x = data.matrix(math_grade2[, 1:32]), y = math_grade2[,33], 
                   nfolds = 10, alpha = 0)

plot(cv.fit)

# the coefficients:
coef(cv.fit,s = "lambda.min")
# prediction
# pred.cv = predict(cv.fit, data.matrix(math_grade2[, 1:32]), 
#                  type = "response", s = "lambda.min")

best_lambda <- cv.fit$lambda.min

# use the best model of lambda to make predictions on testing data
y_predicted <- predict(r_model, s = best_lambda, newx = data.matrix(grade_tst[, 1:32]))
tss = sum((grade_tst[,33] - mean(grade_tst[,33]))^2)
sse = sum((y_predicted - grade_tst[,33])^2)
#find R-Squared
rsq <- 1 - sse/tss
rsq
#find RMSE
sqrt(mean((grade_tst$G3 - y_predicted)^2))

#The R-squared is higher than the one we got in linear regression, which is 0.839. 
#Since this is a ridge regression, none of the coefficients are opted out, we use all of them to predict G3.
#The RMSE is 1.475236, also is lower than that for linear regression.

# Lasso
set.seed(1)
library(glmnet)
r_model = glmnet(x = data.matrix(grade_trn[, 1:32]), y = grade_trn[,33], alpha = 1)
plot(r_model)

# use k-fold cross validation to identify the lambda value that produces the lowest test m
cv.fit = cv.glmnet(x = data.matrix(math_grade2[, 1:32]), y = math_grade2[,33],
                   nfolds = 10, alpha = 1)
plot(cv.fit)

# the coefficients:
coef(cv.fit,s = "lambda.min")

# prediction
# pred.cv = predict(cv.fit, data.matrix(math_grade2[, 1:32]),
#                  type = "response", s = "lambda.min")
best_lambda <- cv.fit$lambda.min
# use the best model of lambda to make predictions on testing data
y_predicted <- predict(r_model, s = best_lambda, newx = data.matrix(grade_tst[, 1:32])) 
tss = sum((grade_tst[,33] - mean(grade_tst[,33]))^2)
sse = sum((y_predicted - grade_tst[,33])^2)
#find R-Squared
rsq <- 1 - sse/tss
rsq

#find RMSE
sqrt(mean((grade_tst$G3 - y_predicted)^2))

# principle component

set.seed(1)
library(pls)
pcr_model = pcr(G3~.,data = grade_trn, scale = TRUE, validation = "CV")
summary(pcr_model)
validationplot(pcr_model)

# make predictions: (if only use 23 comps, which explains 81.35% of the variation)
pcr_pred = predict(pcr_model, grade_tst, ncomp = 23)
#find R-Squared
tss = sum((grade_tst[,33] - mean(grade_tst[,33]))^2)
sse = sum((pcr_pred - grade_tst[,33])^2)
rsq <- 1 - sse/tss
rsq
rmse = sqrt(mean((grade_tst$G3 - pcr_pred)^2))
rmse

# make predictions: (if use all(41) comps, which explains 100% of the variation)
pcr_pred = predict(pcr_model, grade_tst, ncomp = 41)
#find R-Squared
tss = sum((grade_tst[,33] - mean(grade_tst[,33]))^2)
sse = sum((pcr_pred - grade_tst[,33])^2)
rsq <- 1 - sse/tss
rsq
rmse = sqrt(mean((grade_tst$G3 - pcr_pred)^2))
rmse

#If we use 23 principal components, the R-squared is quite low, which is 0.7394109, the rmse is 2.09615.
#If we use all(41) principal components, the R-squared is 0.851307 and the rmse is 1.583396.
#This indicate this problem is not proper to use PCA techniques, which means that the multi-colinearity between features are not obvious.


