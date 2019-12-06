#install packages
install.packages("randomForest")
install.packages("Metrics")
install.packages("gbm")
install.packages("xgboost")
install.packages(c("corrplot", "gbm", "ggthemes", "Metrics", "mlr", "randomForest", "xgboost","easypackages"))

#required libraries
library(plyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(ggthemes)
library(caret)
library(MASS)
library(randomForest)
library(party)
library(readr)
library(class)
library(caret)
library(e1071)
library(mlr)
library("cluster")
library("factoextra")
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(Metrics)
library(xgboost)
library(BSDA)
library(easypackages)
libraries("plyr","corrplot","ggplot2","gridExtra","ggthemes","caret","MASS","randomForest","party","readr","class","caret","e1071","mlr","cluster","factoextra","rpart","gbm","Metrics","xgboost","glmnet","biglasso","leaps","")


#Dataset loading
Churn_Modelling <- read_csv("Churn_Modelling.csv")
#View(Churn_Modelling)
Churn_Model <- Churn_Modelling  #(Used for Hypothesis Testing)

#Data Cleaning
Churn_Modelling[,c('RowNumber', 'CustomerId', 'Surname', 'Geography')] <- NULL
Churn_Modelling$Gender[Churn_Modelling$Gender=='Male'] <- 1
Churn_Modelling$Gender[Churn_Modelling$Gender=='Female'] <- 0
count(Churn_Modelling$Exited)
hist(Churn_Modelling$Exited)
y <- Churn_Modelling$Exited
x_data <- Churn_Modelling
#x_data[,c('Exited')] <- NULL
sapply(x_data,class)
x_data$Gender <- as.numeric(x_data$Gender)
x <- (x_data - min(x_data))/(max(x_data)-min(x_data))
x
summary(x_data)


#Training and Testing Data Split
set.seed(1) 
row.number <- sample(x=1:nrow(x), size=0.7*nrow(x))
x_train = x[row.number,]
x_test = x[-row.number,]
head(x_train)
head(x_test)


#Training and Testing Data Split on normal data
set.seed(1) 
row.number <- sample(x=1:nrow(x_data), size=0.7*nrow(x_data))
train = x_data[row.number,]
test = x_data[-row.number,]
head(train)
head(test)


#Descriptive Statis
summary(x_data)
#Scatterplot
scatter.smooth(x = x_data$Balance, y = x_data$EstimatedSalary,  main="Scatterplot with Regression Line", xlab="Balance", ylab="EstimatedSalary")
scatter.smooth(x = x_data$CreditScore, y = x_data$Age,  main="Scatterplot with Regression Line", xlab="CreditScore", ylab="Age")
scatter.smooth(x = x_data$Balance, y = x_data$Age,  main="Scatterplot with Regression Line", xlab="Balance", ylab="Age")
scatter.smooth(x = x_data$CreditScore, y = x_data$Balance,  main="Scatterplot with Regression Line", xlab="CreditScore", ylab="Balance")
#Density Plot
density.Exited <- density(x_data$Exited)
plot(density.Exited, main="Exited Density of Customers")
polygon(density.Exited, col="blue")
#Boxplot
boxplot(x_data$CreditScore ~ x_data$Age, data = x_data, main="Boxplot", xlab="CreditScore", ylab="Age")
boxplot(x_data$Balance ~ x_data$Age, data = x_data, main="Boxplot", xlab="Balance", ylab="Age")


#Hypothesis Testing
set.seed(1) 
#One Sample t-test - Is the credit score greater than 500?
onesample <- t.test(Churn_Model$CreditScore, mu = 500, alternative = "greater")
onesample

#Two Sample t-test - Do males and females have the same credit scores?
male_cust <- subset(Churn_Model, subset = (Churn_Model$Gender == 'Male'))
female_cust <- subset(Churn_Model, subset = (Churn_Model$Gender == 'Female'))
twosample <- t.test(male_cust$CreditScore, female_cust$CreditScore, data = Churn_Model, var.equal = TRUE)
twosample

#Paired t-test - Did more males retain their banks accounts in comparison to the females?
pairedtest <- t.test(male_cust$Exited, female_cust$Exited, alternative = "greater")
pairedtest

#Test of equal proportions - Does the geography of the customer impact on whether the customer leaves the bank?
proptest <- prop.test(table(Churn_Model$Geography, Churn_Model$Exited), p = NULL, conf.level = 0.95, correct = FALSE, alternative = c("two.sided"))
proptest

#F-test - Test the variance of Estimated Salary in males and females?
ftest <- var.test(male_cust$EstimatedSalary, female_cust$EstimatedSalary, data = Churn_Model, alternative = "less")
ftest

#Z-test - Considering the Active Customers, is the mean of Exited Clients more than the ones who chose to stay?
active_cust <- subset(Churn_Model, subset = (Churn_Model$IsActiveMember == '1'))
inactive_cust <- subset(Churn_Model, subset = (Churn_Model$IsActiveMember == '0'))
ztest <- z.test(active_cust$Exited, inactive_cust$Exited, alternative = "greater", mu = 0 , sigma.x = sd(active_cust$Exited), sigma.y = sd(inactive_cust$Exited))
ztest

#ANOVA
res.aov <- aov(y ~ x$CreditScore+x$Balance+x$EstimatedSalary, data = x_train)
summary(res.aov)

#Linear Regression
set.seed(1) 
linear.model <- lm(y ~ x$CreditScore + x$Balance + x$EstimatedSalary, data = x_train)
summary(linear.model)
prediction.lm <- predict(linear.model, newdata=x_test)
prediction.lm
prediction.lm <- ifelse(prediction.lm>0.5,1,0)
linear.prediction.lm <- mean(prediction.lm!=x_test$Exited)
print(paste('Accuracy',1-linear.prediction.lm))

#Logistic Regression
set.seed(1) 
mylogit <- glm(Exited ~ ., data = x_train, family = binomial(link='logit'))
summary(mylogit)
#confint(mylogit)
#confint.default(mylogit)
#exp(coef(mylogit))
log_pred <- predict(mylogit, newdata = x_test)
tab_log <- table(x_test$Exited,log_pred)
tab_log
log_pred <- ifelse(log_pred>0.5,1,0)
log_pred_res <- mean(log_pred!= x_test$Exited)
print(paste('Accuracy', 1-log_pred_res))

#KNN
set.seed(1) 
#nrow(x_train)
x.train.target<- x[1:7000,10]
x.test.target<- x[7001:10000,10]
knn_model <- knn(x_train,x_test,cl=x.train.target,k=13)
summary(knn_model)
tab <- table(knn_model,x.test.target)
tab
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)

#SVM
set.seed(1) 
svm_model <- svm(as.factor(Exited) ~ ., data=x_train)
summary(svm_model)
#svm_model1 <- svm(x_data,y)
#summary(svm_model1)
svm_pred <- predict(svm_model,x_test)
system.time(pred <- predict(svm_model,x))
tab_svm <- table(x_test$Exited,svm_pred)
accuracy(tab_svm)

#Naive Bayes
set.seed(1) 
NBclassfier=naiveBayes(as.factor(Exited) ~ ., data=x_train)
print(NBclassfier)
NB_Predictions <- predict(NBclassfier, x_test)
tab_naive <- table(NB_Predictions,x_test$Exited)

accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab_naive)

#Decision Tree
set.seed(1) 
fit <- rpart(y~x$CreditScore+x$Gender+x$Age+x$Tenure+x$Balance+x$NumOfProducts+x$HasCrCard+x$IsActiveMember+x$EstimatedSalary, data = x_train, method = 'class')
fit <- rpart(as.factor(x_train$Exited) ~ ., data = x_train, method = 'class')
rpart.plot(fit, extra = 106)
predict_unseen <-predict(fit, x_test, type = 'class')
table_mat <- table(x_test$Exited, predict_unseen)
table_mat
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(table_mat)
library(party)
fit <- ctree(as.factor(x_train$Exited) ~ ., data = x_train)
plot(fit, main="Conditional Inference Tree for Churn Model")
predict_unseen <- predict(fit, x_test)
table_mat <- table(x_test$Exited, predict_unseen)
table_mat
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(table_mat)
mean(predict_unseen == x_test$Exited)

#Random Forest
set.seed(1) 
randomforest.model <- randomForest(as.factor(x_train$Exited) ~ ., data = x_train, importance = TRUE)
randomforest.model
plot(randomforest.model)
randomforest.model_learnt_mtry2 <- randomForest(as.factor(x_train$Exited) ~ ., data = x_train, ntree = 200, mtry = 2, importance = TRUE)
print(randomforest.model_learnt_mtry2)
plot(randomforest.model_learnt_mtry2)
randomforest.model_learnt_mtry4 <- randomForest(as.factor(x_train$Exited) ~ ., data = x_train, ntree = 200, mtry = 4, importance = TRUE)
randomforest.model_learnt_mtry4
randomforest.model_learnt_mtry7 <- randomForest(as.factor(x_train$Exited) ~ ., data = x_train, ntree = 200, mtry = 7, importance = TRUE)
randomforest.model_learnt_mtry7
plot(randomforest.model_learnt_mtry7)
predTest <- predict(randomforest.model_learnt_mtry2,x_test)
predicted_table <- table(observed=x_test$Exited,predicted=predTest)
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(predicted_table)
mean(predTest == x_test$Exited) 
importance(randomforest.model_learnt_mtry2)        
varImpPlot(randomforest.model_learnt_mtry2) 


#Gradient Boosting Machine Algorithm
set.seed(1) 
churn_gbm.model <- gbm(as.factor(x_train$Exited) ~ . ,data = x_train, distribution = "gaussian",n.trees = 5000,shrinkage = 0.01, interaction.depth = 4,cv.folds = 5)
churn_gbm.model
predmatrix_gbm<-predict(churn_gbm.model,x_test,n.trees = 5000)
predicted_table1 <- table(observed=x_test$Exited,predicted=predmatrix_gbm)
length(predmatrix_gbm)
prediction <- as.numeric(predicted_table1 > 0.5)
print(head(prediction))
err <- mean(as.numeric(predicted_table1 > 0.5) != x_test$Exited)
print(paste("test-error=", err))

#XGBOOST
set.seed(1) 
bstDense <- xgboost(data = as.matrix(x_train), label = x_train$Exited, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
dtrain <- xgb.DMatrix(data = as.matrix(x_train), label = x_train$Exited)
bstDMatrix <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic")
bst <- xgboost(data = dtrain, max.depth = 2, eta = 1, nthread = 2, nrounds = 2, objective = "binary:logistic", verbose = 1)
pred <- predict(bst, as.matrix(x_test))
# size of the prediction vector
print(length(pred))
## [1] 1611
# limit display of predictions to the first 10
print(head(pred))
prediction <- as.numeric(pred > 0.5)
print(head(prediction))
err <- mean(as.numeric(pred > 0.5) != x_test$Exited)
print(paste("test-error=", err))
predicted_table <- table(observed=x_test$Exited,predicted=pred)
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(predicted_table)


#Kmeans Clustering
set.seed(1411)
df <- x_train
fviz_nbclust(scale(df), kmeans, method = "silhouette") + labs(subtitle = "Silhouette method")
#Creating 3 clusters
kmean1 = kmeans(df[ ,], 3)
kmean1
plot(df[,], col=kmean1$cluster)
points(kmean1$centers[, c("CreditScore","Age","Balance", "EstimatedSalary")],
       col="white", pch="*", cex=5)
kmean1$centers

#Regularization Techniques
model_x <- model.matrix(x$Exited~.,x)[,-1]
Exited <- (x$Exited)
#generating a general linear regression model
model_lreg = (lm(Exited ~., data= x))
summary(model_lreg)
par(mfrow=c(3,5))
for(i in 1:10){
  plot(as.matrix(x_data[,i]), Exited,xlab = i)
  abline(lm(Exited~as.matrix(x_data[,i])),col = "blue")
}
#gerating a lasso model
lasso_fit <- glmnet(model_x,Exited, alpha=1)
length(lasso_fit$lambda)
coef(lasso_fit)[,50]
par(mfrow=c(1,1))
plot(lasso_fit ,xvar="lambda",main="Lasso (alpha=1)")


#genrating a RIDGE MODEL
ridge_fit <- glmnet(model_x,Exited,alpha=0)
plot(ridge_fit,xvar="lambda",label=TRUE)

ridge_cv <- cv.glmnet(model_x,Exited,alpha=0)
plot(ridge_cv)

set.seed(141197)
train=sample(1:nrow(model_x),size=0.7*nrow(model_x)) 
test=(-train)
fit_training <- glmnet(model_x[train,],Exited[train],alpha=1,lambda=100)
cv_training <- cv.glmnet(model_x[train,],Exited[train],alpha=1) 
par(mfrow=c(1,1))
plot(cv_training)

cv_bestlam <- cv_training$lambda.min
predict(lasso_fit,type ="coefficients",s=cv_bestlam)[1:10,] 
#genrating REGRESSION SUMMARY
reg_fit <- regsubsets(Exited~., data = x,
                      nvmax = 19)
reg_summary <- summary(reg_fit)
names(reg_summary)
plot(reg_summary$rss, xlab="No. of variables", ylab="RSS",main = "RSS for Exiters")
which.min(reg_summary$rss)
#etended lasso model
#linear LASSO extended
model_x_matrix <- as.big.matrix(model_x,backingfile = "")
fit_big <- biglasso(model_x_matrix, Exited, family = 'gaussian')
plot(fit_big, log.l = TRUE, main = 'Extended lasso')

