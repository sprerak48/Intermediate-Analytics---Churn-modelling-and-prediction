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

#Churn_Modelling <- read_csv("~/Downloads/Churn_Modelling.csv")
Churn_Modelling <- read_csv("C:/Users/Chirag/Desktop/FinalProject-Intermediate/Churn_Modelling/Churn_Modelling.csv")
#View(Churn_Modelling)

summary(Churn_Modelling)

Churn_Modelling[,c(1:3, 5)] <- NULL
Churn_Modelling$Gender[Churn_Modelling$Gender=='Male'] <- 1
Churn_Modelling$Gender[Churn_Modelling$Gender=='Female'] <- 0


y <- Churn_Modelling$Exited
y

#Churn_Modelling$Exited <- NULL


#Churn_Modelling_norm <- as.data.frame(t(apply(Churn_Modelling[1,3:6,9], 1, function(x) (x - min(x))/(max(x)-min(x)))))
#summary(Churn_Modelling_norm)

normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

Churn_Modelling.new<- as.data.frame(lapply(Churn_Modelling[,c(1,3:6,9)],normalize))
head(Churn_Modelling.new)


set.seed(1) 
row.number <- sample(x=1:nrow(Churn_Modelling.new), size=0.8*nrow(Churn_Modelling.new))
train = Churn_Modelling.new[row.number,]
test = Churn_Modelling.new[-row.number,]
head(train)
head(test)

dim(train)
dim(test)

Churn_Modelling.new.train.target<- Churn_Modelling.new[1:8000,6]
Churn_Modelling.new.test.target<- Churn_Modelling.new[8001:10000,6]



logistic_model <- glm(Churn_Modelling$Exited ~ Churn_Modelling$CreditScore+Churn_Modelling$EstimatedSalary, family=binomial(link="logit"), data=train)
summary(logistic_model)

logistic_model$coefficients[2]
exp(logistic_model$coefficients[1]+logistic_model$coefficients[2])/(1+exp(logistic_model$coefficients[1]+logistic_model$coefficients[2]))


pred <- predict(logistic_model, newdata = test, type = "response")
pred


model<- knn(train=train, test=test, cl=Churn_Modelling.new.train.target, k=8)
table(Churn_Modelling.new.test.target, model)


#K-mean clustering and KNN algorithms
install.packages("factoextra")
library(factoextra)
install.packages("rpart.plot")
library(rpart.plot)
install.packages("party")
library(party)
install.packages("fpc")
library(fpc)
install.packages("caret")
library(caret)
install.packages("NClust")
library(NbClust)

set.seed(1)
Churn_Modelling <- (na.omit(Churn_Modelling))

# Perform K-Means with 2 clusters
library(datasets)

set.seed(2)
pred_head <- head(Churn_Modelling,100)
churn_df = pred_head[,c(1,9)]
churn_kmean1 = kmeans(churndf, 2, nstart = 100)
churn_kmean1
# Plot results
kmean_churn_plot <- plot(churn_df, col =(churn_kmean1$cluster+16) , 
                           main="churn_K-Means result with 3 clusters", pch=20, cex=2)

churn_kmean1$cluster

#optimal number of clustering
cluster.churn <- churn_df
within.cluster.sos <- (nrow(cluster.churn)-1)*sum(apply(cluster.churn,2,var))
for (i in 1:15) within.cluster.sos[i] <- sum(kmeans(cluster.churn,
                                                    centers=i)$withinss)

plot(1:15, within.cluster.sos, type="b", xlab="Number of Clusters",
     ylab="Within groups sum of squares",
     main="Total within-clusters sum of squares with the Elbow Method",
     pch=20, cex=2)



# Perform K-Means with the optimal number of clusters identified from the Elbow method
set.seed(2)
km2 = kmeans(churn_df, 8, nstart=100)

# Examine the result of the clustering algorithm
km2

plot(churn_df, col =(km2$cluster+1) , main="K-Means optimal clustering result with 8 clusters", pch=20, cex=2)


#Random Forest
install.packages("randomForest")
library(randomForest)


randomforest.model <- randomForest(EstimatedSalary ~ ., data = train, importance = TRUE)
randomforest.model

randomforest.model_learnt_mtry2 <- randomForest(EstimatedSalary ~ ., data = train, ntree = 200, mtry = 2, importance = TRUE)
randomforest.model_learnt_mtry2
#as negative var expalined so model is overfitted

randomforest.model_learnt_mtry4 <- randomForest(EstimatedSalary ~ ., data = train, ntree = 200, mtry = 4, importance = TRUE)
randomforest.model_learnt_mtry4

randomforest.model_learnt_mtry5 <- randomForest(EstimatedSalary ~ ., data = train, ntree = 200, mtry = 5, importance = TRUE)
randomforest.model_learnt_mtry5

predTrain <- predict(randomforest.model_learnt_mtry2, train, type = "class")
# Checking classification accuracy
table(predTrain, train$EstimatedSalary) 

# Predicting on Validation set
predTest <- predict(randomforest.model_learnt_mtry2, test,type = "class")
# Checking classification accuracy
                    
predicted_table <- table(observed=test$EstimatedSalary,predicted=predTest)

importance(randomforest.model_learnt_mtry2)        
varImpPlot(randomforest.model_learnt_mtry2) 
# negative %IncMSE means It can be inferred that the variable does not have a role in the prediction,i.e, not important.
#The importance measures show how much MSE or Impurity increase when that variable is randomly permuted.


#plot the predicted values (out-of-bag estimation) vs. true values
plot( predict(randomforest.model_learnt), train$EstimatedSalary)
abline(c(0,1),col=3)



#GBM
install.packages("gbm")
library(gbm)
churn_gbm.model=gbm(EstimatedSalary ~ . ,data = train,distribution = "gaussian",n.trees = 5000,
                 shrinkage = 0.01, interaction.depth = 4)
churn_gbm.model

summary(churn_gbm.model) 
#Summary gives a table of Variable Importance and a plot of Variable Importance
#the most importatnt varibale is marked at top in summary\
cor(train$CreditScore,train$EstimatedSalary)#negetive correlation coeff-r
cor(train$Balance,train$EstimatedSalary)#positive correlation coeff-r


n.trees = seq(from=100 ,to=5000, by=50) #no of trees-a vector of 100 values 

#Generating a Prediction matrix for each Tree
predmatrix_gbm<-predict(churn_gbm.model,test,n.trees = n.trees)
dim(predmatrix_gbm) #dimentions of the Prediction Matrix

#Calculating The Mean squared Test Error
test.error<-with(test,apply( (predmatrix_gbm-EstimatedSalary)^2,2,mean))
head(test.error) #contains the Mean squared test error for each of the 100 trees averaged



