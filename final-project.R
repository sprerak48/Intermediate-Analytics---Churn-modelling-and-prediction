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
Churn_Modelling <- read_csv("~/Downloads/Churn_Modelling.csv")
View(Churn_Modelling)

summary(Churn_Modelling)

Churn_Modelling[,c(1:3, 5)] <- NULL
Churn_Modelling$Gender[Churn_Modelling$Gender=='Male'] <- 1
Churn_Modelling$Gender[Churn_Modelling$Gender=='Female'] <- 0


y <- Churn_Modelling$Exited
y

#Churn_Modelling$Exited <- NULL


Churn_Modelling_norm <- as.data.frame(t(apply(Churn_Modelling[1,3:6,9], 1, function(x) (x - min(x))/(max(x)-min(x)))))
summary(Churn_Modelling_norm)

normalize <- function(x){
  return ((x-min(x))/(max(x)-min(x)))
}

Churn_Modelling.new<- as.data.frame(lapply(Churn_Modelling[,c(1,3:6,9)],normalize))
head(Churn_Modelling.new)



iris.train<- iris.new[1:130,]
iris.train.target<- iris[1:130,5]
iris.test<- iris.new[131:150,]
iris.test.target<- iris[131:150,5]
summary(iris.new)


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

