library(readr)
library('BSDA')

Churn_Modelling <- read_csv("~/Downloads/Churn_Modelling.csv")
View(Churn_Modelling)
summary(Churn_Modelling)

#Hypothesis Testing

#One Sample t-test - Is the credit score greater than 500?
onesample <- t.test(Churn_Modelling$CreditScore, mu = 500, alternative = "greater")
onesample

#Two Sample t-test - Do males and females have the same credit scores?
male_cust <- subset(Churn_Modelling, subset = (Churn_Modelling$Gender == 'Male'))
female_cust <- subset(Churn_Modelling, subset = (Churn_Modelling$Gender == 'Female'))
twosample <- t.test(male_cust$CreditScore, female_cust$CreditScore, data = Churn_Modelling, var.equal = TRUE)
twosample

#Paired t-test - Did more males retain their banks accounts in comparison to the females?
pairedtest <- t.test(male_cust$Exited, female_cust$Exited, alternative = "greater")
pairedtest

#Test of equal proportions - Does the geography of the customer impact on whether the customer leaves the bank?
proptest <- prop.test(table(Churn_Modelling$Geography, Churn_Modelling$Exited), p = NULL, conf.level = 0.95, correct = FALSE, alternative = c("two.sided"))
proptest

#F-test - Test whether the variance of Estimated Salary is greater in males or females?
ftest <- var.test(male_cust$EstimatedSalary, female_cust$EstimatedSalary, data = Churn_Modelling, alternative = "less")
ftest

#Z-test - Considering the Active Customers, is the mean of Exited Clients more than the ones who chose to stay?
active_cust <- subset(Churn_Modelling, subset = (Churn_Modelling$IsActiveMember == '1'))
inactive_cust <- subset(Churn_Modelling, subset = (Churn_Modelling$IsActiveMember == '0'))
ztest <- z.test(active_cust$Exited, inactive_cust$Exited, alternative = "greater", mu = 0 , sigma.x = sd(active_cust$Exited), sigma.y = sd(inactive_cust$Exited))
ztest
