# Problem-8 #
# The data given in the mvn3c2dmix.csv file follow a multivariate 
# distribution with 3 components in 2 dimensions.
# a) Plot the data with respect to classes and comment on the plot
# b) Estimate the parameters of the model using EM algorithm and
#     comment on the estimates.
# c) Draw the estimated density curves for the mixture model and
#     comment.

rm(list=ls())

mydata= read.csv(file.choose(),as.is=TRUE)
head(mydata)

# a)
plot(mydata[1:2],col=c("red","green","blue")[mydata$classes],
     xlab="X1", ylab="X2", pch=19)

# b)
library(mixtools)
model= mvnormalmixEM(mydata[1:2],k=3,epsilon=1e-04)
model$mu
model$sigma
pi=model$lambda
pi

# c)
plot(model,which=2)

