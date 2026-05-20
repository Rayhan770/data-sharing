# Problem-7 #
# Generate multivariate Gaussian mixture data and use the
# generated data set for estimating the parameters of the
# model by EM algorithm

rm(list=ls())


gen.mix= function(n,k,mu,sigma)
{
  library(MASS)
  d=length(mu[1,])  # number of dimensions
  result= matrix(rep(NA,n*d),ncol=d)
  colnames(result)= paste0("X",1:d)
  
  for(i in 1:n)
  {
    result[i,] = mvrnorm(1,mu= mu[k[i],],Sigma = sigs[,,k[i]])
  }
  result
}


set.seed(186)
n= 360

mu= matrix(c(4,4,5,5,6.5,5),ncol= 2, byrow= TRUE)

sigs= array(rep(NA,2*2*3),c(2,2,3))  # 3D matrix
sigs[,,1]= matrix(c(0.25, 0.25, 0.21, 0.25), nrow=2, byrow= TRUE)
sigs[,,2]= matrix(c(0.25,-0.21,0.21,0.25),nrow=2,byrow= TRUE)
sigs[,,3]= matrix(c(0.25,0.21,0.21,0.25),nrow=2,byrow= TRUE)

pi= c(0.2,0.5,0.3)  # mixing coefficients

classes= sample(1:3,n,replace= TRUE, prob= pi)

mydata= gen.mix(n,classes,mu,sigs)

# mydata= read.csv("mydata.csv",as.is= TRUE)

plot(mydata,col=c("red","green","blue")[classes],
     xlab="X1",ylab="X2", pch = 19)

library(mixtools)
model= mvnormalmixEM(mydata,k=3,epsilon=1e-04)
model$mu
model$sigma
plot(model,which=2)
