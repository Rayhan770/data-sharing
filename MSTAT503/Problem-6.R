# Problem-6 #
# Generate three-component Gaussian mixture data and use
# the generated data set for estimating the parameters
# of the model by EM algorithm

rm(list=ls())
# Generate three-component Gaussian mixture data
N= 100000
# Sample N random uniforms U
U=runif(N)
# gen.y Variable to store the samples from the mixture distribution
gen.y= rep(NA,N)
# True parameters
p= c(0.3,0.5,0.2)
m= c(0,10,3)
s= c(1,1,0.1)

# Sampling from the mixture
for(i in 1:N)
{
  if(U[i] < p[1])
  {
    gen.y[i]= rnorm(1,m[1],s[1])
  }
  else if(U[i] < p[1] + p[2]){
    gen.y[i] = rnorm(1,m[2],s[2])
  } 
  else{
  gen.y[i] = rnorm(1,m[3],s[3])  
  }
}

library(mixtools)
gm3= normalmixEM(gen.y, k=3, lambda=c(p),mu=c(m),sigma=c(s))
gm3$lambda
gm3$mu
gm3$sigma
gm3$loglik

# plotting the true and estimated densities as for check
x=seq(min(gen.y),max(gen.y),0.1)
true.pdf= p[1]*dnorm(x,m[1],s[1]) +
  p[2]*dnorm(x,m[2],s[2]) + 
  p[3]*dnorm(x,m[3],s[3])
est.pdf= gm3$lambda[1]*dnorm(x,gm3$mu[1],gm3$sigma[2]) + 
  gm3$lambda[2]*dnorm(x,gm3$mu[2],gm3$sigma[2]) + 
  gm3$lambda[3]*dnorm(x,gm3$mu[3],gm3$sigma[3])

plot(x,true.pdf,main="Density Estimated of the Mixture Model",
     col= "black",lwd=2)
lines(x,est.pdf,col="red",lwd=2)
legend("topright",c("True Density","Estimated Density"),
       col=c("black","red"),lwd=2)

