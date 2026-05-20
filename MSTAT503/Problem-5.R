# Problem-5 #
# Mixture Models #
# Two component Gaussian mixture models fitting by EM algorithm #

rm(list=ls())

y=c(-0.39,0.12,0.94,1.67,1.76,2.44,3.72,4.28,4.92,5.53,
    0.06,0.48,1.01,1.80,3.25,4.12,4.60,5.28,6.22)
n=length(y)

hist(y,breaks=8,main="Twenty fictitious data")

# initial values
pi1=0.5
pi2=0.5
mu1=1
mu2=3
sigma1=0.8
sigma2=0.8

tiny=10^(-7)  # a small value to test convergence

loglik.old=0
mix.pdf=pi1*dnorm(y,mu1,sigma1)+pi2*dnorm(y,mu2,sigma2)
loglik.new=sum(log(mix.pdf))
k=1

# Loop for EM-iteration
while(abs(loglik.new-loglik.old) >= tiny)
{
  # E-step
  z1j= pi1*dnorm(y,mu1,sigma1)/
    (pi1*dnorm(y,mu1,sigma1)+pi2*dnorm(y,mu2,sigma2))
  z2j= pi2*dnorm(y,mu2,sigma2)/
    (pi1*dnorm(y,mu1,sigma1)+ pi2*dnorm(y,mu2,sigma2))
  
  # M-step
  pi1= sum(z1j)/n
  pi2= sum(z2j)/n
  mu1= sum(z1j*y)/sum(z1j)
  mu2= sum(z2j*y)/sum(z2j)
  sigma1= sqrt(sum(z1j*(y-mu1)^2)/sum(z1j))
  sigma2= sqrt(sum(z2j*(y-mu2)^2)/sum(z2j))
  
  # Log-likelihood
  loglik.old= loglik.new
  mix.pdf= pi1*dnorm(y,mu1,sigma1) + pi2*dnorm(y,mu2,sigma2)
  loglik.new= sum(log(mix.pdf))
  k= k+1
  cat(c(k,pi1,pi2,mu1,mu2,sigma1,sigma2,loglik.new),"\n")
}

# Plotting fitted density
h=hist(y,breaks=8, main="Twenty fictitious data")

# Add a Normal Curve
ynew=seq(min(y),max(y),0.1)
obs.fit= dnorm(ynew,mean=mean(y),sd=sd(y))
obs.fit=obs.fit*diff(h$mids[1:2])*length(y)
lines(ynew,obs.fit,col="blue",lty=1,lwd=2)

# 2-fold Mixture density
mix.fit= pi1*dnorm(ynew,mu1,sigma1) +
  pi2*dnorm(ynew,mu2,sigma2)
mix.fit= mix.fit*diff(h$mids[1:2])*length(y)
lines(ynew,mix.fit,col="red",lty=2,lwd=2)

legend("topright",c("Without mixture density",
                    "2-fold Normal mixture density"),
       col=c("blue","red"),
       text.col=c("blue","red"),
       lty=c(1,2),lwd=2)


# Comparison with mixtools package-results
library(mixtools)
gm=normalmixEM(y,k=2,lambda=c(0.5,0.5),mu=c(1,3),sigma=c(0.8,0.8))
gm$lambda
gm$mu
gm$sigma
gm$loglik
plot(gm)

# Plotting
plot(gm,which=2)
lines(density(y),col="black",lty=2,lwd=2)

# 2-fold Mixture density
mix.fit= gm$lambda[1]*dnorm(ynew,gm$mu[1],gm$sigma[1]) +
  gm$lambda[2]*dnorm(ynew,gm$mu[2],gm$sigma[2])
lines(ynew,mix.fit,col="blue",lty=3,lwd=2)
legend("topright",c("Without mixture density",
                    "2-fold Normal mixture density"),
       col=c("black","blue"),
       text.col=c("black","blue"),
       lty=c(2,3),lwd=2)

