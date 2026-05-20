# Problem-10 #
# Cluster analysis of European Protein Consumption
# data using R.

rm(list=ls())

# Read data
food=read.csv(file.choose(),header=TRUE)
food
head(food)
tail(food)

dm= dist(food[,c("WhiteMeat","RedMeat")])

cs=hclust(dm,method="single")
plot(cs,labels=food$Country)
cc=hclust(dm,method="complete")
plot(cc,labels=food$Country)
ca=hclust(dm,method="average")
plot(ca,labels=food$Country)

par(mfrow=c(1,3)) # multiple plots in 1 row
plot(cs)
plot(ca)
plot(cc,labels = food$Country)
rect.hclust(cc,3)

food.new= food[,-1]
dm.new= dist(food.new,method= "euclidean")
ca.new=hclust(dm.new,method="average")
plot(ca.new)
plot(ca.new,labels=food$Country)
rect.hclust(ca.new,3)

# with clustering on all protein groups (p=9),
#   number of clusters to 7 (k=7)
set.seed(186) # to fix the random starting clusters
kmfit= kmeans(food[,-1],centers=7,nstart=10)

o=order(kmfit$cluster)  # list of cluster assignments
data.frame(food$Country[o],kmfit$cluster[o])

# Plotting cluster assignments on Red 
#   and White meat scatter plot.
plot(food$Red, food$White, type="n",
     xlim=c(3,19),xlab="Red Meat",
     ylab="White Meat")
text(x=food$Red, y=food$White, labels=food$Country,
     col=rainbow(7)[kmfit$cluster])


library(cluster)
clusplot(food[,-1],kmfit$cluster,main="2D representation 
         of the Cluster solution",color=TRUE,shade=TRUE,
         labels=2,lines=0)

table(food[,1],kmfit$cluster)


# Alternative: Applying EM algorithm to mixture of normal
# distributions
library(mixtools)
X1= food[,-1]
set.seed(1)
out2all= mvnormalmixEM(X1,arbvar=FALSE,k=2,epsilon=1e-02)
out2all
prob1= round(out2all$posterior[,1],digits=3)
prob2= round(out2all$posterior[,2],digits=3)
prob= round(out2all$posterior[,1])
data.frame(food$Country,prob1,prob2,prob)
o=order(prob)
data.frame(food$Country[o],prob[o])

