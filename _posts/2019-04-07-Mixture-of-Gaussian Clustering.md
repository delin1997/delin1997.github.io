---
layout: post
title: Mixture-of-Gaussian Clustering——R realization
categories: machine_learning

---

# Mixture-of-Gaussian Clustering

Author: Delin Zhao

---

### Algorithm
![GMM algorithm](/assets/GMM.png)

---

### Probability density of Gaussian distribution:

**dmvnorm.R**
{% highlight R %}
dmvnorm <- function(x,mu,sigma){
  sigma <- as.matrix(sigma)
  y <- 1/sqrt((2*pi)^(length(mu))*det(sigma))*exp(-t(x-mu)%*%solve(sigma)%*%(x-mu)/2)
  return(y)
}
{% endhighlight %}

---

### GMM-clustering:

**Mix_Gaussian_cluster.R**
Mixture_of_Gaussian_clustering <- function(k,D,mu.start=mu,sigma.start=sigma,alpha.start=alpha,iter=10,plot=TRUE){
  # k is the number of mixture components.
  # D is the dataset.
  # mu is a matrix in which the first column is the mean vector of the first Gaussian distribution 
  # and the second column corresponds to the second Gaussian distribution, etc.
  # By default, the initial mu will be made up of the first k rows of the data.
  # sigma is a three-dimension array in which the first matrix is the sigma matrix of the first Guassian 
  # distribution and ect.
  # By default, each matrix of the initial sigma is set as a diagonal matrix with each diagonal element 0.1.
  # alpha is the mixture coefficient and it's a k*1 vector with each element 1/k by default.
  # iter is the number of iteration and it's 10 by default.
  # this code will by default plot a graph of clustering results as long as the data has only two variables.
  source("dmvnorm.R")
  D <- as.matrix(D)
  alpha <- rep(1/k,k)
  mu <- matrix(c(D[6,],D[22,],D[27,]),nrow = 2)
  #mu <- t(D[1:k,])
  n <- ncol(D)
  m <- nrow(D)
  sigma <- array(rep(diag(rep(0.1,n),n),k),dim = c(n,n,k))
  gamma <- matrix(0,ncol = k,nrow = m)
  t <- 0
  while(t<iter){
    for(j in 1:m){
      for(i in 1:k){
        gamma[j,i] <- alpha[i]*dmvnorm(D[j,],mu[,i],sigma[,,i])
      }
      gamma[j,] <- gamma[j,]/sum(gamma[j,])
    }
    for(i in 1:k){
      mu[,i] <- t(t(gamma[,i])%*%D)/sum(gamma[,i])
      S <- matrix(0,nrow = n,ncol = n)
      for(j in 1:m){
        S <- S+gamma[j,i]*(D[j,]-mu[,i])%*%t(D[j,]-mu[,i])
      }
      sigma[,,i] <- S/sum(gamma[,i])
      alpha[i] <- sum(gamma[,i])/m
    }
    t <- t+1
  }
  tag_cl <- matrix(0,nrow = m,ncol = 1)
  for(j in 1:m){
    tag_cl[j] <- which.max(gamma[j,])
  }
  if(plot & n==2){
    par(mar = c(5,5,1,1))
    plot(NA,xlab = "x",ylab = "y",xlim = c(min(D[,1]),1.2*max(D[,1])), ylim = c(min(D[,2]),1.2*max(D[,2])),
      sub=paste("iterate",iter,"times"))
    points(D[,1],D[,2],type="p",pch=tag_cl)
    # points in different groups will have different shapes, one group one shape 
    center.x <- vector(length = k,mode = "numeric")
    center.y <- vector(length = k,mode = "numeric")
    for(i in 1:k){
      center.x[i] <- mean(D[which(tag_cl==i),1])
      center.y[i] <- mean(D[which(tag_cl==i),2])
    }
    points(center.x,center.y,type="p",pch=4,col = "blue")
    # mark the center of each group
    points(mu[1,],mu[2,],pch=8,col = "red")
    # mark the mean of each Gaussian distribution
  }
  return(tag_cl)
}
{% endhighlight %}

---

### Use watermelon dataset 4.0 as an example:

{% highlight R %}
watermalon <- read.table("watermalon.txt",sep = ",",header=T)
source("Mix_Gaussian_cluster.R",encoding = "utf-8")
par(mfrow=c(2,2))
iter <- matrix(c(5,10,20,50),nrow = 4)
f <- function(x){
  Mixture_of_Gaussian_clustering(k=3,D=watermalon[,2:3],iter = x,plot = TRUE)
}
tag_cl <- apply(iter, 1, f)
{% endhighlight %}

---

### Clustering results:
![Clustering results](/assets/result.png)

---

### Reference
《机器学习》周志华著.–北京：清华大学出版社
