---
layout: post
title: Mixture-of-Gaussian clustering
categories: machine learning
---

First we should write a simple piece of code which can return the probability density of Gaussian distribution, or multivariate normal distribution.

{% highlight R %}
dmvnorm <- function(x,mu,sigma){
  sigma <- as.matrix(sigma)
  y <- 1/sqrt((2*pi)^(length(mu))*det(sigma))*exp(-t(x-mu)%*%solve(sigma)%*%(x-mu)/2)
  return(y)
}
{% endhighlight %}
