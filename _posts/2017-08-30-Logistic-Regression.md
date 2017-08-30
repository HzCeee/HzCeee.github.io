---
layout: post
title: "Logistic Regression"
author: "HzCeee"
categories: Machine Learning
tags: [ML Algorithm]
image:
  feature: 
  teaser: 
  credit:
  creditlink:
---

For the binary classification problem where $y \in \{ 0 , 1 \}$, use logistic regression ignoring the fact that $y$ is discrete-valued:

$$
h_{\theta}(x) = g(\theta^T x) = \frac{1}{1 + e^{(- \theta^T x)}}
$$

where

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

Then the binary classfication problem can be solved by evaluating the probability of each discrete value with:

$$
P(y = 1 | x ; \theta) = h_{\theta}(x) \\\\
P(y = 0 | x ; \theta) = 1 - h_{\theta}(x)
$$

which can be written more compactly as

$$
p(y | x ; \theta) = (h_{\theta}(x))^y (1 - h_{\theta}(x)^{1-y})
$$

which means 

$$
y \sim Bernoulli( h_{\theta}(x) )
$$

Thus, the likelihood of the parameters is:

$$
\begin{aligned}
L(\theta) &= p(\overrightarrow{y} | X ; \theta) \\\\
&= \prod_{i = 1}^m p(y^{(i)} | x^{(i)} ; \theta) \\\\
&= \prod_{i = 1}^m (h_{\theta}(x^{(i)}))^{y^{(i)}}(1 - h_{\theta}(x^{(i)}))^{1 - y^{(i)}}
\end{aligned}
$$

Therefore the log likelihood function is:

$$
\begin{aligned}
l(\theta) &= \log L(\theta) \\\\
&= \sum_{i = 1}^m \left [ y^{(i)} \log h(x^{(i)}) + (1 - y^{(i)}) \log (1 - h(x^{(i)}) \right ]
\end{aligned}
$$

Then, we can define the cost function:

$$
\begin{aligned}
J(\theta) &= - l(\theta) \\\\
&= - \sum_{i = 1}^m \left [ y^{(i)} \log h(x^{(i)}) + (1 - y^{(i)}) \log (1 - h(x^{(i)}) \right ]
\end{aligned}
$$

Starts with some initial $\theta$, and repeatedly performs the update to maximize log likelihood function $l(\theta)$:

$$
\theta := \theta + \alpha \nabla_{\theta} l(\theta)
$$

Take derivatives with one training example $(x, y)$ to derive the stochastic gradient ascent rule:

$$
\begin{aligned}
\frac{\partial}{\partial \theta_j} l(\theta) &= 
(y \frac{1}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)})\frac{\partial}{\partial \theta_j}g(\theta^T x) \\\\
&= (y \frac{1}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)}) g(\theta^T x)(1 - g(\theta^T x)) \frac{\partial}{\partial \theta_j} \theta^T x \\\\
&= (y (1 - g(\theta^T x)) - (1 - y) g (\theta^T x)) x_j \\\\
&= (y - h_{\theta}(x)) x_j \\\\
\end{aligned}
$$

This therefore gives us the stochatic gradient ascent rule:

$$
\begin{aligned}
\theta_j &:= \theta_j + \alpha (y^{(i)} - h_{\theta}(x^{(i)})) x_j^{(i)} \\\\
&= \theta_j + \alpha (y^{(i)} - \frac{1}{1 + e^{- \theta^T x}}) x^{(i)}_j \\\\
\end{aligned}
$$