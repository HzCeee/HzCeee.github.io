---
layout: post
title: "The Gaussian Discriminant Analysis Model"
author: "HzCeee"
categories: Machine Learning
tags: [Generative Learning Algorithm]
image:
  feature: 
  teaser: 
  credit:
  creditlink:
---

Model $p(y | x)$ indirectly by modelling $p(x | y)$ and applying Bayes Rule:

$$
\begin{aligned}
p(y | x) &= \frac{p(x | y) p(y)}{p(x)} \\\\
&= \frac{p(x | y) p(y)}{p(x | y = 1)p(y = 1) + p(x | y = 0)p(y = 0)} \\\\
\end{aligned}
$$

## The Gaussian Discriminant Analysis Model

When the input features $x$ are continues-valued random variables, use the model:

$$
y \sim Bernoulli(\phi) \\\\
x | y = 0 \sim \mathcal{N}(\mu_0 , \Sigma) \\\\
x | y = 1 \sim \mathcal{N}(\mu_1, \Sigma) \\\\
$$

Specifically,

$$
p(y) = \phi^y (1 - \phi)^{1 - y} \\\\
p(x | y = 0) = \frac{1}{(2 \pi)^{n / 2} | \Sigma |^{1 / 2}} exp(- \frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0)) \\\\
p(x | y = 1) = \frac{1}{(2 \pi)^{n / 2} | \Sigma |^{1 / 2}} exp(- \frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1))
$$

The log-likelihood function is given by:

$$
\begin{aligned}
l(\phi, \mu_0, \mu_1, \Sigma) &= \log \prod_{i = 1}^m p(x^{(i)}, y^{(i)}; \phi, \mu_0, \mu_1, \Sigma) \\\\
&= \log \prod_{i = 1}^m p(x^{(i)} | y^{(i)}; \mu_0, \mu_1, \Sigma) p(y^{(i)}; \phi) \\\\
\end{aligned}
$$

Estimate parameters by maximizing $l(\phi, \mu_0, \mu_1, \Sigma)$:

$$
\phi = \frac{1}{m} \sum_{i = 1}^m 1 \{ y^{(i)} = 1 \} \\\\
\mu_0 = \frac{\sum_{i = 1}^m 1\{y^{(i)} = 0\} x^{(i)}}{\sum_{i = 1}^m 1 \{ y^{(i)} = 0\}} \\\\
\mu_1 = \frac{\sum_{i = 1}^m 1\{ y^{(i)} = 1 \} x^{(i)}}{\sum_{i = 1}^m 1 \{ y^{(i)} = 1\} } \\\\
\Sigma = \frac{1}{m} \sum_{i = 1}^m(x^{(i)} - \mu_{y^{(i)}})(x^{(i)} - \mu_{y^{(i)}})^T
$$

Thus, $p(y = 1 | x)$ can be derived as a function of $x$ with parameters above:

$$
\begin{aligned}
p(y = 1 | x) &= \frac{p(x | y = 1) p(y = 1)}{p(x | y = 1) p(y = 1) + p(x | y = 0) p(y = 0)} \\\\
&= \frac{\frac{1}{(2 \pi)^{n / 2} | \Sigma |^{1 / 2}} exp(- \frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1)) \cdot \phi}
{\frac{1}{(2 \pi)^{n / 2} | \Sigma |^{1 / 2}} exp(- \frac{1}{2} (x - \mu_1)^T \Sigma^{-1} (x - \mu_1)) \cdot \phi + \frac{1}{(2 \pi)^{n / 2} | \Sigma |^{1 / 2}} exp(- \frac{1}{2} (x - \mu_0)^T \Sigma^{-1} (x - \mu_0)) \cdot (1 - \phi)} \\\\
\end{aligned}
$$

The decision boundary can be given at which $p(y = 1 | x) = 0.5$.
