---
layout: post
title: "Linear Regression"
author: "HzCeee"
categories: Machine Learning
tags: [ML Algorithm]
image:
  feature: 
  teaser: 
  credit:
  creditlink:
---

Given a training set with $m$ training examples $\{ (x^{(i)}, y^{(i)}); i = 1, \cdots , m \}$ where $x^{(i)} = (x_1^{(i)}, x_2^{(i)}, \cdots , x_n^{(i)})^T$ and $y^{(i)}$ denotes the target variable, to learn a function $h : \mathcal{X} \mapsto \mathcal{Y}$ so that $h(x)$ is a good predictor for the corresponding value of $y$.

Approximate $y$ as a linear function of $x$, then

$$
h_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n
$$

To simplify the notation, let $h_{\theta}(x) = h(x)$ and $\theta$ be the $n + 1$-dimensional vector as $\theta = (\theta_0, \theta_1, \cdots, \theta_n)^T$ and introduce the intercept term $x_0 = 1$, so that

$$
h(x) = \sum_{i = 0}^{n} \theta_i x_i = \theta^T x \\\\
x = (x_0, x_1, \cdots, x_n)^T
$$

where $n$ is the number of features (not counting $x_0$).

Define the design matrix $X$ to be $m$-by-$n + 1$:

$$
X =
\begin{bmatrix}
x^{(1)}_0 & x^{(1)}_1 & \cdots & x^{(1)}_n \\\\
x^{(2)}_0 & x^{(2)}_1 & \cdots & x^{(2)}_n \\\\
\vdots \\\\
x^{(m)}_0 & x^{(m)}_1 & \cdots & x^{(m)}_n \\\\
\end{bmatrix}
=
\begin{bmatrix}
(x^{(1)})^T \\\\
(x^{(2)})^T \\\\
\vdots \\\\
(x^{(m)})^T \\\\
\end{bmatrix}
$$

Also, let $\overrightarrow{y}$ be the $m$-dimensional vector:

$$
\overrightarrow{y} = 
\begin{bmatrix}
y^{(1)} \\\\
y^{(2)} \\\\
\vdots \\\\
y^{(m)} \\\\
\end{bmatrix}
$$

Assume

$$
y^{(i)} = \theta^T x^{(i)} + \varepsilon^{(i)}
$$

and $\varepsilon^{(i)}$ are distributed IID (independently and identically distributed) according to a normal distribution as $\varepsilon^{(i)} \sim \mathcal{N}(0, \sigma^2)$. Thus,

$$
p(\varepsilon{(i)}) = \frac{1}{\sqrt{2 \pi} \sigma} exp (- \frac{(\varepsilon^{(i)})^2}{2 \sigma^2})
$$

This implies that

$$
p(y^{(i)}|x^{(i)}; \theta) = \frac{1}{\sqrt{2 \pi} \sigma} exp (- \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 \sigma^2})
$$

Thus the likelihood function $L(\theta) = L(\theta; X,\vec{y}) = p(\vec{y} \| X; \theta)$ can be written as

$$
\begin{aligned}
L(\theta) &= \prod^m_{i = 1} p(y^{(i)} | x^{(i)}; \theta) \\\\
&= \prod^m_{i = 1} \frac{1}{\sqrt{2 \pi} \sigma} exp (- \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 \sigma^2})
\end{aligned}
$$

To simplize the calculatioin, instead of maximizeing $L(\theta)$, maximize the log likelihood $l(\theta)$:

$$
\begin{aligned}
l(\theta) &= \log L(\theta) \\\\
&= \log \prod_{i = 1}^m \frac{1}{\sqrt{2 \pi} \sigma} exp (- \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 \sigma^2}) \\\\
&= \sum_{i = 1}^m \log \frac{1}{\sqrt{2 \pi} \sigma} exp (- \frac{(y^{(i)} - \theta^T x^{(i)})^2}{2 \sigma^2}) \\\\
&= m \log \frac{1}{\sqrt{2 \pi} \sigma} - \frac{1}{\sigma^2} \frac{1}{2} \sum_{i = 1}^m (y^{(i)} - \theta^T x^{(i)})^2
\end{aligned}
$$

Hence, maximizing $l(\theta)$ gives the same answer as minimizing $\frac{1}{2} \sum_{i = 1}^m (y^{(i)} - \theta^T x^{(i)})^2$ which is known as least mean sequares function.

Then, we can define the cost function:

$$
\begin{aligned}
J(\theta) &= \frac{1}{2} \sum_{i = 1}^m (y^{(i)} - \theta^T x^{(i)})^2 \\\\
&= \frac{1}{2} \sum_{i = 1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2
\end{aligned}
$$

**1. Gradient Descent Algorithm**

Starts with some initial $\theta$, and repeatedly performs the update to minimize $J(\theta)$:

$$
\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$

where $\alpha$ is the learning rate. Here $J(\theta) = \frac{1}{2} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2$, so that

$$
\begin{aligned}
\frac{\partial}{\partial \theta_j} J(\theta) &= \frac{\partial}{\partial \theta_j} \frac{1}{2} \sum_{i = 1}^{m} (h_{\theta}(x^{(i)}) - y^{(i)})^2 \\\\
&= \frac{1}{2} \sum_{i = 1}^{m} \frac{\partial}{\partial \theta_j} (\sum_{k = 0}^{n} \theta_k x_k^{(i)} - y^{(i)})^2 \\\\
&= \frac{1}{2} \sum_{i = 1}^{m} (2 \cdot (\sum_{k = 0}^{n} \theta_k x_k^{(i)} - y^{(i)}) \cdot x_j^{(i)}) \\\\
&= \sum_{i = 1}^{m} ((h_{\theta}(x^{(i)}) - y^{(i)}) \cdot x_j^{(i)})
\end{aligned}
$$

this gives the update rule (Widrow-Hoff Laerning Rule or LMS Update Rule):

$$
\begin{aligned}
\theta_j &:= \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) \\\\
&= \theta_j + \alpha \sum_{i = 1}^{m} ((y^{(i)} - (h_{\theta}(x^{(i)})) \cdot x_j^{(i)})
\end{aligned}
$$

**2. The Normal Equations**

This method performs the minimization explicitly and without resorting to an iterative algorithm.

Since $h_{\theta}(x^{(i)}) = \sum_{j = 0}^{n} \theta_j x_j^{(i)} = (x^{(i)})^T \theta$, so

$$
\begin{aligned}
X \theta - \overrightarrow{y} &= 
\begin{bmatrix}
(x^{(1)})^T \theta \\\\
(x^{(2)})^T \theta \\\\
\vdots \\\\
(x^{(m)})^T \theta \\\\
\end{bmatrix}
-
\begin{bmatrix}
y^{(1)} \\\\
y^{(2)} \\\\
\vdots \\\\
y^{(m)} \\\\
\end{bmatrix} \\\\
&=
\begin{bmatrix}
h_{\theta}(x^{(1)}) - y^{(1)} \\\\
h_{\theta}(x^{(2)}) - y^{(2)} \\\\
\vdots \\\\
h_{\theta}(x^{(m)}) - y^{(m)} \\\\
\end{bmatrix}
\end{aligned}
$$

Thus, using the fact that $z^T z = \sum_i z_i^2$:

$$
\begin{aligned}
\frac{1}{2}(X \theta - \overrightarrow{y})^T(X \theta - \overrightarrow{y}) 
&= \frac{1}{2} \sum_{i = 1}^m (h_{\theta}(x^{(i)}) - y^{(i)})^2 \\\\
&= J(\theta)
\end{aligned}
$$

Hence, 

$$
\begin{aligned}
\nabla_{\theta}J(\theta) &=
\nabla_{\theta} \frac{1}{2} (X \theta - \overrightarrow{y})^T(X \theta - \overrightarrow{y}) \\\\
&= \frac{1}{2} \nabla_{\theta}(\theta^T X^T X \theta - \theta^T X^T \overrightarrow{y} - \overrightarrow{y} X \theta + \overrightarrow{y}^T \overrightarrow{y}) \\\\
&= \frac{1}{2} \nabla_{\theta} tr(\theta^T X^T X \theta - \theta^T X^T \overrightarrow{y} - \overrightarrow{y} X \theta + \overrightarrow{y}^T \overrightarrow{y}) \\\\
&= \frac{1}{2} \nabla_{\theta} (tr \theta^T X^T X \theta - 2 tr \overrightarrow{y}^T X \theta) \\\\
&= \frac{1}{2} (X^T X \theta + X^T X \theta - 2 X^T \overrightarrow{y}) \\\\
&= X^T X \theta - X^T \overrightarrow{y}
\end{aligned}
$$

To minimize $J(\theta)$, set $\nabla_{\theta}J(\theta) = 0$, thus

$$
\theta = (X^T X)^{-1} X^T \overrightarrow{y}
$$
