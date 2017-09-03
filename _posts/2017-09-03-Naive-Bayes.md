---
layout: post
title: "Naive Bayes"
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

When the input features $x$ are discrete-valued random variables, use Naive Bayes classifier.

Consider building an email spam filter. Represent an email via a feature vector whose length $n$ is equal to the number of words in the dictionary. Specifically, if an email contains the $i$-th word of the dictionary, set $x_i = 1$; otherwise, $x_i = 0$.

To model $p(x | y)$, assume that the $x_i$’s are conditionally independent given $y$ which means $p(x_i | y) = p(x_i | y, x_j)$ that is not the same as independent like $p(x_i) = p(x_i | x_j)$.

Thus,

$$
\begin{aligned}
p(x_1, x_2, \cdots, x_n | y) &=
p(x_1 | y) P(x_2 | y, x1) \cdots p(x_n | y, x_1, x_2, \cdots, x_{n - 1}) \\\\
&= p(x_1 | y) p(x_2 | y) \cdots p(x_n | y) \\\\
&= \prod_{i = 1}^m p(x_i | y) \\\\
\end{aligned}
$$

Parameterized model by $\phi_{i | y = 1} = p(x_i = 1 | y = 1)$, $\phi_{i | y = 0} = p(x_i = 1 | y = 0)$, and $\phi_y = p(y = 1)$. Therefore the joint likelihood function is:

$$
\begin{aligned}
\mathcal{L}(\phi_y, \phi_{j | y = 0}, \phi_{j | y = 1}) &= \prod_{i = 1}^m p(x^{(i)}, y^{(i)}) \\
&= \prod_{i = 1}^m ((\prod_{j = 1}^{n} p(x_j^{(i)} | y^{(i)}; \phi_{j | y = 0}, \phi_{j | y = 1}) ) p(y^{(i)}; \phi_y))
\end{aligned}
$$

Maximizing this gives the maximum likelihood estimates:

$$
\phi_{j | y = 1} = \frac{\sum_{i = 1}^m 1 \{ x_j^{(i)} = 1 \land y^{(i)} = 1 \}}{\sum_{i = 1}^m 1 \{ y^{(i)} = 1 \}} \\\\
\phi_{j | y = 0} = \frac{\sum_{i = 1}^m 1 \{ x_j^{(i)} = 1 \land y^{(i)} = 0 \}}{\sum_{i = 1}^m 1 \{ y^{(i)} = 0 \}} \\\\
\phi_{y} = \frac{\sum_{i = 1}^m 1 \{ y^{(i)} = 1 \}}{m} \\\\
$$

Thus, p(y = 1 | x) can be derived as a function of x with parameters above:

$$
\begin{aligned}
p(y = 1 | x) &= \frac{p(x | y = 1) p(y = 1)}{p(x)} \\\\
&= \frac{(\prod_{i = 1}^n p(x_i | y = 1))p(y = 1)}
{(\prod_{i = 1}^n p(x_i | y = 1))p(y = 1) + (\prod_{i = 1}^n p(x_i | y = 0))p(y = 0)} \\\\
\end{aligned}
$$

Note that even if some original input attribute were continuous valued, it is common to discretetize it by turning into a small set of discrete values and apply Naive Bayes.

**Lapalce Smoothing**
Consider the possibility that $x_i$ is not included in any training examples, thus

$$
\phi_{x_i | y = 1} = 0 \\\\
\phi_{x_i | y = 0} = 0 \\\\
$$

Hence, $p(y = 1 | x) = \frac{0}{0}$ and can not make a prediction.

To avoid this, we can use Laplace smoothing: add 1 to the numerator and k to the denominator where k is the number of discrete values that $y$ can take. Therefore, obtain the following estimates of the parameters:

$$
\phi_{j | y = 1} = \frac{\sum_{i = 1}^m 1 \{ x_j^{(i)} = 1 \land y^{(i)} = 1 \} + 1}{\sum_{i = 1}^m 1 \{ y^{(i)} = 1 \} + 2} \\\\
\phi_{j | y = 0} = \frac{\sum_{i = 1}^m 1 \{ x_j^{(i)} = 1 \land y^{(i)} = 0 \} + 1}{\sum_{i = 1}^m 1 \{ y^{(i)} = 0 \} + 2} \\\\
$$

so that a dirable property still holds:

$$
1 - \phi_{j | y = 1} =
\frac{\sum_{i = 1}^m 1 \{ x_j^{(i)} = 0 \land y^{(i)} = 1 \} + 1}{\sum_{i = 1}^m 1 \{ y^{(i)} = 1 \} + 2} \\\\
1 - \phi_{j | y = 0} =
\frac{\sum_{i = 1}^m 1 \{ x_j^{(i)} = 0 \land y^{(i)} = 0 \} + 1}{\sum_{i = 1}^m 1 \{ y^{(i)} = 0 \} + 2} \\\\
$$