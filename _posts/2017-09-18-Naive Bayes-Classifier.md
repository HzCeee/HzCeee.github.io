---
layout: post
title: "Naive Bayes Classifier"
author: "HzCeee"
categories: Machine Learning
tags: [Generative Learning Algorithm]
image:
  feature: 
  teaser: 
  credit:
  creditlink:
---

__Naive Bayes Classifier__

Classifier $\hat{f}(\boldsymbol{x})$:

$$
\begin{aligned}
\hat{f}(\boldsymbol{x})
& = \mathop{\arg \max}_{y \in \mathcal{Y}} P[Y = y | X = \boldsymbol{x}] \\
& = \mathop{\arg \max}_{y \in \mathcal{Y}} \frac{P[X = \boldsymbol{x} | Y = y] \cdot P[Y = y]}{P[X = \boldsymbol{x}]} \\
& = \mathop{\arg \max}_{y \in \mathcal{Y}} P[X = \boldsymbol{x} | Y = y] \cdot P[Y = y] \\
\end{aligned}
$$

By applying Bayes Rule, model $p(Y = y \|X = \boldsymbol{x})$ indirectly by modelling $P[X = \boldsymbol{x} \| Y = y]$.

Model the probabilty of $\boldsymbol{x}$ with parameter setting $\theta$ and let $P(\boldsymbol{x} \| \theta) = p_{\theta}(\boldsymbol{x})$.

The likelihood function is given by:

$$
\mathcal{L}(\theta | \mathcal{X}) = P(\mathcal{X} | \theta) = P(\boldsymbol{x}_1, \boldsymbol{x}_2, \cdots, \boldsymbol{x}_n) = \prod_{i = 1}^n P(\boldsymbol{x}_i | \theta) = \prod_{i = 1}^n p_{\theta}(\boldsymbol{x}_i)
$$

Then parameter setting $\theta$ can be chosen by:

$$
\theta = \mathop{\arg \max}_{\theta} \mathcal{L}(\theta | \mathcal{X}) = \mathop{\arg \max}_{\theta} \prod_{i = 1}^n p_{\theta}(\boldsymbol{x}_i)
$$

For example, model $\boldsymbol{x} \| y = i$ with Guassian Model:

$$
p_{\mu,\sigma^2}(\boldsymbol{x}) = \frac{1}{\sqrt{2 \pi \sigma^2}} exp \left( - \frac{(\boldsymbol{x} - \mu)^2}{2 \sigma^2} \right)
$$

Then the parameter $\theta = \\{ \mu, \sigma^2 \\}$ can be computed as:

$$
\begin{aligned}
\mathop{\arg \max}_{\mu, \sigma^2} \mathcal{L} (\theta | \mathcal{X})
& = \mathop{\arg \max}_{\mu, \sigma^2} \log \mathcal{L} (\theta | \mathcal{X}) \\
& = \mathop{\arg \max}_{\mu, \sigma^2} \log (\prod_{i = 1}^n p_{\mu, \sigma^2}(\boldsymbol{x}_i)) \\
& = \mathop{\arg \max}_{\mu, \sigma^2} \sum_{i = 1}^n \log (p_{\mu, \sigma^2}(\boldsymbol{x}_i)) \\
& = \mathop{\arg \max}_{\mu, \sigma^2} \sum_{i = 1}^n \left[ -\frac{1}{2} \log(2 \pi \sigma^2) - \frac{(\boldsymbol{x_i} - \mu)^2}{2 \sigma^2} \right]
\end{aligned}
$$

To compute $\theta = \\{ \mu, \sigma^2 \\}$, let $g_i(\mu, \sigma^2) = -\frac{1}{2} \log(2 \pi \sigma^2) - \frac{(\boldsymbol{x_i} - \mu)^2}{2 \sigma^2}$:

$$
\begin{aligned}
0
& = \nabla_\mu \left(\sum_{i = 1}^n g_i(\mu, \sigma^2) \right) \\
& = \sum_{i = 1}^n \frac{\boldsymbol{x}_i - \mu}{\sigma^2} \\
& = \frac{\sum_{i = 1}^n \boldsymbol{x}_i - n \mu}{\sigma^2} \\
\end{aligned}
\\
\begin{aligned}
0
& = \nabla_{\sigma^2} \left(\sum_{i = 1}^n g_i(\mu, \sigma^2) \right) \\
& = \sum_{i = 1}^n \left( - \frac{1}{2} \frac{4 \pi \sigma}{2 \pi \sigma^2} + 2 \frac{(\boldsymbol{x}_i - \mu)^2}{2 \sigma^3} \right) \\
& = \sum_{i = 1}^n \left( - \frac{1}{\sigma} + \frac{(\boldsymbol{x}_i - \mu)^2}{\sigma^3} \right) \\
& = \frac{- n \sigma^2 + \sum_{i = 1}^n (\boldsymbol{x}_i - \mu)^2}{\sigma^3} \\
\end{aligned}
$$

Thus,

$$
\mu = \frac{1}{n} \sum_{i = 1}^n \boldsymbol{x}_i
\\
\sigma^2 =\frac{1}{n} \sum_{i = 1}^n (\boldsymbol{x}_i - \mu)^2
$$

Then use the parameter $\theta =\\{ \mu, \sigma^2 \\}$ and predict with classifier $\hat{f}(\boldsymbol{x}) = \mathop{\arg \max}_{y \in \mathcal{Y}} P_\theta [X = \boldsymbol{x} \| Y = y] \cdot P[Y = y] $ 

---

__Optimality of Bayes Classifier__

For any classifier $h$: (Take binary classification as an example)

$$
\begin{aligned}
P[h(\boldsymbol{x}) = y | X = \boldsymbol{x}]
& = P(h(\boldsymbol{x}) = 0, Y = 0 | X = \boldsymbol{x}) + P(h(\boldsymbol{x}) = 1, Y = 1 | X = \boldsymbol{x}) \\
& = 1[h(\boldsymbol{x}) = 1]P[Y = 1 | X = \boldsymbol{x}] + 1[h(\boldsymbol{x}) = 0]P[Y = 0 | X = \boldsymbol{x}] \\
& = 1[h(\boldsymbol{x}) = 1] \eta(\boldsymbol{x}) + 1[h(\boldsymbol{x}) = 0](1 - \eta(\boldsymbol{x}))
\end{aligned}
$$

So, for naive bayes classifier $f(\boldsymbol{x})$ and other classifier $g(\boldsymbol{x})$:

$$
\begin{aligned}
& \quad P[f(\boldsymbol{x}) = y | X = \boldsymbol{x}] - P[g(\boldsymbol{x}) = y | X = \boldsymbol{x}] \\
& = \eta(\boldsymbol{x})\left[ 1[f(\boldsymbol{x}) = 1] - 1[g(\boldsymbol{x}) = 1] \right] + (1 - \eta(\boldsymbol{x})) \left[ 1[f(\boldsymbol{x}) = 0] - 1[g(\boldsymbol{x}) = 0] \right] \\
& = (2 \eta(\boldsymbol{x}) - 1) \left[ 1[f(\boldsymbol{x}) = 1] - 1[g(\boldsymbol{x}) = 1] \right] \\
& \geq 0
\end{aligned}
$$
