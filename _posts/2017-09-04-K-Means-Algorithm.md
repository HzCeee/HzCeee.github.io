---
layout: post
title: "K-means Algorithm"
author: "HzCeee"
categories: Machine Learning
tags: [Unsupervised Learning]
image:
  feature: 
  teaser: 
  credit:
  creditlink:
---

The $k$-means clustering algorithm is as follows:

$$
\begin{aligned}
&Initialize \ cluster \ centroids \ \mu_1, \mu_2, \cdots, \mu_k \in \mathbb{R}^n \ randomly \\\\
&Repeat \ until \ convergence \ \{ \\\\
&\qquad for \ i \ from \ 1 \ to \ m  \ \{ \\\\
&\qquad \qquad c^{(i)} := arg \min_j ||x^{(i)} - \mu_j||^2 \\\\
&\qquad \} \\\\
&\qquad for \ j \ from \ 1 \ to \ k  \ \{ \\\\
&\qquad \qquad \mu_j := \frac{\sum_{i = 1}^m 1\{ c^{(i)} = j \} x^{(i)}}{\sum_{i = 1}^m 1\{ c^{(i)} = j \}} \\\\
&\qquad \} \\\\
&\}
\end{aligned}
$$

In the algorithm above, $k$ is the number of clusters we want to find.
