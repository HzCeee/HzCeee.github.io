---
layout: post
title: "Back Propagation Algorithm"
author: "HzCeee"
categories: Machine Learning
tags: [Nerual Network]
image:
  feature: 
  teaser: 
  credit:
  creditlink:
---

In logistic regression, the cost function without regularization term is:

$$
J(\theta) = \frac{1}{m} \sum_{i = 1}^m \left[ 
y^{(i)}\left(- \log \left( h_{\theta}(x^{(i)}) \right) \right)
+ (1 - y^{(i)}) \left(- \log \left(1 - h_{\theta}(x^{(i)} \right) \right)
\right]
$$

In neural network with sigmoid function as the active function in each neurons, the cost function can be expressed as:

$$
J(\Theta) = \frac{1}{m} \sum_{i = 1}^m \sum_{k = 1}^{S_L}
\left[ 
y^{(i)}_k \left(- \log \left( h_{\theta}(x^{(i)})_k \right) \right)
+ (1 - y^{(i)}_k) \left(- \log \left(1 - h_{\theta}(x^{(i)})_k \right) \right)
\right]
$$

where $m$ is the number of training set, $L$ is the number of layers in network and $S_L$ is the number of neurons in the layer $L$.

Define $\delta_i^{(l)} = \frac{\partial J(\Theta)}{\partial z_i^{(l)}} $, $\Delta_{ij}^l= \frac{\partial J(\Theta)}{\partial \Theta^{(l)}_{ij}} $, $z^{(l)}_i$ and $a^{(l)}_i = g(z^{(l)}_i)$ are the input and output of the neuron $i$ in layer $l$ respectively. Thus, to minimize $J(\Theta)$, use the gradient descent algorithm to update weights $\Theta$:

$$
\begin{aligned}
\Delta_{ij}^{(l)}
&= \frac{\partial J(\Theta)}{\partial \Theta^{(l)}_{ij}} \\\\
&= \frac{\partial J(\Theta)}{\partial z^{(l + 1)}_{i}} \frac{\partial z^{(l + 1)}_{i}}{\partial \Theta^{(l)}_{ij}} \\\\
&= \delta^{(l + 1)}_i a^{(l)}_j
\end{aligned}
$$

where $\delta_i^{(l)}$ can be computed as:

$$
\begin{aligned}
\delta_i^{(l)}
&= \frac{\partial J(\Theta)}{\partial z^{(l)}_{i}} \\\\
&= \frac{\partial J(\Theta)}{\partial a^{(l)}_i} \frac{\partial a^{(l)}_i}{\partial z^{(l)}_i} \\\\
&= \sum_{j = 1}^{S_{l + 1}} \left[\frac{\partial J(\Theta)}{\partial z^{(l + 1)}_j} \frac{\partial z^{(l + 1)}_j}{\partial a^{(l)}_i} \right] \cdot
\frac{\partial a^{(l)}_i}{\partial z^{(l)}_i} \\\\
&= \sum_{j = 1}^{S_{l + 1}} \left[ \delta_j^{(l + 1)} \Theta^{(l)}_{ji} \right] \cdot g'(z^{(l)}_i) \\\\
\end{aligned}
$$

By vectorization using vector $\delta^{(l)}​$ and $z^{(l)}​$ and matrix $\Theta^{(l)}​$:

$$
\delta^{(l)} = (\Theta^{(l)})^T \delta^{(l + 1)} \odot g'(z^{(l)})
$$

where operator $\odot$ is defined as:

$$
A \odot B = C \\
C_{ij} = A_{ij}B_{ij}
$$

Thus,

$$
\Delta^{(l)} = \delta^{(l + 1)} (a^{(l)})^T
$$

and $\delta^{(l)}$ can be computed backwards from $\delta^{(L)}$ by $\delta^{(l)} = (\Theta^{(l)})^T \delta^{(l + 1)} \odot g'(z^{(l)})$.
