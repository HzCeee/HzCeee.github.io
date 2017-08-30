---
layout: post
title: "Back Propagation Algorithm"
author: "HzCeee"
categories: Machine Learning
tags: [Nerual Network]
image:
  feature: neural-network.jpg
  teaser: neural-network-teaser.jpg
  credit:
  creditlink:
---

In logistic regression, the cost function without regularization term is:

$$
J(\theta) = \frac{1}{m} \sum_{k = 1}^m 
\left[ 
y^{(k)}\left(- \log \left( h_{\theta}(x^{(k)}) \right) \right)
+ (1 - y^{(k)}) \left(- \log \left(1 - h_{\theta}(x^{(k)}) \right) \right)
\right]
$$

In neural network with sigmoid function as the active function in each neuron, the cost function can be expressed as:

$$
J(\Theta) = \frac{1}{m} \sum_{k = 1}^m \sum_{i = 1}^{S_L}
\left[ 
y^{(k)}_i \left(- \log \left( h_{\theta}(x^{(k)})_i \right) \right)
+ (1 - y^{(k)}_i) \left(- \log \left(1 - h_{\theta}(x^{(k)})_i \right) \right)
\right]
$$

where $m$ is the number of training set, $L$ is the number of layers in network and $S_l$ is the number of neurons in the layer $l$.

Define $z_i^{(l,k)}​$ and $a_i^{(l,k)} = g(z_i^{(l,k)})​$ are the input and output of the $i​$-th neuron in $l​$-th layer respectively with example $x^{(k)}​$, $\delta_i^{(l,k)} = \frac{\partial J(\Theta)}{\partial z_i^{(l,k)}}​$,  $\Delta_{ij}^{(l)}= \frac{\partial J(\Theta)}{\partial \Theta_{ij}^{(l)}}​$ where $\Theta_{ij}^{(l)}​$ is the weight between the $i​$-th neuron in $(l+1)​$-th layer and the $j​$-th neuron in $l​$-th layer. Thus, to minimize $J(\Theta)​$, use the gradient descent algorithm to update weights $\Theta​$:

$$
\begin{aligned}
\Delta_{ij}^{(l)}
&= \frac{\partial J(\Theta)}{\partial \Theta^{(l)}_{ij}} \\\\
&= \sum_{k = 1}^m \left[ \frac{\partial J(\Theta)}{\partial z^{(l + 1,k)}_{i}} \frac{\partial z^{(l + 1,k)}_{i}}{\partial \Theta^{(l)}_{ij}} \right] \\\\
&= \sum_{k = 1}^m \left[ \delta^{(l + 1,k)}_i a^{(l,k)}_j \right]
\end{aligned}
$$

where $\delta_i^{(l,k)}$ can be computed as:

$$
\begin{aligned}
\delta_i^{(l,k)}
&= \frac{\partial J(\Theta)}{\partial z^{(l,k)}_{i}} \\\\
&= \frac{\partial J(\Theta)}{\partial a^{(l,k)}_i} \frac{\partial a^{(l,k)}_i}{\partial z^{(l,k)}_i} \\\\
&= \sum_{j = 1}^{S_{l + 1}} \left[\frac{\partial J(\Theta)}{\partial z^{(l + 1,k)}_j} \frac{\partial z^{(l + 1,k)}_j}{\partial a^{(l,k)}_i} \right] \cdot
\frac{\partial a^{(l,k)}_i}{\partial z^{(l,k)}_i} \\\\
&= \sum_{j = 1}^{S_{l + 1}} \left[ \delta_j^{(l + 1,k)} \Theta^{(l)}_{ji} \right] \cdot g'(z^{(l,k)}_i) \\\\
\end{aligned}
$$

By vectorization using vector $\delta^{(l,k)} = (\delta_1^{(l,k)}, \delta_2^{(l,k)}, \cdots, \delta_{S_l}^{(l,k)})^T$ and $z^{(l,k)} = (z_1^{(l,k)}, z_1^{(l,k)}, \cdots, z_{S_l}^{(l,k)})^T$ and matrix $\Theta^{(l)}$:

$$
\Theta^{(l)} = 
\left[
\begin{matrix}
 \Theta_{11}^{(l)} & \Theta_{12}^{(l)} & \cdots & \Theta_{1 S_{l}}^{(l)} \\\\
 \Theta_{21}^{(l)} & \Theta_{22}^{(l)} & \cdots & \Theta_{2 S_{l}}^{(l)} \\\\
 \vdots & \vdots & \ddots & \vdots \\\\
 \Theta_{S_{l+1} 1}^{(l)} & \Theta_{S_{l+1} 2}^{(l)} & \cdots & \Theta_{S_{l+1} S_{l}}^{(l)} \\\\
\end{matrix}
\right]
$$

So that

$$
\delta^{(l,k)} = (\Theta^{(l)})^T \delta^{(l + 1,k)} \odot g'(z^{(l,k)})
$$

where operator $\odot$ is defined as:

$$
A \odot B = C \\
C_{ij} = A_{ij}B_{ij}
$$

Similarly, define vector $a^{(l,k)} = (a_1^{(l,k)}, a_2^{(l,k)}, \cdots, a_{S_l}^{(l,k)})^T$ and matrix $\Delta^{(l)}$:

$$
\Delta^{(l)} = 
\left[
\begin{matrix}
 \Delta_{11}^{(l)} & \Delta_{12}^{(l)} & \cdots & \Delta_{1 S_{l}}^{(l)} \\\\
 \Delta_{21}^{(l)} & \Delta_{22}^{(l)} & \cdots & \Delta_{2 S_{l}}^{(l)} \\\\
 \vdots & \vdots & \ddots & \vdots \\\\
 \Delta_{S_{l+1} 1}^{(l)} & \Delta_{S_{l+1} 2}^{(l)} & \cdots & \Delta_{S_{l+1} S_{l}}^{(l)} \\\\
\end{matrix}
\right]
$$

So that

$$
\Delta^{(l)} = \sum_{k = 1}^m \delta^{(l + 1,k)} (a^{(l,k)})^T
$$

and $\delta^{(l,k)}$ can be computed backwards from $\delta^{(L,k)}$ by $\delta^{(l,k)} = (\Theta^{(l)})^T \delta^{(l + 1,k)} \odot g'(z^{(l,k)})$.

Specifically, when $g(z)$ is a sigmoid function where

$$
g(z) = \frac{1}{1 + e^{-z}}
$$

so that

$$
\begin{aligned}
g'(z) &= - \frac{1}{(1 + e^{-z})^2} \cdot - e^{-z} \\\\
&= \frac{e^{-z} + 1 - 1}{(1 + e^{-z})^2} \\\\
&= \frac{1}{1 + e^{-z}} - \frac{1}{(1 + e^{-z})^2} \\\\
&= \frac{1}{1 + e^{-z}} (1 - \frac{1}{1 + e^{-z}}) \\\\
&= g(z) \cdot (1 - g(z))
\end{aligned}
$$

and $\delta_i^{(L,k)}$ can be computed with the sigmoid function $g(z)$

$$
\begin{aligned}
\delta_i^{(L,k)}
&= \frac{\partial \left\{ \sum_{i = 1}^{S_L}
\left[ y^{(k)}_i \left(- \log \left( h_{\theta}(x^{(k)})_i \right) \right)
+ (1 - y^{(k)}_i) \left(- \log \left(1 - h_{\theta}(x^{(k)})_i \right) \right)
\right] \right\} }
{\partial z_i^{(L,k)}} \\\\
&= \frac{\partial \left\{ \sum_{i = 1}^{S_L}
\left[ y^{(k)}_i \left(- \log \left( g(z_i^{(L,k)}) \right) \right)
+ (1 - y^{(k)}_i) \left(- \log \left(1 - g(z_i^{(L,k)}) \right) \right)
\right] \right\} }
{\partial z_i^{(L,k)}} \\\\
&= \frac{\partial \left\{
y^{(k)}_i \left(- \log \left( g(z_i^{(L,k)}) \right) \right)
+ (1 - y^{(k)}_i) \left(- \log \left(1 - g(z_i^{(L,k)}) \right) \right)
\right\}}{\partial z_i^{(L,k)}} \\\\
&= - y_i^{(k)} \frac{1}{g(z_i^{(L,k)})}g'(z_i^{(L,k)})
+ (1 - y_i^{(k)}) \cdot - \frac{1}{1 - g(z_i^{(L,k)})} \cdot - g'(z_i^{(L,k)}) \\\\
&= - y_i^{(k)} (1 - g(z_i^{(L,k)})) + (1 - y_i^{(k)}) g(z_i^{(L,k)}) \\\\
&= g(z_i^{(L,k)}) - y_i^{(k)} \\\\
&= a_i^{(L,k)} - y_i^{(k)} \\\\
\end{aligned}
$$
