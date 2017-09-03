---
layout: post
title: "Support Vector Machine"
author: "HzCeee"
categories: Machine Learning
tags: [ML Algorithm]
image:
  feature: 
  teaser: 
  credit:
  creditlink:
---

# Support Vector Machine

Introduce a new notation for SVM: use $y \in \\{ -1, 1 \\}$ instead of $\\{ 0, 1 \\}$, let $w = [\theta_1, \theta_2, \cdots, \theta_n]^T$ and $b = \theta_0$. Thus,

$$
h_{\theta}(x) = h_{w,b}(x) = g(w^T x + b)
$$

where

$$
g(z) = 
\begin{cases}
1 & \text{$z \geq 0$} \\\\
-1 & \text{otherwise}
\end{cases}
$$

## Functional and Geometric Margins

### Functional Margins

Given a training example $(x^{(i)} , y^{(i)})$, the functional margin of $(w,b)$ with respect to the training example is:

$$
\hat \gamma ^{(i)} = y^{(i)} (w^T x ^{(i)} + b)
$$

Given a training set $S = \\{ (x^{(i)} , y^{(i)}); i = 1, \cdots, m \\}$, the functional margin is:

$$
\hat \gamma = \min_{i = 1, \cdots, m} \hat \gamma ^ {(i)}
$$

### Geometric Margins

Note that $w$ is the normal vector to the separating hyperplane. 

Use point $A$ to represent $x^{(i)}$ on the positive side and point $B$ to represent the point lying on the decision boundary, and satisfy:

$$
\overrightarrow{BA} // \vec{w}
$$

Therefore $B$ is given by $x^{(i)} - \gamma^{(i)} \cdot w / \|w\|$ . Hence,

$$
w^T (x^{(i)} - \gamma^{(i)} \frac{w}{\|w\|}) + b = 0
$$

Solving for $\gamma^{(i)}$ yields

$$
\gamma^{(i)} = (\frac{w}{\|w\|})^T x^{(i)} + \frac{b}{\|w\|}
$$

Similarly, use point $A$ to represent $x^{(i)}$ on the negative side, therefore $B$ is the same point as mentioned above. Therefore $B$ is given by $x^{(i)} + \gamma^{(i)} \cdot w / \|w\|$. Hence,

$$
w^T (x^{(i)} + \gamma^{(i)} \frac{w}{\|w\|}) + b = 0
$$

Solving for $\gamma^{(i)}$ yields

$$
\gamma^{(i)} = - (\frac{w}{\|w\|})^T x^{(i)} - \frac{b}{\|w\|}
$$

To combine these two cases, define the geometric margin with respect to a training example $(x^{(i)} , y^{(i)})$ to be

$$
\gamma^{(i)} = y^{(i)} ((\frac{w}{\|w\|})^T x^{(i)} + \frac{b}{\|w\|}) = \frac {\hat \gamma ^{(i)}}{\|w\|}
$$

Given a training set $S = \\{ (x^{(i)} , y^{(i)}); i = 1, \cdots, m \\}$, the geometric margin is:

$$
\gamma = \min_{i = 1, \cdots, m} \gamma ^ {(i)}
$$

## The Optimal Margin Classifier

### Problem Formulation

#### Linearly Separable Case

##### Original Problem

Given a training set that is linearly separable using some separating hyperplane. To find the one that achieves the maximum geometric margin, pose the following optimization problem:

$$
\begin{aligned}
\max_{\gamma, w, b} & \quad \gamma \\\\
s.t. & \quad y^{(i)} (w^T x^{(i)} + b) \geq \gamma, i = 1, \cdots, m \\\\
& \quad \|w\| = 1
\end{aligned}
$$

However, it could not be solved by the standard optimization software because of the constraint $\|w\| = 1$. So, it can be tranformed into a nicer one with the property that $\gamma = \hat \gamma / \|w\|$:

$$
\begin{aligned}
\max_{\hat \gamma, w, b} & \quad \frac{\hat \gamma}{\|w\|} \\\\
s.t. & \quad y^{(i)} (w^T x^{(i)} + b) \geq \hat \gamma, i = 1, \cdots, m \\\\
\end{aligned}
$$

However the objective function $\frac{\hat \gamma}{\|w\|}$ is non-convex. Consider the fact that we can add arbitrary scaling constraint on $w$ and $b$ without changing the constraint because $\hat \gamma ^{(i)} = y^{(i)} (w^T x ^{(i)} + b)$. Thus, we can introduce the specfic scaling constraint so that

$$
\hat \gamma = 1
$$

Therefore maximizing  $\frac{\hat \gamma}{\|w\|} = \frac{1}{\|w\|}$ is the same thing as minimizing $\|w\|^2$, so the original problem can be transformed as:

$$
\begin{aligned}
\max_{\gamma, w, b} & \quad \frac{1}{2} \|w\|^2 \\\\
s.t. & \quad y^{(i)} (w^T x^{(i)} + b) \geq 1, i = 1, \cdots, m \\\\
\end{aligned}
$$

And now this optimization problem can be solved using commercial quadratic programming code.

##### Dual Problem

The constraints can be written as:

$$
g_i(w) = - y^{(i)}(w^T x^{(i)} + b) + 1 \leq 0
$$

the Lagrangian is:

$$
\mathcal{L} (w, b, \alpha) = \frac{1}{2} ||w||^2 - \sum_{i = 1}^m \alpha_i [y^{(i)} (w^T x^{(i)} + b) - 1]
$$

To get the dual form of the problem, first minimize $\mathcal{L}(w, b, \alpha)$ with respect to $w$ and $b$ for fixed $\alpha$. Thus,

$$
\nabla_w \mathcal{L} (w, b, \alpha) = w - \sum_{i = 1}^m \alpha_i y^{(i)} x^{(i)} = 0 \\\\
\frac{\partial}{\partial b} \mathcal{L}(w, b, \alpha) = \sum_{i = 1}^m \alpha_i y^{(i)} = 0
$$

Then we have:

$$
w = \sum_{i = 1}^m \alpha_i y^{(i)} x^{(i)} \\\\
\sum_{i = 1}^m \alpha_i y^{(i)} = 0
$$

Plug $w$ and $b$ above back into the Lagrangian and simplify:

$$
\mathcal{L}(w, b, \alpha) = \sum_{i = 1}^m \alpha_i - \frac{1}{2} \sum_{i,j = 1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j (x^{(i)})^T x^{(j)}
$$

Thus the dual optimization problem:

$$
\begin{aligned}
\max_{\alpha} & \quad W(\alpha) = \sum_{i = 1}^m \alpha_i - \frac{1}{2} \sum_{i,j = 1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j (x^{(i)})^T x^{(j)} \\\\
s.t. & \quad \alpha_i \geq 0, i = 1, \cdots, m \\\\
& \quad \sum_{i = 1}^m \alpha_i y^{(i)} = 0
\end{aligned}
$$

#### Non-Linearly Separable Case

##### Original Problem

To make the algorithm work for non-linearly separable datasets as well as be less sensitive to outliers, reformulate our optimization as follows:

$$
\begin{aligned}
\max_{\gamma, w, b} & \quad \frac{1}{2} ||w||^2 + C \sum_{i = 1}^m \xi_i \\\\
s.t. & \quad y^{(i)} (w^T x^{(i)} + b) \geq 1 - \xi_i, i = 1, \cdots, m \\\\
& \quad \xi_i \geq 0, i = 1, \cdots, m \\\\
\end{aligned}
$$

Thus, examples are now permitted to have (functional) margin less than 1, and if an example has functional margin $1 − \xi_i$, we would pay a cost of the objective function being increased by $C \xi_i$.

##### Dual Problem

The Lagrangian is:

$$
\mathcal{L}(w, b, \xi, \alpha, r) = \frac{1}{2} w^T w + C \sum_{i = 1}^m \xi_i - \sum_{i = 1}^m \alpha_i[y^{(i)}(x^T w + b) - 1 + \xi_i] - \sum_{i = 1}^m r_i \xi_i
$$

after setting the derivatives with respect to $w$ and $b$ to zero as before, substituting them back in and simplifying, obtain the following dual form of the problem:

$$
\begin{aligned}
\max_{\alpha} & \quad W(\alpha) = \sum_{i = 1}^m \alpha_i - \frac{1}{2} \sum_{i,j = 1}^m y^{(i)} y^{(j)} \alpha_i \alpha_j (x^{(i)})^T x^{(j)} \\\\
s.t. & \quad 0 \geq \alpha_i \geq C, i = 1, \cdots, m \\\\
& \quad \sum_{i = 1}^m \alpha_i y^{(i)} = 0
\end{aligned}
$$

### Sequential Minimal Optimization(SMO) Algorithm

For the two dual problems above, after finding $\alpha$ that maximize $W(\alpha)$ subject to the constraints, $w$ can be derived by $\nabla_w \mathcal{L} (w, b, \alpha) = w - \sum_{i = 1}^m \alpha_i y^{(i)} x^{(i)} = 0$ and $b$ can also be found by considering the primal problem directly. Thus, we can make a prediction by $y = h_{w, b} (x) = g(w^T x + b)$ where $g(z) = 1$ if $z \geq 0$ and $g(z) = -1$ otherwise.

In order to get optimal $\alpha$, consider Coordinate Ascent Algorithm. However,

$$
\begin{aligned}
\alpha_1 y^{(1)} &= - \sum_{i = 2}^m \alpha_i y^{(i)} \\\\
\alpha_1 y^{(1)} y^{(1)} &= - y^{(1)} \sum_{i = 2}^m \alpha_i y^{(i)} \\\\
\alpha_1 &= - y^{(1)} \sum_{i = 2}^m \alpha_i y^{(i)} \\\\
\end{aligned}
$$

we can’t make any change to $\alpha_1$ without violating the constraint because $\alpha_1$ is determined by the other $\alpha_i$'s.

Thus, we must update at least two of them simultaneously in order to keep satisfying the constraints:

$$
\begin{aligned}
&Repeat \ until \ convergence \ \{ \\\\
&\qquad SelectPairToUpdate(\alpha_i, \alpha_j) \\\\
&\qquad Reoptimze(\alpha_i, \alpha_j)\\\\
&\}
\end{aligned}
$$

$SelectPairToUpdate()$ function in the algorithm often uses a heuristic that tries to pick the two that will allow us to make the biggest progress towards the global maximum. 

$Reoptimze(\alpha_i, \alpha_j)$ function in the algorithm reoptimize $W(\alpha)$ with respect to $\alpha_i$ and $\alpha_j$ while holding all the other $\alpha_k (k \neq i, j)$ fixed.