---
layout: post
title: "Nearest Neighbor Classifier"
author: "HzCeee"
categories: Machine Learning
tags: [Classification]
image:
  feature: 
  teaser: 
  credit:
  creditlink:
---

__Nearest Neighbor Classfication__

Assign the label to the same label by taking majority among $k$-nearest neighbors.

Measure closeness between two examples $x_1, x_2 \in \mathcal{R}^d$:

- Compute distance

  - Normed distance

    $\rho(x_1, x_2) = \left[ \|x_1^{(1)} - x_2^{(1)} \|^p + \cdots + \|x_1^{(d)} - x_2^{(d)} \|^p \right]^{1/p} = \\| x_1 - x_2 \\|_p$

    - Euclidean distance ($p = 2$)
    - Manhatten distance ($p = 1$)
    - Max distance ($p = \infty$)
    - Count 'non-zero' distance ($p = 0$)

- Compute similarity

  - $\rho(x_1, x_2) = \frac{1}{1 + \\| x_1 - x_2 \\|_2}$
  - $\rho(x_1, x_2) = \sum_{i = 1}^d \frac{1}{1 + \| x_1^{(i)} - x_2^{(i)} \|}$
  - Cosine similarity: $\rho(x_1, x_2) = cos(\angle(x_1, x_2)) = \frac{x1 \cdot x2}{\\| x1 \\|_2 \\| x_2 \\|_2}$

- Use domain expertise

  - Edit distance (to compare genome sequences): Minimum number of insertions, deletions and mutations needed
  - Kendell-Tau distance (to compare rankings): Bubble sort distance to make one ranking order same as the other

---

__$k$-NN Optimality__

__Theorem 1:__

For fixed $k$, as and $n \to \infty$, $k$-NN classifier error converges to no more than twice Bayes classifier error.

> __1-NN case__
>
> __Notation:__
> \\
> $P[e]$ = NN error rate
> \\
> $D_n = (X_n, Y_n)$ = labeled training data (size n)
> \\
> $x_n$ = nearset neighbour of $x_t$ in $D_n$
> \\
> $y_n$ = label of the nearest neighbor
> \\
> $P^{*}[e]$ = Bayes error rate
>
> $$
> \begin{aligned}
> & \quad \lim_{n \to \infty} P_{y_t, D_n} [e | x_t] \\
> & = \lim_{n \to \infty} \int P_{y_t, Y_n} [e | x_t, X_n]P[X_n | x_t] d X_n \\
> & = \lim_{n \to \infty} \int P_{y_t, y_n} [e | x_t, x_n]P[x_n | x_t] d x_n \\
> & = \lim_{n \to \infty} \int \left[ 1 - \sum_{y \in \mathcal{Y}} P(y_t = y, y_n = y | x_t, x_n) \right] P[x_n | x_t] d x_n \\
> & = \lim_{n \to \infty} \int \left[ 1 - \sum_{y \in \mathcal{Y}} P(y_t = y | x_t) P(y_n = y | x_n) \right] P[x_n | x_t] d x_n \\
> & = 1 - \sum_{y \in \mathcal{Y}} P^2 (y_t = y | x_t) \\
> & \leq 1 - P^2 (y_t = y^{*} | x_t) \\
> & \leq 2 ( 1 - P(y_t = y^{*} | x_t) ) \\
> & = 2P^{*} [e | x_t] \\
> \end{aligned}
> $$
>

__Theorem 2:__

If $k \to \infty$, $k/n \to 0$, and $n \to \infty$, $k$-NN classifier converges to Bayes classifier.

---

__Improvement__

- Speed up

  - Original Search

    $n$ = number of training data

    $d$ = representation dimension

    Computational cost: $O(nd)$

  - Binary Search (finding medians)

    Sorting Cost: $O(n \log n)$

    Search Cost: $O(\log n)$

    Computational Cost: $O(n \log n)$

- Make distance mesasurement robust

  Take 2-norm as an example:

  - Orginal distance

    $\rho(x_1, x_2) = \left[ (x_1^{(1)} - x_2^{(1)})^2 + \cdots + (x_1^{(d)} - x_2^{(d)})^2 \right]^{1/2} = \\| x_1 - x_2 \\|_2$

  - Re-weight distance

    $\rho(x_1, x_2; w) = \left[ w_1 \cdot (x_1^{(1)} - x_2^{(1)})^2 + \cdots + w_d \cdot (x_1^{(d)} - x_2^{(d)})^2 \right]^{1/2} = \left[ (x_1 - x_2)^T W (x_1 - x_2) \right]^{1/2}$

    Where

    $$
    W = 
    \begin{bmatrix}
    w_1 & 0 & \cdots & 0 \\
    0 & w_2 & \cdots & 0 \\
    \vdots & \vdots & & \vdots \\
    0 & 0 & \cdots & w_d \\
    \end{bmatrix}
    $$

    Create two set: Similar set $S := \\{ (x_i, x_j) \| y_i = y_j \\}$ and Dissimilar set $D := \\{ (x_i, x_j) \| y_i \neq y_j \\}$

    Define Cost function $\Psi(w)$ and minimize $\Psi(x)$ with respect to $w$:

    $$
    \Psi(w) = \lambda \sum_{(x_i, x_j) \in S} \rho (x_i, x_j; w) - (1 - \lambda) \sum_{(x_i, x_j) \in D} \rho(x_1, x_2; w)
    $$

- Save memory

  - Original method

    Keep all the training data during test time

  - Improved method

    Label each cell and only keep cell in memory

