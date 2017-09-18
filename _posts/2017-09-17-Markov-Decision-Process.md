---
layout: post
title: "Markov Decision Process"
author: "HzCeee"
categories: Machine Learning
tags: [Reinforcement Learning]
image:
  feature: 
  teaser:
  credit:
  creditlink:
---

__Markov Reward Process__

A Markov Reward Process is a tuple $\langle \mathcal{S}, \mathcal{P}, \mathcal{R}, \mathcal{\gamma} \rangle$ where $\mathcal{S}$ is a finite set of states, $\mathcal{P}$ is a state transition probability matrix ($\mathcal{P}\_{ss'} = \mathbb{P}[S_{t + 1} = s' \| S_t = s]$), $\mathcal{R}$ is a reward function ($\mathcal{R}\_s = \mathbb{E}[R_{t + 1} \| S_t = s]$) and $\mathcal{\gamma}$ is a discount factor ($\mathcal{\gamma} \in \[0, 1\]$).

Each state $s_t$ can be mapped to a reward $R_t$.

Return $G_t$ is the total discounted reward $R_t$ from time-step $t$:

$$
G_t = R_{t + 1} + \gamma R_{t + 2} + \cdots = \sum_{k = 0}^\infty \gamma^k R_{t + k + 1}
$$

State value function $v(s)$ of an MRP is the expected return staring from state $s$:

$$
\begin{aligned}
v(s) 
& = \mathbb{E}[G_t | S_t = s] \\
& = \mathbb{E}[R_{t + 1} + \gamma R_{t + 2} + \gamma^2 R_{t + 3} + \cdots | S_t = s] \\
& = \mathbb{E}[R_{t + 1} + \gamma(R_{t + 2} + \gamma R_{t + 3} + \cdots) | S_t = s] \\
& = \mathbb{E}[R_{t + 1} + \gamma G_{t + 1} | S_t = s] \\
& = \mathbb{E}[R_{t + 1} | S_t = s] + \gamma \mathbb{E}[G_{t + 1} | S_t = s] \\
& = \mathcal{R}_s + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'} v(s')
\end{aligned}
$$

Above equations can be written in a vectorized form:

$$
\mathcal{v} = \mathcal{R} + \gamma \mathcal{P} \mathcal{v}
$$

Specifically,

$$
\begin{bmatrix}
v(1) \\
\vdots \\
v(n)
\end{bmatrix}
=
\begin{bmatrix}
\mathcal{R}\_1 \\
\vdots \\
\mathcal{R}\_n
\end{bmatrix}
+
\gamma
\begin{bmatrix}
\mathcal{P}\_{11} & \cdots & \mathcal{P}\_{1n} \\
\vdots & & \vdots \\ 
\mathcal{P}\_{n1} & \cdots & \mathcal{P}\_{nn} \\
\end{bmatrix}
\begin{bmatrix}
v(1) \\
\vdots \\
v(n)
\end{bmatrix}
$$

And $\mathcal{v}$ can be solved directly ($O(n^3)$):

$$
\begin{aligned}
\mathcal{v} 
& = \mathcal{R} + \gamma \mathcal{P} \mathcal{v} \\
(I - \gamma \mathcal{P}) \mathcal{v} & = \mathcal{R} \\
\mathcal{v} & = (I - \gamma \mathcal{P})^{-1} \mathcal{R} \\
\end{aligned}
$$

---

__Markov Decision Process__

A Markov Decision Process is a tuple $\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle$ where $\mathcal{S}$ is a finite set of states, $\mathcal{A}$ is a finite set of actions, $\mathcal{P}$ is a state transition probability matrix ($\mathcal{P}\_{ss'}^a = \mathbb{P}[S_{t + 1} = s' \| S_t = s, A_t = a]$), $\mathcal{R}$ is a reward function ($\mathcal{R}\_s^a = \mathbb{E}[R_{t + 1} \| S_t = s, A_t = a]$) and $\mathcal{\gamma}$ is a discount factor ($\mathcal{\gamma} \in \[0, 1\]$).

Action $a$ is determined by policy $\pi(a \| s) = \mathbb{P}[A_t = a \| S_t = s]$ which indicates the probability of possible action $a \in \mathcal{A}(s)$ in state $s$.

The state and reward sequence $S_1, R_2, S_2, \cdots$ is a Markov Reward Process $\langle \mathcal{S}, \mathcal{P}^{\pi}, \mathcal{R}^{\pi}, \gamma \rangle$ where

$$
\mathcal{P}_{ss'}^{\pi} = \sum_{a \in \mathcal{A}} \pi(a | s) \mathcal{P}_{ss'}^a \\
\mathcal{R}_s^{\pi} = \sum_{a \in \mathcal{A}} \pi(a | s) \mathcal{R}_s^a
$$

The state-value function $v_{\pi}(s)$ of MDP is the expected return starting from state $s$, and then following policy $\pi$:

$$
\begin{aligned}
v_{\pi}(s)
& = \mathbb{E}_{\pi} [G_t | S_t = s] \\
& = \mathbb{E}_{\pi} [R_{t + 1} + \gamma R_{t + 2} + \gamma^2 R_{t + 3} + \cdots | S_t = s] \\
& = \mathbb{E}_{\pi} [R_{t + 1} + \gamma(R_{t + 2} + \gamma R_{t + 3} + \cdots) | S_t = s] \\
& = \mathbb{E}_{\pi} [R_{t + 1} + \gamma G_{t + 1} | S_t = s] \\
& = \mathbb{E}_{\pi} [R_{t + 1} | S_t = s] + \gamma \mathbb{E}_{\pi} [G_{t + 1} | S_t = s] \\
& = \sum_{a \in \mathcal{A}} \pi(a | s) \mathcal{R_s^a} + \gamma \sum_{a \in \mathcal{A}} \sum_{s' \in \mathcal{S}}\mathcal{P}_{ss'}^a v_{\pi}(s') \\
& = \sum_{a \in \mathcal{A}} \pi(a | s) \left( \mathcal{R_s^a} + \gamma \sum_{s' \in \mathcal{S}}\mathcal{P}_{ss'}^a v_{\pi}(s') \right)
\end{aligned}
$$

The action-value function $q_{\pi} (s,a)$ is the expected return staring from state $s$, taking action $a$, and then following policy $\pi$:

$$
\begin{aligned}
q_{\pi}(s,a)
& = \mathbb{E}_{\pi} [G_t | S_t = s, A_t = a] \\
& = \mathbb{E}_{\pi} [R_{t + 1} + \gamma R_{t + 2} + \gamma^2 R_{t + 3} + \cdots | S_t = s, A_t = a] \\
& = \mathbb{E}_{\pi} [R_{t + 1} + \gamma(R_{t + 2} + \gamma R_{t + 3} + \cdots) | S_t = s, A_t = a] \\
& = \mathbb{E}_{\pi} [R_{t + 1} + \gamma G_{t + 1} | S_t = s, A_t = a] \\
& = \mathbb{E}_{\pi} [R_{t + 1} | S_t = s, A_t = a] + \gamma \mathbb{E}_{\pi} [G_{t + 1} | S_t = s, A_t = a] \\
& = \mathcal{R}_s^a + \gamma \sum_{s' \in \mathcal{S}} \mathcal{P}_{ss'}^a \sum_{a' \in \mathcal{A}} \pi(a' | s') q_{\pi} (s', a')
\end{aligned}
$$

By vectorization,

$$
\begin{aligned}
v_{\pi}
& = \mathcal{R}^{\pi} + \gamma \mathcal{P}^{\pi} v_{\pi} \\
& = (I - \gamma \mathcal{P}^{\pi})^{-1} \mathcal{R}^{\pi} \\
\end{aligned}
$$

__Optimal Value Function__

The optimal state-value function:

$$
v_{*}(s) = \max_{\pi} v_{\pi}(s)
$$

The optimal action-value function:

$$
q_{*}(s,a) = \max_{\pi} q_{\pi}(s,a)
$$

And optimal policy can be found by maximizing over $q_{*}(s,a)$:

$$
\pi_{*}(a | s) = 
\begin{cases}
1 & if \ a = \arg\max_{a \in \mathcal{A}} q_{*}(s,a) \\
0 & otherwise
\end{cases}
$$
