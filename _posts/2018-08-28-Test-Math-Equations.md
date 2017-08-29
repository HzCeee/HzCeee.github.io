# Flag 3
# Support Vector Machine
Introduce a new notation for SVM: use $y \in \{ -1, 1 \}$ instead of $\{ 0, 1 \}$, let $w = [\theta_1, \theta_2, \cdots, \theta_n]^T$ and $b = \theta_0$. Thus,

$$
h_{\theta}(x) = h_{w,b}(x) = g(w^T x + b)
$$

where

$$
g(z) = 
\begin{cases}
1 & \text{$z \geq 0$} \\
0 & \text{otherwise}
\end{cases}
$$

## Functional and Geometric Margins
### Functional Margins
Given a training example $(x^{(i)} , y^{(i)})$, the functional margin of $(w,b)$ with respect to the training example is:

$$
\hat \gamma ^{(i)} = y^{(i)} (w^T x ^{(i)} + b)
$$

Given a training set $S = \{ (x^{(i)} , y^{(i)}); i = 1, \cdots, m \}$, the functional margin is:

$$
\hat \gamma = \min_{i = 1, \cdots, m} \hat \gamma ^ {(i)}
$$

### Geometric Margins
Note that $w$ is the normal vector to the separating hyperplane. 

Use point $A$ to represent $x^{(i)}$ on the positive side and point $B$ to represent the point lying on the decision boundary, and satisfy:

$$
\overrightarrow{BA} // \vec{w}
$$

Therefore $B$ is given by $x^{(i)} - \gamma^{(i)} \cdot w / ||w||$. Hence,

$$
w^T (x^{(i)} - \gamma^{(i)} \frac{w}{||w||}) + b = 0
$$

Solving for $\gamma^{(i)}$ yields

$$
\gamma^{(i)} = (\frac{w}{||w||})^T x^{(i)} + \frac{b}{||w||}
$$

Similarly, use point $A$ to represent $x^{(i)}$ on the negative side, therefore $B$ is the same point as mentioned above. Therefore $B$ is given by $x^{(i)} + \gamma^{(i)} \cdot w / ||w||$。 Hence,

$$
w^T (x^{(i)} + \gamma^{(i)} \frac{w}{||w||}) + b = 0
$$

Solving for $\gamma^{(i)}$ yields

$$
\gamma^{(i)} = - (\frac{w}{||w||})^T x^{(i)} - \frac{b}{||w||}
$$

To combine these two cases, define the geometric margin with respect to a training example $(x^{(i)} , y^{(i)})$ to be

$$
y^{(i)} = y^{(i)} ((\frac{w}{||w||})^T x^{(i)} + \frac{b}{||w||})
$$

Given a training set $S = \{ (x^{(i)} , y^{(i)}); i = 1, \cdots, m \}$, the geometric margin is:

$$
\gamma = \min_{i = 1, \cdots, m} \gamma ^ {(i)}
$$
