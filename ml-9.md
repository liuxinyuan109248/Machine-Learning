# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 09 Nearest Neighbors</span>

### <span style="color: #6ED3C5">Dimension Reduction</span>

<span style="color: #6FA8FF">**Johnson-Lindenstrauss Lemma:**</span> For any $0 < \epsilon < 1$ and any integer $n$, let $k$ be a positive integer such that $k \geq \dfrac{8 \ln n}{\epsilon^2}$. For any set $X$ of $n$ points in $\mathbb{R}^d$, there exists a map $f: \mathbb{R}^d \to \mathbb{R}^k$ such that for all $u, v \in X$,
$$(1 - \epsilon) \|u - v\|^2 \leq \|f(u) - f(v)\|^2 \leq (1 + \epsilon) \|u - v\|^2.$$

### <span style="color: #6ED3C5">Principal Component Analysis (PCA)</span>

<span style="color: #6FA8FF">**PCA**</span> is a dimensionality reduction technique that identifies the directions (principal components) along which the variation in the data is maximal. PCA can be viewed through two equivalent mathematical lenses:

* **Maximize Variance**: Find a unit vector $v$ that maximizes the "spread" of the data when projected:
    $$\max_{\|v\|=1} \sum_{i=1}^n (x_i \cdot v)^2 = v^T X X^T v$$
* **Minimize Reconstruction Error**: Find a subspace that minimizes the distance between original points and their projections:
    $$\min \sum_{i=1}^n \|x_i - (x_i \cdot v)v\|^2 = \sum_{i=1}^n \|x_i\|^2 - \sum_{i=1}^n (x_i \cdot v)^2$$

The optimal vector $v$ is the **eigenvector** of the matrix $XX^T$ corresponding to its **largest eigenvalue**. For a $k$-dimensional representation, we select the eigenvectors associated with the **top $k$ eigenvalues**.

### <span style="color: #6ED3C5">Power Method</span>

The Power Method is an iterative numerical algorithm used to calculate the dominant eigenvalue and its corresponding eigenvector of a matrix $A$. Starting with a random initial vector $b_0$, the vector is repeatedly multiplied by $A$ and normalized to prevent numerical overflow: $b_{k+1} = \dfrac{A b_k}{\|A b_k\|}$. This sequence converges to the eigenvector associated with the largest eigenvalue.

To find subsequent principal components, we use **deflation** to "remove" the influence of the eigenvectors already found:

1. **Extract Eigenvalue**: Calculate $\lambda_1 = b_k^T A b_k$ once the vector has converged.
2. **Update Matrix**: Subtract the rank-1 matrix formed by the found eigenvector: $A \leftarrow A - \lambda_1 v_1 v_1^T$.
3. **Repeat**: Apply the Power Method to the updated matrix to find the next dominant eigenvalue/vector pair.

### <span style="color: #6ED3C5">Locality Sensitive Hashing (LSH)</span>

<span style="color: #6FA8FF">**Nearest neighbor problem:**</span> find $x^* = \arg\min_{x \in X} \|x - q\|$ for a query point $q$

<span style="color: #6FA8FF">**$R$-near neighbor problem:**</span> find any $x \in X$ such that $\|x - q\| \leq R$

<span style="color: #6FA8FF">**Randomized $c$-approximate $R$-near neighbor problem:**</span> if there exists $x \in X$ such that $\|x - q\| \leq R$, return some $x' \in X$ such that $\|x' - q\| \leq cR$ with probability at least $1 - \delta$

<span style="color: #6FA8FF">**Locality Sensitive Hashing (LSH):**</span> A family $\mathcal{H}$ of hash functions is called $(R, cR, P_1, P_2)$-sensitive if for any $x, y \in \mathbb{R}^d$,

- if $\|x - y\| \leq R$, then $\Pr_{h \in \mathcal{H}}[h(x) = h(y)] \geq P_1$
- if $\|x - y\| \geq cR$, then $\Pr_{h \in \mathcal{H}}[h(x) = h(y)] \leq P_2$

**Example:** binary vectors with Hamming distance: choose a random coordinate $i$ and define $h(x) = x_i$, $P_1 = 1 - \dfrac{R}{d}$, $P_2 = 1 - \dfrac{cR}{d}$

<span style="color: #6FA8FF">**Amplification:**</span> To amplify the gap between $P_1$ and $P_2$, use the following techniques:

- <span style="color: #6FA8FF">**AND construction:**</span> Choose $k$ hash functions $h_1, h_2, \ldots, h_k$ independently from $\mathcal{H}$ and define $g(x) = (h_1(x), h_2(x), \ldots, h_k(x))$. Then,
  - if $\|x - y\| \leq R$, $\Pr[g(x) = g(y)] \geq P_1^k$
  - if $\|x - y\| \geq cR$, $\Pr[g(x) = g(y)] \leq P_2^k$
- <span style="color: #6FA8FF">**OR construction:**</span> Choose $L$ hash functions $g_1, g_2, \ldots, g_L$ independently from the amplified family (e.g., from the AND construction) and define $G(x) = (g_1(x), g_2(x), \ldots, g_L(x))$. Then,
  - if $\|x - y\| \leq R$, $\Pr[G(x) = G(y)] \geq 1 - (1 - P_1^k)^L$
  - if $\|x - y\| \geq cR$, $\Pr[G(x) = G(y)] \leq L P_2^k$
  - definition of $G(x) = G(y)$: there exists some $i$ such that $g_i(x) = g_i(y)$
- **overall LSH scheme:** Use $L$ hash tables, each with a hash function from the amplified family. For a query point $q$, check all $L$ hash tables to find candidate points and verify them. Stop the search after $2L+1$ points have been checked.
- **data structure size:** $O(n L)$ (the size of each hash table is $O(n)$)

<span style="color: #6FA8FF">**Theorem:**</span> Given an $(R, cR, P_1, P_2)$-sensitive family $\mathcal{H}$ with $P_1 > P_2$ and parameters $k = \log_{1/P_2} n$ and $L = n^{\rho}$ where $\rho = \dfrac{\log(1/P_1)}{\log(1/P_2)}$, the above LSH scheme solves the randomized $c$-approximate $R$-near neighbor problem with space complexity $O(n^{1+\rho})$ and query time $O(n^{\rho})$. To be specific, if there exists $x \in X$ such that $\|x - q\| \leq R$, the scheme returns some $x' \in X$ such that $\|x' - q\| \leq cR$ with probability at least $\dfrac{1}{2} - \dfrac{1}{e}$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

For a query point $q$, let $S$ be the set of points in $X$ such that $\|x - q\| \leq R$ and $T$ be the set of points in $X$ such that $\|x - q\| \geq cR$.

- $\Pr[\text{no point in } S \text{ is found}] \leq (1 - P_1^k)^L \leq e^{-L P_1^k} = e^{-n^{\rho} \cdot n^{-\rho}} = e^{-1}$
- $\mathbb{E}[\text{number of points in } T \text{ found}] \leq |T| \cdot L P_2^k \leq n \cdot n^{\rho} \cdot n^{-\rho} = L$

By Markov's inequality, the probability that more than $2L$ points in $T$ are found is at most $\dfrac{1}{2}$. Therefore, the probability that at least one point in $S$ is found and at most $2L$ points in $T$ are found is at least $\dfrac{1}{2} - \dfrac{1}{e}$.

</details>

### <span style="color: #6ED3C5">$\ell_2$ LSH family</span>

<span style="color: #6FA8FF">**Theorem:**</span> The family of hash functions defined by $h_{a,b}(v) = \left\lfloor \dfrac{a \cdot v + b}{w} \right\rfloor$, where $a$ is a random vector with each entry drawn from Gaussian distribution $\mathcal{N}(0, 1)$, $b$ is drawn uniformly from $[0, w)$, and $w$ is a fixed width parameter, is an $(R, cR, P_1, P_2)$-sensitive family for $\ell_2$ distance with $P_1 = p(R)$ and $P_2 = p(cR)$, where $p(d) = \displaystyle \int_0^w \dfrac{1}{d} f\left(\dfrac{t}{d}\right) \left(1 - \dfrac{t}{w}\right) dt$ and $f(x) = \sqrt{\dfrac{2}{\pi}} e^{-x^2/2}$ is the PDF of the half-normal distribution.

<span style="color: #6FA8FF">**Proof Sketch:**</span> The probability that $h_{a,b}(x) = h_{a,b}(y)$ depends on the distance $\|x - y\|$. By projecting onto the random vector $a$, the difference $a \cdot (x - y)$ follows a normal distribution with mean 0 and variance $\|x - y\|^2$. The absolute value $|a \cdot (x - y)|$ thus follows a half-normal distribution. The hash values collide if the projected distance falls within the same interval of width $w$, leading to the integral expression for $p(d)$.

### <span style="color: #6ED3C5">$\ell_1$ LSH family</span>

<span style="color: #6FA8FF">**Theorem:**</span> The family of hash functions defined by $h_{a,b}(v) = \left\lfloor \dfrac{a \cdot v + b}{w} \right\rfloor$, where $a$ is a random vector with each entry drawn from Cauchy distribution $\text{Cauchy}(0, 1)$, $b$ is drawn uniformly from $[0, w)$, and $w$ is a fixed width parameter, is an $(R, cR, P_1, P_2)$-sensitive family for $\ell_1$ distance with $P_1 = p(R)$ and $P_2 = p(cR)$, where $p(d) = \displaystyle \int_0^w \dfrac{2}{\pi} \dfrac{d}{d^2 + t^2} \left(1 - \dfrac{t}{w}\right) dt$.

<span style="color: #6FA8FF">**Proof Sketch:**</span> Similar to the $\ell_2$ case, the projection $a \cdot (x - y)$ follows a Cauchy distribution when $a$ is drawn from a Cauchy distribution. The probability of collision depends on the distance $\|x - y\|_1$ and leads to the integral expression for $p(d)$.

### <span style="color: #6ED3C5">Metric Learning</span>

<span style="color: #6FA8FF">**Neighborhood Components Analysis (NCA):**</span> Learn a linear transformation $A$ to minimize the leave-one-out classification error of 1-NN on the training set. The probability that point $x_i$ selects point $x_j$ as its neighbor is given by $p_{ij} = \dfrac{\exp(-\|Ax_i - Ax_j\|^2)}{\sum_{k \neq i} \exp(-\|Ax_i - Ax_k\|^2)}$. The objective is to maximize the expected number of correctly classified points $\sum_i \sum_{j \in C_i} p_{ij}$, where $C_i$ is the set of points with the same label as $x_i$. Once $A$ is learned, the classification of a new point $x$ is done by finding its nearest neighbor in the transformed space.

<span style="color: #6FA8FF">**Distribution Alignment:**</span> Minimize the discrepancy between the source and target distributions after transformation using Kullback-Leibler divergence as the loss function:
$$p_{ij}^X = \dfrac{\exp(-\|Ax_i - Ax_j\|^2)}{\sum_{k \neq i} \exp(-\|Ax_i - Ax_k\|^2)}, \quad p_{ij}^Y = \dfrac{\exp(-\|Ay_i - Ay_j\|^2)}{\sum_{k \neq i} \exp(-\|Ay_i - Ay_k\|^2)}, \quad L = \sum_{i,j} p_{ij}^X \log \dfrac{p_{ij}^X}{p_{ij}^Y}.$$

### <span style="color: #6ED3C5">Large Margin Nearest Neighbor (LMNN)</span>

<span style="color: #6FA8FF">**LMNN**</span> is a distance metric learning algorithm designed to improve the performance of <span style="color: #6FA8FF">**$k$-Nearest Neighbor ($k$-NN)**</span> classification. It learns a transformation $f(x)$ such that similar points are pulled together and dissimilar points are pushed apart.

The goal is to maintain a margin $r$ between "target neighbors" (same class) and "impostors" (different class):

$$L = \max(0, \|f(x) - f(x^+)\|_2 - \|f(x) - f(x^-)\|_2 + r)$$

* <span style="color: #6FA8FF">**$x$ (Anchor):**</span> The current data point.
* <span style="color: #6FA8FF">**$x^+$ (Target Neighbor):**</span> A point belonging to the **same class** ($C_x$) that we want to keep close.
* <span style="color: #6FA8FF">**$x^-$ (Impostor):**</span> A point belonging to a **different class** that we want to push outside the margin.
* <span style="color: #6FA8FF">**$r$ (Margin):**</span> A positive constant $(r > 0)$ defining the required safety gap.

To make the learning process more robust and efficient, the algorithm focuses on the most challenging examples:

* <span style="color: #6FA8FF">**Hardest Target Neighbor ($x^+$):**</span> The point in the same class that is currently **farthest** from the anchor.
    $$x^+ = \arg\max_{y \in C_x} \|f(x) - f(y)\|_2$$
* <span style="color: #6FA8FF">**Hardest Impostor ($x^-$):**</span> The point in a different class that is currently **closest** to the anchor.
    $$x^- = \arg\min_{y \notin C_x} \|f(x) - f(y)\|_2$$

**Key Objective:** The model effectively learns a metric where for every point, its $k$-nearest neighbors belong to the same class, while points from other classes are separated by a wide margin.
