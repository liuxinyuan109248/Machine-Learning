# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 09 Nearest Neighbors</span>

### <span style="color: #6ED3C5">Dimension Reduction</span>

<span style="color: #6FA8FF">**Johnson-Lindenstrauss Lemma:**</span> For any $0 < \epsilon < 1$ and any integer $n$, let $k$ be a positive integer such that $k \geq \dfrac{8 \ln n}{\epsilon^2}$. For any set $X$ of $n$ points in $\mathbb{R}^d$, there exists a map $f: \mathbb{R}^d \to \mathbb{R}^k$ such that for all $u, v \in X$,
$$(1 - \epsilon) \|u - v\|^2 \leq \|f(u) - f(v)\|^2 \leq (1 + \epsilon) \|u - v\|^2.$$

### <span style="color: #6ED3C5">Principal Component Analysis (PCA)</span>

- maximize variance $\sum_{i=1}^n (x_i \cdot v)^2 = v^T X X^T v$ subject to $\|v\| = 1$, where $X \in \mathbb{R}^{d \times n}$ is the data matrix with each $x_i$ as a column
- minimize reconstruction error $\sum_{i=1}^n \|x_i - (x_i \cdot v)v\|^2 = \sum_{i=1}^n \|x_i\|^2 - \sum_{i=1}^n (x_i \cdot v)^2$
- solution: $v$ is the eigenvector of $X X^T$ corresponding to the largest eigenvalue
- for $k$-dimensional subspace, take top $k$ eigenvectors

### <span style="color: #6ED3C5">Power Method</span>

- to find the largest eigenvalue and corresponding eigenvector of a matrix $A$
- start with a random vector $b_0$ and iterate $b_{k+1} = \dfrac{A b_k}{\|A b_k\|}$
- converges to the eigenvector corresponding to the largest eigenvalue
- compute eigenvalues and eigenvectors iteratively by deflation: after finding the largest eigenvalue $\lambda_1 = b_k^T A b_k$ and eigenvector $v_1 = b_k$, update $A \leftarrow A - \lambda_1 v_1 v_1^T$ and repeat

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
- overall LSH scheme: Use $L$ hash tables, each with a hash function from the amplified family. For a query point $q$, check all $L$ hash tables to find candidate points and verify them. Stop the search after $2L+1$ points have been checked.
- data structure size: $O(n L)$ (the size of each hash table is $O(n)$)

<span style="color: #6FA8FF">**Theorem:**</span> Given an $(R, cR, P_1, P_2)$-sensitive family $\mathcal{H}$ with $P_1 > P_2$ and parameters $k = \log_{1/P_2} n$ and $L = n^{\rho}$ where $\rho = \dfrac{\log(1/P_1)}{\log(1/P_2)}$, the above LSH scheme solves the randomized $c$-approximate $R$-near neighbor problem with space complexity $O(n^{1+\rho})$ and query time $O(n^{\rho})$. To be specific, if there exists $x \in X$ such that $\|x - q\| \leq R$, the scheme returns some $x' \in X$ such that $\|x' - q\| \leq cR$ with probability at least $\dfrac{1}{2} - \dfrac{1}{e}$.

<span style="color: #6FA8FF">**Proof:**</span> For a query point $q$, let $S$ be the set of points in $X$ such that $\|x - q\| \leq R$ and $T$ be the set of points in $X$ such that $\|x - q\| \geq cR$. We have

- $\Pr[\text{no point in } S \text{ is found}] \leq (1 - P_1^k)^L \leq e^{-L P_1^k} = e^{-n^{\rho} \cdot n^{-\rho}} = e^{-1}$
- $\mathbb{E}[\text{number of points in } T \text{ found}] \leq |T| \cdot L P_2^k \leq n \cdot n^{\rho} \cdot n^{-\rho} = L$

By Markov's inequality, the probability that more than $2L$ points in $T$ are found is at most $\dfrac{1}{2}$. Therefore, the probability that at least one point in $S$ is found and at most $2L$ points in $T$ are found is at least $\dfrac{1}{2} - \dfrac{1}{e}$.

### <span style="color: #6ED3C5">LSH for Euclidean Distance</span>

- hash function: $h_{a,b}(v) = \left\lfloor \dfrac{a \cdot v + b}{w} \right\rfloor$, where $a$ is a random vector with each entry drawn from $\mathcal{N}(0, 1)$, $b$ is drawn uniformly from $[0, w)$, and $w$ is a fixed width parameter
- collision probability: $\Pr[h_{a,b}(u) = h_{a,b}(v)] = p(d) = \displaystyle \int_0^w \dfrac{1}{d} f\left(\dfrac{t}{d}\right) \left(1 - \dfrac{t}{w}\right) dt$, where $d = \|u - v\|$ and $f(x) = \sqrt{\dfrac{2}{\pi}} e^{-x^2/2}$ is the PDF of the half-normal distribution

### <span style="color: #6ED3C5">Metric Learning</span>

<span style="color: #6FA8FF">**Neighborhood Components Analysis (NCA):**</span> Learn a linear transformation $A$ to minimize the leave-one-out classification error of 1-NN on the training set. The probability that point $x_i$ selects point $x_j$ as its neighbor is given by
$$p_{ij} = \dfrac{\exp(-\|f(x_i) - f(x_j)\|^2)}{\sum_{k \neq i} \exp(-\|f(x_i) - f(x_k)\|^2)},$$
where $f(x) = A x$. The objective is to maximize the expected number of correctly classified points $\sum_i \sum_{j \in C_i} p_{ij}$, where $C_i$ is the set of points with the same label as $x_i$.

### <span style="color: #6ED3C5">Large Margin Nearest Neighbor (LMNN)</span>

- $L = \max(0, \|f(x) - f(x^+)\|_2 - \|f(x) - f(x^-)\|_2 + r)$, where $r > 0$ is the margin, $x^+$ is a target neighbor (same class), and $x^-$ is an impostor (different class)
- pick the hard cases: $x^+ = \arg\max_{y \in C_x} \|f(x) - f(y)\|_2$ and $x^- = \arg\min_{y \notin C_x} \|f(x) - f(y)\|_2$
