# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 04 Matrix Completion</span>

The goal of <span style="color: #6FA8FF">**Matrix Completion**</span> is to recover missing entries from a partially observed matrix $M \in \mathbb{R}^{n \times m}$.

* **Objective:** Given a set of observed entries $\Omega$, estimate the complete matrix.
* **Key Assumption:** The underlying matrix is <span style="color: #6FA8FF">**low-rank**</span>, meaning the data lies on a low-dimensional structure.

<span style="color: #6FA8FF">**A. Convex Relaxation:**</span> Since rank minimization is NP-hard, we use <span style="color: #6FA8FF">**Nuclear Norm Minimization**</span> as a convex proxy, which minimizes the sum of singular values (nuclear norm) subject to the observed data constraints.

<span style="color: #6FA8FF">**B. Matrix Factorization (Non-Convex):**</span> Decompose the target matrix into the product of two low-rank factors: $X \in \mathbb{R}^{n \times r}$ and $Y \in \mathbb{R}^{m \times r}$. The objective function is defined as $\min_{X, Y} \|P_\Omega(XY^T) - P_\Omega(M)\|_F^2$, where $P_\Omega$ is the projection onto the observed entries.

<span style="color: #6FA8FF">**C. Alternating Minimization**:</span> A common algorithm to solve the factorization problem by iteratively updating $X$ and $Y$:
For $t = 1, 2, \dots, T$:

1. **Update $X_t$:** Fix $Y_{t-1}$, solve $\min_X \|P_\Omega(XY_{t-1}^T) - P_\Omega(M)\|_F^2$.
2. **Update $Y_t$:** Fix $X_t$, solve $\min_Y \|P_\Omega(X_tY^T) - P_\Omega(M)\|_F^2$.

To understand the convergence of non-convex methods, we analyze the stationary points where $\nabla f(x) = 0$:

| Hessian Condition $\nabla^2 f(x)$ | Point Characterization |
| :--- | :--- |
| $\nabla^2 f(x) \succ 0$ | **Local Minimum** |
| $\nabla^2 f(x) \prec 0$ | **Local Maximum** |
| Both positive & negative eigenvalues | <span style="color: #6FA8FF">**Strict Saddle Point**</span> |
| $\nabla^2 f(x) \succeq 0$ | **Local Minimum** or <span style="color: #6FA8FF">**Flat Saddle Point**</span> |

A function $f$ is considered <span style="color: #6FA8FF">**Strict Saddle**</span> if it contains no "flat" saddle points.
