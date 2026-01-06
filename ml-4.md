# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 04 Matrix Completion</span>

### <span style="color: #6ED3C5">Matrix Completion</span>

- given a partially observed matrix, recover the missing entries
- <span style="color: #6FA8FF">**low-rank assumption:**</span> the underlying complete matrix is low-rank
- <span style="color: #6FA8FF">**convex relaxation:**</span> (e.g., <span style="color: #6FA8FF">**nuclear norm minimization**</span>) relax the rank constraint to a convex problem (e.g., minimize nuclear norm, sum of singular values, subject to data constraints)
- <span style="color: #6FA8FF">**matrix factorization:**</span> decompose the matrix into product of two low-rank matrices
- <span style="color: #6FA8FF">**alternating minimization:**</span> iteratively update the low-rank factors
  - $\Omega$: set of observed entries
  - $P_\Omega(M)$: projection of matrix $M$ onto observed entries
  - objective: minimize $\|P_\Omega(XY^T)-P_\Omega(M)\|_F^2$ over $X,Y\in\mathbb{R}^{n\times r}$
  - for $t=1,2,\ldots,T$
    - fix $Y_{t-1}$, update $X_t$ by solving $\min_X\|P_\Omega(XY_{t-1}^T)-P_\Omega(M)\|_F^2$
    - fix $X_t$, update $Y_t$ by solving $\min_Y\|P_\Omega(X_tY^T)-P_\Omega(M)\|_F^2$
- **singular value thresholding:** apply soft-thresholding to singular values

for stationary points $\nabla f(x)=0$

- if $\nabla^2f(x)\succ 0$, local minimum
- if $\nabla^2f(x)\prec 0$, local maximum
- if $\nabla^2f(x)$ has both positive and negative eigenvalues, <span style="color: #6FA8FF">**strict saddle point**</span>
- if $\nabla^2f(x)\succeq 0$, local minimum or <span style="color: #6FA8FF">**flat saddle point**</span>

$f$ is <span style="color: #6FA8FF">**strict saddle**</span> if it does not have any flat saddle points
