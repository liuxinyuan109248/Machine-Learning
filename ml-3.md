# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 03 SGD, SVRG and Mirror Descent</span>

### <span style="color: #6ED3C5">Stochastic Gradient Descent (SGD)</span>

<span style="color: #6FA8FF">**Theorem:**</span> If $f$ is convex and $L$-smooth and $x^*=\arg\min f(x)$, then running SGD with variance $Var(g_t)\le\sigma^2$ with step size $\eta\le\dfrac{1}{L}$ gives $\mathbb{E}[f(\overline{x_t})]\le f(x^*)+\dfrac{\|x_0-x^*\|^2}{2\eta t}+\eta\sigma^2$ where $\overline{x_t}=\dfrac{1}{t}\sum_{i=1}^tx_i$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>
  
First we prove that $\mathbb{E}[f(x_{t+1})]\le f(x^*)+\dfrac{1}{2\eta}\cdot\mathbb{E}[\|x_t-x^*\|^2-\|x_{t+1}-x^*\|^2]+\eta\sigma^2$. Since $f$ is $L$-smooth,
$$
\begin{aligned}
&f(x_{t+1})-f(x^*)-\frac{1}{2\eta}(\|x_t-x^*\|^2-\|x_{t+1}-x^*\|^2) \\
\le&f(x_t)+\langle\nabla f(x_t),x_{t+1}-x_t\rangle+\frac{L}{2}\|x_{t+1}-x_t\|^2-f(x^*)-\frac{1}{2\eta}(\|x_t-x^*\|^2-\|x_t-x^*-\eta g_t\|^2) \\
\le&\langle\nabla f(x_t),x_t-x^*\rangle+\langle\nabla f(x_t),x_{t+1}-x_t\rangle+\frac{L\eta^2}{2}\|g_t\|^2-\langle g_t,x_t-x^*\rangle+\frac{\eta}{2}\|g_t\|^2 \\
=&\langle\nabla f(x_t),x_t-x^*\rangle-\eta\langle\nabla f(x_t),g_t\rangle+\frac{\eta}{2}(1+L\eta)\|g_t\|^2-\langle g_t,x_t-x^*\rangle \\
\end{aligned}
$$
Taking expectation w.r.t. $g_t$ on both sides,
$$
\begin{aligned}
&\mathbb{E}[f(x_{t+1})]-f(x^*)-\frac{1}{2\eta}\cdot\mathbb{E}[\|x_t-x^*\|^2-\|x_{t+1}-x^*\|^2] \\
\le&\langle\nabla f(x_t),x_t-x^*\rangle-\eta\langle\nabla f(x_t),\nabla f(x_t)\rangle+\frac{\eta}{2}(1+L\eta)(\|\nabla f(x_t)\|^2+\sigma^2)-\langle\nabla f(x_t),x_t-x^*\rangle \\
=&\frac{\eta}{2}(1+L\eta)\sigma^2-\frac{\eta}{2}(1-L\eta)\|\nabla f(x_t)\|^2\le\eta\sigma^2, \\
\end{aligned}
$$
finishing the proof of the first part. Now take telescoping sum over $t=0,1,\ldots,t-1$,
$$\sum_{i=0}^{t-1}(\mathbb{E}[f(x_{i+1})]-f(x^*))\le\frac{1}{2\eta}(\|x_0-x^*\|^2-\mathbb{E}[\|x_t-x^*\|^2])+\eta t\sigma^2\le\frac{\|x_0-x^*\|^2}{2\eta}+\eta t\sigma^2$$
By convexity of $f$, $\mathbb{E}[f(\overline{x_t})]\le\dfrac{1}{t}\sum_{i=0}^{t-1}\mathbb{E}[f(x_{i+1})]\le f(x^*)+\dfrac{\|x_0-x^*\|^2}{2\eta t}+\eta\sigma^2$.
  
</details>

<span style="color: #6FA8FF">**Proposition:**</span> SGD achieves an $\epsilon$-optimal solution in $O(1/\epsilon^2)$ iterations, or equivalently, the convergence rate is $O(1/\sqrt{t})$.

<span style="color: #6FA8FF">**Proof:**</span> If we set $t=\dfrac{2\sigma^2\|x_0-x^*\|^2}{\epsilon^2},\eta=\dfrac{\epsilon}{2\sigma^2}$, then $\mathbb{E}[f(\overline{x_t})]-f(x^*)\le\dfrac{\|x_0-x^*\|^2}{2\eta t}+\eta\sigma^2=\epsilon$.

### <span style="color: #6ED3C5">Stochastic Variance Reduced Gradient (SVRG)</span>

- for $s=1,2,\ldots,S$
  - set $\tilde{x}=\tilde{x}_{s-1}$
  - compute full gradient $\tilde{\mu}=\nabla f(\tilde{x})=\dfrac{1}{n}\sum_{i=1}^n\nabla f_i(\tilde{x})$
  - set $x_0=\tilde{x}$
  - for $t=1,2,\ldots,m$
    - randomly pick $i_t\in\{1,2,\ldots,n\}$
    - compute $g_t=\nabla f_{i_t}(x_{t-1})-\nabla f_{i_t}(\tilde{x})+\tilde{\mu}$
    - update $x_t=x_{t-1}-\eta g_t$
  - randomly pick $\tilde{x}_s$ from $\{x_0,x_1,\ldots,x_{m-1}\}$

<span style="color: #6FA8FF">**Theorem:**</span> If each $f_i$ is $\mu$-strongly convex and $L$-smooth, then SVRG achieves linear convergence rate $\dfrac{2L\eta}{1-2L\eta}+\dfrac{1}{m\mu\eta(1-2L\eta)}$, and is faster than GD if the condition number $\kappa=\dfrac{L}{\mu}$ is large.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

$$
\begin{aligned}
&\mathbb{E}[\|\nabla f_{i_t}(x_{t-1})-\nabla f_{i_t}(\tilde{x})+\tilde{\mu}\|^2] \\
\le&2\mathbb{E}[\|\nabla f_{i_t}(x_{t-1})-\nabla f_{i_t}(x^*)\|^2]+2\mathbb{E}[\|\nabla f_{i_t}(\tilde{x})-\nabla f_{i_t}(x^*)-\nabla f(\tilde{x})\|^2] \\
=&2\mathbb{E}[\|\nabla f_{i_t}(x_{t-1})-\nabla f_{i_t}(x^*)\|^2]+2\mathbb{E}[\|\nabla f_{i_t}(\tilde{x})-\nabla f_{i_t}(x^*)-\mathbb{E}[\nabla f_{i_t}(\tilde{x})-\nabla f_{i_t}(x^*)]\|^2] \\
\le&2\mathbb{E}[\|\nabla f_{i_t}(x_{t-1})-\nabla f_{i_t}(x^*)\|^2]+2\mathbb{E}[\|\nabla f_{i_t}(\tilde{x})-\nabla f_{i_t}(x^*)\|^2] \\
\le&4L\mathbb{E}[f_{i_t}(x_{t-1})-f_{i_t}(x^*)-\langle\nabla f_{i_t}(x^*),x_{t-1}-x^*\rangle]+4L\mathbb{E}[f_{i_t}(\tilde{x})-f_{i_t}(x^*)-\langle\nabla f_{i_t}(x^*),\tilde{x}-x^*\rangle] \\
=&4L(f(x_{t-1})-f(x^*)-\langle\nabla f(x^*),x_{t-1}-x^*\rangle)+4L(f(\tilde{x})-f(x^*)-\langle\nabla f(x^*),\tilde{x}-x^*\rangle) \\
=&4L(f(x_{t-1})-f(x^*)+f(\tilde{x})-f(x^*)) \\
\end{aligned}
$$
and therefore
$$
\begin{aligned}
\mathbb{E}[\|x_t-x^*\|^2]=&\|x_{t-1}-x^*\|^2-2\eta\langle x_{t-1}-x^*,\mathbb{E}[g_t]\rangle+\eta^2\cdot\mathbb{E}[\|g_t\|^2] \\
\le&\|x_{t-1}-x^*\|^2-2\eta\langle x_{t-1}-x^*,\nabla f(x_{t-1})\rangle+4L\eta^2(f(x_{t-1})-f(x^*)+f(\tilde{x})-f(x^*)) \\
\le&\|x_{t-1}-x^*\|^2-2\eta(f(x_{t-1})-f(x^*))+4L\eta^2(f(x_{t-1})-f(x^*)+f(\tilde{x})-f(x^*)) \\
=&\|x_{t-1}-x^*\|^2-2\eta(1-2L\eta)(f(x_{t-1})-f(x^*))+4L\eta^2(f(\tilde{x})-f(x^*)) \\
\end{aligned}
$$
If $\tilde{x}_s$ is randomly picked from $\{x_0,x_1,\ldots,x_{m-1}\}$, then
$$
\begin{aligned}
&\mathbb{E}[\|x_m-x^*\|^2]+2\eta(1-2L\eta)\cdot m\cdot\mathbb{E}[f(\tilde{x}_s)-f(x^*)] \\
\le&\mathbb{E}[\|x_m-x^*\|^2]+2\eta(1-2L\eta)\cdot\sum_{t=1}^m\mathbb{E}[f(x_{t-1})-f(x^*)] \\
\le&\|x_0-x^*\|^2+4mL\eta^2(f(\tilde{x})-f(x^*)) \\
=&\|\tilde{x}-x^*\|^2+4mL\eta^2(f(\tilde{x})-f(x^*)) \\
\le&\frac{2}{\mu}[f(\tilde{x})-f(x^*)]+4mL\eta^2(f(\tilde{x})-f(x^*)) \\
=&2\left(2mL\eta^2+\frac{1}{\mu}\right)(f(\tilde{x})-f(x^*)), \\
\end{aligned}
$$
and therefore $\mathbb{E}[f(\tilde{x}_s)-f(x^*)]\le\left(\dfrac{2L\eta}{1-2L\eta}+\dfrac{1}{m\mu\eta(1-2L\eta)}\right)(f(\tilde{x}_{s-1})-f(x^*))$.

<span style="color: #6FA8FF">**Gradient Descent:**</span> When $\kappa=\dfrac{L}{\mu}$ is large, $\kappa\log(1/\epsilon)$ iterations are needed to achieve an $\epsilon$-optimal solution.

<span style="color: #6FA8FF">**SVRG:**</span> Set $\eta=\dfrac{1}{10L},m=50\kappa$, then $\dfrac{2L\eta}{1-2L\eta}+\dfrac{1}{m\mu\eta(1-2L\eta)}=\dfrac{0.2}{0.8}+\dfrac{10L}{50\kappa\mu\times 0.8}=\dfrac{1}{4}\left(1+\dfrac{L}{\kappa\mu}\right)=\dfrac{1}{2}$. Therefore, $\log(1/\epsilon)$ iterations are needed to achieve an $\epsilon$-optimal solution.

</details>

<span style="color: #6FA8FF">**Stochastic Average Gradient (SAG):**</span> $x_{k+1}=x_k-\dfrac{\eta}{n}\sum_{i=1}^n y_i^k$, where $y_i^k=\nabla f_i(x_k)$ if $i=i_k$ and $y_i^k=y_i^{k-1}$ otherwise, and $i_k$ is randomly picked from $\{1,2,\ldots,n\}$.

<span style="color: #6FA8FF">**SAGA:**</span> $x_{k+1}=x_k-\eta\left(\nabla f_{i_k}(x_k)-y_{i_k}^k+\dfrac{1}{n}\sum_{i=1}^n y_i^k\right)$, where $y_i^k=\nabla f_i(x_k)$ if $i=i_k$ and $y_i^k=y_i^{k-1}$ otherwise, and $i_k$ is randomly picked from $\{1,2,\ldots,n\}$.

<span style="color: #6FA8FF">**Comparison of SVRG, SAG and SAGA:**</span>

| Method | Linear Convergence Rate | Stochastic Gradient Biased? | Storage Cost |
|--------|------------------|-----------------------------|--------------|
| SVRG   | $\dfrac{2L\eta}{1-2L\eta}+\dfrac{1}{m\mu\eta(1-2L\eta)}$ | No                          | O(d)         |
| SAG    | $1 - \min\left\{\dfrac{\mu}{16L},\dfrac{1}{10n}\right\}, \quad \eta = \dfrac{1}{16L}$ | Yes                         | O(nd)        |
| SAGA   | $1 - \min\left\{\dfrac{\mu}{3L},\dfrac{1}{4n}\right\}, \quad \eta = \dfrac{1}{3L}$  | No                          | O(nd)       |

### <span style="color: #6ED3C5">Mirror Descent</span>

<span style="color: #6FA8FF">**Bregman Divergence:</span>** $V_x(y)=w(y)-w(x)-\langle\nabla w(x),y-x\rangle,\forall x,y$

<span style="color: #6FA8FF">**Proposition:**</span> (proof omitted)

- If $w$ is 1-strongly convex, then $V_x(y)\ge\dfrac{1}{2}\|x-y\|^2$
- $\nabla V_x(y)=\nabla w(y)-\nabla w(x)$
- $\langle\nabla V_x(y),y-u\rangle=V_y(u)+V_x(y)-V_x(u)$
- If $w(x)=\dfrac{1}{2}\|x\|^2$, then $V_x(y)=\dfrac{1}{2}\|y-x\|^2$ (Euclidean distance)
- If $w(x) = \sum_i x_i \log x_i$, then $V_x(y)=\sum_i y_i \log \dfrac{y_i}{x_i}$ (KL divergence)
  
<span style="color: #6FA8FF">**Mirror Descent:**</span> $x_{k+1}=\text{Mirr}(\alpha\cdot\nabla f(x_k)),\text{Mirr}(g)=\arg\min_y(V_x(y)+\langle g,y-x\rangle)$

<span style="color: #6FA8FF">**Theorem:**</span> If $f$ is convex and $\rho$-Lipschitz continuous, and $w$ is 1-strongly convex, then mirror descent achieves an $\epsilon$-optimal solution in $O(\rho^2/\epsilon^2)$ iterations.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>
  
Taking derivative w.r.t. $y$ and setting it to zero, $\nabla V_{x_k}(x_{k+1})+\alpha\nabla f(x_k)=0$, and therefore
$$
\begin{aligned}
&\alpha\langle\nabla f(x_k),x_k-u\rangle \\
=&\alpha\langle\nabla f(x_k),x_k-x_{k+1}\rangle+\alpha\langle\nabla f(x_k),x_{k+1}-u\rangle \\
=&\alpha\langle\nabla f(x_k),x_k-x_{k+1}\rangle-\langle\nabla V_{x_k}(x_{k+1}),x_{k+1}-u\rangle \\
=&\alpha\langle\nabla f(x_k),x_k-x_{k+1}\rangle-V_{x_{k+1}}(u)-V_{x_k}(x_{k+1})+V_{x_k}(u) \\
\le&\alpha\langle\nabla f(x_k),x_k-x_{k+1}\rangle-\frac{1}{2}\|x_k-x_{k+1}\|^2-V_{x_{k+1}}(u)+V_{x_k}(u) \\
\le&\frac{\alpha^2}{2}\|\nabla f(x_k)\|^2+V_{x_k}(u)-V_{x_{k+1}}(u) \\
\end{aligned}
$$
Telescoping sum over $k=0,1,\ldots,T-1$, let $\overline{x}=\dfrac{1}{T}\sum_{k=1}^Tx_k,u=x^*=\arg\min f(x)$
$$\implies\alpha T(f(\overline{x})-f(x^*))\le\alpha\left(\sum_{k=1}^Tf(x_k)-Tf(x^*)\right)\le\alpha\sum_{k=0}^{T-1}\langle\nabla f(x_k),x_k-x^*\rangle\le\frac{\alpha^2}{2}\sum_{k=0}^{T-1}\|\nabla f(x_k)\|^2+V_{x_0}(x^*)$$
If $f$ is $\rho$-Lipschitz continuous, then $f(\overline{x})-f(x^*)\le\dfrac{\alpha\rho^2}{2}+\dfrac{V_{x_0}(x^*)}{\alpha T}$. Setting $\dfrac{\alpha\rho^2}{2}=\dfrac{V_{x_0}(x^*)}{\alpha T}=\dfrac{\epsilon}{2}$ gives
$$\alpha=\frac{\epsilon}{\rho^2},\quad T=\frac{2\rho^2V_{x_0}(x^*)}{\epsilon^2}\implies f(\overline{x})-f(x^*)\le\epsilon$$

</details>

<span style="color: #6FA8FF">**Total Variation Distance (TV):**</span> $TV(p,q)=\dfrac{1}{2}\|p-q\|_1=\max_{A\subseteq\Omega}|p(A)-q(A)|$

<span style="color: #6FA8FF">**Pinsker's Inequality:**</span> $KL(p\|q)\ge 2TV(p,q)^2$

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

Let $r(x)=\dfrac{p(x)}{q(x)}-1\ge -1$, then
$$
KL(p\|q)=\mathbb{E}_q[(1+r(x))\log(1+r(x))-r(x)]\ge\frac{1}{2}\mathbb{E}_q\left[\frac{r(x)^2}{1+r(x)/3}\right]\ge\frac{1}{2}\cdot\frac{(\mathbb{E}_q[|r(x)|])^2}{\mathbb{E}_q[1+r(x)/3]}=\frac{1}{2}(\mathbb{E}_q[|r(x)|])^2=2TV(p,q)^2
$$

</details>

<span style="color: #6FA8FF">**Corollary:**</span> If $f$ is convex, $\|\nabla f(x)\|_{\infty}\le\rho,\forall x$, and $w(x) = \sum_i x_i \log x_i$, then mirror descent achieves an $\epsilon$-optimal solution in $O(\rho^2/\epsilon^2)$ iterations.

<span style="color: #6FA8FF">**Proof Sketch:**</span> By Cauchy-Schwarz inequality, $\langle\nabla f(x_k),x_k-x_{k+1}\rangle\le\|\nabla f(x_k)\|_{\infty}\|x_k-x_{k+1}\|_1\le\rho\|x_k-x_{k+1}\|_1$. By Pinsker's inequality, $\alpha\langle\nabla f(x_k),x_k-x_{k+1}\rangle-V_{x_k}(x_{k+1})\le\alpha\rho\|x_k-x_{k+1}\|_1-\dfrac{1}{2}\|x_k-x_{k+1}\|_1^2\le\dfrac{\alpha^2\rho^2}{2}$. The rest of the proof is similar to the previous theorem.

<span style="color: #6FA8FF">**Multiplicative Weights Update (MWU):**</span> for $i=1,2,\ldots,n$, $x_{k+1}(i)=\dfrac{x_k(i)\exp(-\alpha\cdot\nabla f(x_k)(i))}{\sum_{j=1}^nx_k(j)\exp(-\alpha\cdot\nabla f(x_k)(j))}$

<span style="color: #6FA8FF">**Proposition:**</span> MWU is equivalent to mirror descent with $w(x)=\sum_i x_i\log x_i$ given the constraint that $x$ lies in the probability simplex $\Delta_n=\{x\in\mathbb{R}^n:x(i)\ge 0,\sum_{i=1}^nx(i)=1\}$.

<span style="color: #6FA8FF">**Proof Sketch:**</span> Taking derivative w.r.t. $y(i)$ and setting it to zero, $\log y(i)+1-\log x_k(i)+\alpha\nabla f(x_k)(i)+\lambda=0$, where $\lambda$ is the Lagrange multiplier for the constraint $\sum_{i=1}^ny(i)=1$. Therefore, $y(i)=x_k(i)\exp(-\alpha\cdot\nabla f(x_k)(i)-\lambda)$. Using the constraint gives the MWU update.

<span style="color: #6FA8FF">**Jensen-Shannon Divergence:**</span> $JS(p,q)=\dfrac{1}{2}KL\left(p\Big\|\dfrac{p+q}{2}\right)+\dfrac{1}{2}KL\left(q\Big\|\dfrac{p+q}{2}\right)$

<span style="color: #6FA8FF">**Wasserstein Distance:**</span> $W_1(p,q)=\inf_{\gamma\in\Gamma(p,q)}\mathbb{E}_{(X,Y)\sim\gamma}[\|X-Y\|]$, where $\Gamma(p,q)$ is the set of all joint distributions with marginals $p$ and $q$. Alternatively, by <span style="color: #6FA8FF">**Kantorovich-Rubinstein duality**</span>, $W_1(p,q)=\sup_{\|f\|_{Lip}\le 1}\mathbb{E}_{X\sim p}[f(X)]-\mathbb{E}_{Y\sim q}[f(Y)]$.

<span style="color: #6FA8FF">**Comparison of Probability Metrics:**</span> $W_1$ distance is "smoother" than TV and JS, which in turn are "smoother" than KL.

### <span style="color: #6ED3C5">Comparison of GD, SGD, SVRG and MD</span>

| Method | Assumptions on $f$ | Oracle Complexity to Achieve $\epsilon$-Optimal Solution |
|:---|:---|:---|
| **GD** | Convex, $L$-smooth | $O\left(\dfrac{L}{\epsilon}\right)$ |
| **GD** | $\mu$-Strongly Convex, $L$-smooth | $O\left(\dfrac{L}{\mu}\log(1/\epsilon)\right)$, i.e., linear convergence rate $1-\dfrac{\mu}{L}$ |
| **SGD** | Convex, $L$-smooth | $O\left(\dfrac{\sigma^2}{\epsilon^2}\right)$, where $\sigma^2$ is the variance of stochastic gradient |
| **SVRG** | $\mu$-Strongly Convex, $L$-smooth | Linear convergence rate $\dfrac{2L\eta}{1-2L\eta}+\dfrac{1}{m\mu\eta(1-2L\eta)}$ |
| **MD** | Convex, $\rho$-Lipschitz w.r.t. $\|\cdot\|_2$<br>(i.e., $\|\nabla f(x)\|_2 \le \rho$) | $O\left(\dfrac{\rho^2}{\epsilon^2}\right)$, where $w$ is 1-strongly convex w.r.t. $\|\cdot\|_2$<br>(e.g., $w(x)=\dfrac{1}{2}\|x\|_2^2$, $V_x(y)=\dfrac{1}{2}\|y-x\|_2^2$) |
| **MD** | Convex, $\rho$-Lipschitz w.r.t. $\|\cdot\|_\infty$<br>(i.e., $\|\nabla f(x)\|_\infty \le \rho$) | $O\left(\dfrac{\rho^2}{\epsilon^2}\right)$, where $w$ is 1-strongly convex w.r.t. $\|\cdot\|_1$<br>(e.g., $w(x)=\sum_i x_i \log x_i$, $V_x(y)=KL(y\|x)$) |
