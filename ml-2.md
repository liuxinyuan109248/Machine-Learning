# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 02 Unsupervised Learning and Gradient Descent</span>

### <span style="color: #6ED3C5">Unsupervised Learning</span>

- Clustering
- Principal Component Analysis (PCA)
  - find the directions of maximum variance in high-dimensional data
  - the best-fit line minimizes the sum of squared distances from the points to the line
- Generative Models
- Anomaly Detection
- Dimensionality Reduction

### <span style="color: #6ED3C5">Smooth and Convex Optimization</span>

<span style="color: #6FA8FF">**Convexity:**</span> If $f$ is twice continuously differentiable, then the following statements are equivalent:

- $f(tx+(1-t)y)\le tf(x)+(1-t)f(y),\forall t\in[0,1],x,y$,
- $f(y)\ge f(x)+\langle\nabla f(x),y-x\rangle,\forall x,y$,
- gradient is non-decreasing,
- Hessian is positive semi-definite.

<span style="color: #6FA8FF">**$L$-Lipschitz Continuity:**</span> $|f(y)-f(x)|\le L\|y-x\|,\forall x,y$

<span style="color: #6FA8FF">**$L$-Smoothness / Gradient Lipschitz Continuity:**</span> If $f$ is twice continuously differentiable, then the following statements are equivalent:

- **(a):** $|\nabla f(y)-\nabla f(x)|\le L\|y-x\|,\forall x,y$,
- **(b):** $|f(y)-f(x)-\langle\nabla f(x),y-x\rangle|\le\dfrac{L}{2}\|y-x\|^2,\forall x,y$,
- **(c):** $-L\le\lambda_{\min}(\nabla^2f(x))\le\lambda_{\max}(\nabla^2f(x))\le L$, or equivalently, $-LI\preceq\nabla^2f(x)\preceq LI,\forall x$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

- **(a)$\Rightarrow$(b):**
  $$f(y)-f(x)=\int_0^1\langle\nabla f(x+t(y-x)),y-x\rangle dt=\langle\nabla f(x),y-x\rangle+\int_0^1\langle\nabla f(x+t(y-x))-\nabla f(x),y-x\rangle dt$$
  $$
  \begin{aligned}
  \implies|f(y)-f(x)-\langle\nabla f(x),y-x\rangle|\le&\int_0^1|\langle\nabla f(x+t(y-x))-\nabla f(x),y-x\rangle| dt \\
  \le&\int_0^1L\|t(y-x)\|\cdot\|y-x\| dt=\dfrac{L}{2}\|y-x\|^2 \\
  \end{aligned}
  $$
- **(b)$\Rightarrow$(c):**
  $$|\langle\nabla f(x)-\nabla f(y),x-y\rangle|=|f(y)-f(x)-\langle\nabla f(x),y-x\rangle|+|f(x)-f(y)-\langle\nabla f(y),x-y\rangle|\le L\|x-y\|^2$$
  $$\implies\left|\left\langle\frac{\nabla f(x)-\nabla f(y)}{\|x-y\|},\frac{x-y}{\|x-y\|}\right\rangle\right|\le L\implies -L\le\lambda_{\min}(\nabla^2f(x))\le\lambda_{\max}(\nabla^2f(x))\le L$$
  (otherwise, let $y=x+\epsilon v$, where $v$ is the eigenvector corresponding to $\lambda_{\max}(\nabla^2f(x))$ or $\lambda_{\min}(\nabla^2f(x))$, and let $\epsilon\to 0$)
- **(c)$\Rightarrow$(a):** $\|\nabla f(y)-\nabla f(x)\|=\left\|\int_0^1\nabla^2f(x+t(y-x))(y-x) dt\right\|\le\int_0^1\|\nabla^2f(x+t(y-x))\|\cdot\|y-x\| dt\le L\|y-x\|$

</details>

<span style="color: #6FA8FF">**Proposition:**</span> If $f$ is $L$-smooth, then for $y=x-\eta\nabla f(x)$ and $\eta<\dfrac{2}{L}$, we have $f(y)\le f(x)$.

<span style="color: #6FA8FF">**Proof:**</span> $f(y)-f(x)\le\langle\nabla f(x),y-x\rangle+\dfrac{L}{2}\|y-x\|^2=\langle\nabla f(x),-\eta\nabla f(x)\rangle+\dfrac{L}{2}\|-\eta\nabla f(x)\|^2=\eta\left(\dfrac{L\eta}{2}-1\right)\|\nabla f(x)\|^2\le 0$

<span style="color: #6FA8FF">**Proposition:**</span> If $f$ is $L$-smooth, then for $y=x-\dfrac{1}{L}\nabla f(x)$, we have $f(y)\le f(x)-\dfrac{1}{2L}\|\nabla f(x)\|^2$. (proof omitted)

<span style="color: #6FA8FF">**Proposition:**</span> If $f$ is twice continuously differentiable, then the following statements are equivalent:

- **(a):** $\alpha\le\lambda_{\min}(\nabla^2f(x))\le\lambda_{\max}(\nabla^2f(x))\le\beta$, or equivalently, $\alpha I\preceq\nabla^2f(x)\preceq\beta I,\forall x$,
- **(b):** $\alpha\|x-y\|^2\le\langle\nabla f(x)-\nabla f(y),x-y\rangle\le\beta\|x-y\|^2,\forall x,y$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

- **(a)$\Rightarrow$(b):**
  $$
  \begin{aligned}
  &\langle\nabla f(x)-\nabla f(y),x-y\rangle=\left\langle\int_0^1\nabla^2f(x+t(y-x))(y-x)dt,y-x\right\rangle\\
  =&\int_0^1\langle\nabla^2f(x+t(y-x))(y-x)dt,y-x\rangle\in[\alpha\|x-y\|^2,\beta\|x-y\|^2]\\
  \end{aligned}
  $$
- **(b)$\Rightarrow$(a):** For any unit vector $v$, let $y=x+\epsilon v$, then
  $$\alpha\epsilon^2\le\langle\nabla f(x+\epsilon v)-\nabla f(x),\epsilon v\rangle\le\beta\epsilon^2\implies\alpha\le\left\langle\frac{\nabla f(x+\epsilon v)-\nabla f(x)}{\epsilon},v\right\rangle\le\beta$$
  Let $\epsilon\to 0$, we have $\alpha\le v^T\nabla^2f(x)v\le\beta$. Since $v$ is arbitrary, $\alpha I\preceq\nabla^2f(x)\preceq\beta I$.

</details>

<span style="color: #6FA8FF">**Theorem:**</span> If $f$ is twice continuously differentiable, then the following statements are equivalent:

- **(a):** $0\le f(y)-f(x)-\langle\nabla f(x),y-x\rangle\le\dfrac{L}{2}\|x-y\|^2,\forall x,y$,
- **(b):** $f(y)-f(x)-\langle\nabla f(x),y-x\rangle\ge\dfrac{1}{2L}\|\nabla f(x)-\nabla f(y)\|^2,\forall x,y$,
- **(c):** $0\le\langle\nabla f(x)-\nabla f(y),x-y\rangle\le L\|x-y\|^2,\forall x,y$,
- **(d):** $\langle\nabla f(x)-\nabla f(y),x-y\rangle\ge\dfrac{1}{L}\|\nabla f(x)-\nabla f(y)\|^2,\forall x,y$,
- **(e):** $0\le\lambda_{\min}(\nabla^2f(x))\le\lambda_{\max}(\nabla^2f(x))\le L$, or equivalently, $0\preceq\nabla^2f(x)\preceq LI,\forall x$,
- **(f):** $f$ is convex and $L$-smooth.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

- **(a)$\Leftrightarrow$(f):** by the definition of convexity and $L$-smoothness.
- **(c)$\Leftrightarrow$(e):** apply the previous proposition with $\alpha=0$ and $\beta=L$.
- **(e)$\Leftrightarrow$(f):** by the definition of convexity and $L$-smoothness.
- **(f)$\Rightarrow$(b):** Let $g(x)=f(x)-\langle\nabla f(y),x\rangle,\forall x$, then $g$ is also convex and $L$-smooth. Since $\nabla g(x)=\nabla f(x)-\nabla f(y)$ and $\nabla g(y)=\nabla f(y)-\nabla f(y)=0$, $y$ is a minimizer of $g$. Therefore, since $g$ is $L$-smooth,
  $$g(y)\le g\left(x-\frac{1}{L}\nabla g(x)\right)\le g(x)-\frac{1}{2L}\|\nabla g(x)\|^2\implies f(y)-\langle\nabla f(y),y\rangle\le f(x)-\langle\nabla f(y),x\rangle-\frac{1}{2L}\|\nabla f(x)-\nabla f(y)\|^2$$
  $$\implies\frac{1}{2L}\|\nabla f(x)-\nabla f(y)\|^2\le f(x)-f(y)-\langle\nabla f(y),x-y\rangle$$
  Swapping $x$ and $y$ gives (b).
- **(b)$\Rightarrow$(d):** Swapping $x$ and $y$ and adding the two inequalities gives (d).
- **(d)$\Rightarrow$(c):**
  $$\langle\nabla f(x)-\nabla f(y),x-y\rangle\ge\frac{1}{L}\|\nabla f(x)-\nabla f(y)\|^2\ge 0;$$
  $$
  \begin{aligned}
  &L\|x-y\|^2=L\|x-y\|^2+\frac{1}{L}\|\nabla f(x)-\nabla f(y)\|^2-\frac{1}{L}\|\nabla f(x)-\nabla f(y)\|^2\\
  \ge&2\langle\nabla f(x)-\nabla f(y),x-y\rangle-\frac{1}{L}\|\nabla f(x)-\nabla f(y)\|^2\ge\langle\nabla f(x)-\nabla f(y),x-y\rangle\\
  \end{aligned}
  $$

</details>

<span style="color: #6FA8FF">**$\mu$-Strongly Convexity:**</span> If $f$ is twice continuously differentiable, then the following statements are equivalent:

- **(a):** $f(y)\ge f(x)+\langle\nabla f(x),y-x\rangle+\dfrac{\mu}{2}\|y-x\|^2,\forall x,y$,
- **(b):** $\langle\nabla f(x)-\nabla f(y),x-y\rangle\ge\mu\|x-y\|^2,\forall x,y$,
- **(c):** $\lambda_{\min}(\nabla^2f(x))\ge\mu$, or equivalently, $\nabla^2f(x)\succeq\mu I,\forall x$,
- **(d):** $f(\alpha x+(1-\alpha)y)\le\alpha f(x)+(1-\alpha)f(y)-\dfrac{\mu}{2}\alpha(1-\alpha)\|x-y\|^2,\forall x,y,\alpha\in[0,1]$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

- **(a)$\Rightarrow$(b):** Swapping $x$ and $y$ in (a) and adding the two inequalities gives (b).
- **(b)$\Rightarrow$(c):** Apply the previous proposition with $\alpha=\mu$.
- **(c)$\Rightarrow$(a):** Let $g(x)=f(x)-\dfrac{\mu}{2}\|x\|^2$, then $\nabla g(x)=\nabla f(x)-\mu x$ and $\nabla^2g(x)=\nabla^2f(x)-\mu I\succeq 0$, therefore $g$ is convex. By the definition of convexity,
  $$g(y)\ge g(x)+\langle\nabla g(x),y-x\rangle\implies f(y)-\frac{\mu}{2}\|y\|^2\ge f(x)-\frac{\mu}{2}\|x\|^2+\langle\nabla f(x)-\mu x,y-x\rangle$$
  $$\implies f(y)\ge f(x)+\langle\nabla f(x),y-x\rangle+\frac{\mu}{2}(\|y\|^2-\|x\|^2)+\mu\langle x,y-x\rangle=f(x)+\langle\nabla f(x),y-x\rangle+\frac{\mu}{2}\|y-x\|^2$$
- **(a)$\Rightarrow$(d):** Let $g(\alpha)=f(\alpha x+(1-\alpha)y)-\alpha f(x)-(1-\alpha)f(y)+\dfrac{\mu}{2}\alpha(1-\alpha)\|x-y\|^2$, then $g(0)=g(1)=0$ and $g(\alpha)\le 0,\forall\alpha\in[0,1]$. Therefore $g'(1)\ge 0\implies\langle\nabla f(x),x-y\rangle-f(x)+f(y)-\dfrac{\mu}{2}\|x-y\|^2\ge 0$. Rewriting gives (d).
- **(d)$\Rightarrow$(a):**
  $$
  \begin{aligned}
  f(y)&\ge f(\alpha x+(1-\alpha)y)+\langle\nabla f(\alpha x+(1-\alpha)y),y-(\alpha x+(1-\alpha)y)\rangle+\frac{\mu}{2}\|\alpha x+(1-\alpha)y-y\|^2\\
  &=f(\alpha x+(1-\alpha)y)+\alpha\langle\nabla f(\alpha x+(1-\alpha)y),y-x\rangle+\frac{\mu}{2}\alpha^2\|x-y\|^2\\
  \end{aligned}
  $$
  $$
  \begin{aligned}
  f(x)&\ge f(\alpha x+(1-\alpha)y)+\langle\nabla f(\alpha x+(1-\alpha)y),x-(\alpha x+(1-\alpha)y)\rangle+\frac{\mu}{2}\|\alpha x+(1-\alpha)y-x\|^2\\
  &=f(\alpha x+(1-\alpha)y)+(1-\alpha)\langle\nabla f(\alpha x+(1-\alpha)y),x-y\rangle+\frac{\mu}{2}(1-\alpha)^2\|x-y\|^2\\
  \end{aligned}
  $$
  $$\implies\alpha f(x)+(1-\alpha)f(y)\ge f(\alpha x+(1-\alpha)y)+\frac{\mu}{2}\alpha(1-\alpha)\|x-y\|^2$$
  Rearranging gives (a).

</details>

<span style="color: #6FA8FF">**Proposition:**</span> If $f$ is $\mu$-strongly convex, then $f$ is $\mu$-PL, i.e., $\dfrac{1}{2\mu}\|\nabla f(x)\|^2\ge f(x)-f(x^*),\forall x$, where $x^*=\arg\min f(x)$.

<span style="color: #6FA8FF">**Proof:**</span> $f(x^*)\ge f(x)+\langle\nabla f(x),x^*-x\rangle+\dfrac{\mu}{2}\|x^*-x\|^2\ge f(x)-\|\nabla f(x)\|\cdot\|x^*-x\|+\dfrac{\mu}{2}\|x^*-x\|^2\ge f(x)-\dfrac{1}{2\mu}\|\nabla f(x)\|^2$

### <span style="color: #6ED3C5">Convex Function Convergence</span>

<span style="color: #6FA8FF">**Theorem:**</span> If $f$ is convex and $L$-smooth and $x^*=\arg\min f(x)$, then running gradient descent with step size $\eta\le\dfrac{1}{L}$ gives $f(x_t)-f(x^*)\le\dfrac{\|x_0-x^*\|^2}{2\eta t}$, and therefore $T=\dfrac{L\|x_0-x^*\|^2}{2\eta L\epsilon}$ iterations are sufficient to achieve $f(x_T)-f(x^*)\le\epsilon$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

Since $f$ is $L$-smooth, $f(x_{i+1})\le f(x_i)-\eta\left(1-\dfrac{L\eta}{2}\right)\|\nabla f(x_i)\|^2\le f(x_i)-\dfrac{\eta}{2}\|\nabla f(x_i)\|^2$. By convexity, $f(x_i)\le f(x^*)+\langle\nabla f(x_i),x_i-x^*\rangle$.
Combining the two inequalities,
$$
\begin{aligned}
f(x_{i+1})&\le f(x_i)-\frac{\eta}{2}\|\nabla f(x_i)\|^2\le f(x^*)+\langle\nabla f(x_i),x_i-x^*\rangle-\frac{\eta}{2}\|\nabla f(x_i)\|^2 \\
&\le f(x^*)-\frac{1}{\eta}\langle x_{i+1}-x_i,x_i-x^*\rangle-\frac{1}{2\eta}\|x_{i+1}-x_i\|^2=f(x^*)+\frac{1}{2\eta}\|x_i-x^*\|^2-\frac{1}{2\eta}\|x_{i+1}-x^*\|^2 \\
\end{aligned}
$$
Take telescoping sum over $i=0,1,\ldots,t-1$, $\sum_{i=0}^{t-1}(f(x_{i+1})-f(x^*))\le\dfrac{1}{2\eta}(\|x_0-x^*\|^2-\|x_t-x^*\|^2)\le\dfrac{\|x_0-x^*\|^2}{2\eta}$. Since $f(x_i)$ is non-increasing, $f(x_t)-f(x^*)\le\dfrac{1}{t}\sum_{i=0}^{t-1}(f(x_{i+1})-f(x^*))\le\dfrac{\|x_0-x^*\|^2}{2\eta t}\implies f(x_T)-f(x^*)\le\epsilon$.

</details>

<span style="color: #6FA8FF">**Theorem:**</span> If $f$ is $\mu$-strongly convex and $L$-smooth and $x^*=\arg\min f(x)$, then running gradient descent with step size $\eta=\dfrac{1}{L}$ gives $f(x_t)-f(x^*)\le\left(1-\dfrac{\mu}{L}\right)^t(f(x_0)-f(x^*))$.

<span style="color: #6FA8FF">**Proof:**</span> $\dfrac{\mu}{L}(f(x_t)-f(x^*))\le\dfrac{1}{2L}\|\nabla f(x_t)\|^2\le f(x_t)-f(x_{t+1})\implies f(x_{t+1})-f(x^*)\le\left(1-\dfrac{\mu}{L}\right)(f(x_t)-f(x^*))$

<span style="color: #6FA8FF">**Proposition:**</span> If $f$ is $\mu$-strongly convex and $L$-smooth and $x^*=\arg\min f(x)$, then running gradient descent with step size $\eta=\dfrac{1}{L}$ gives $\|x_t-x^*\|^2\le\left(1-\dfrac{\mu}{L}\right)^t\|x_0-x^*\|^2$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

$$\frac{1}{2L}\|\nabla f(x_t)\|^2\le f(x_t)-f(x_{t+1})\le f(x_t)-f(x^*)\le\langle\nabla f(x_t),x_t-x^*\rangle-\frac{\mu}{2}\|x_t-x^*\|^2$$
$$\implies\frac{L}{2}\|x_t-x_{t+1}\|^2\le L\langle x_t-x_{t+1},x_t-x^*\rangle-\frac{\mu}{2}\|x_t-x^*\|^2\implies\|x_{t+1}-x^*\|^2\le\left(1-\frac{\mu}{L}\right)\|x_t-x^*\|^2$$

</details>

<span style="color: #6FA8FF">**Limitations of Gradient Descent:</span>**

- computing full gradient is expensive for large datasets
- could get stuck in saddle points

### <span style="color: #6ED3C5">Stochastic Gradient Descent (SGD)</span>

- randomly sample a mini-batch of data to compute an unbiased estimate of the gradient
- update parameters using the estimated gradient
- $x_{t+1}=x_t-\eta g_t$, where $g_t$ is the estimated gradient at iteration $t$, $\mathbb{E}[g_t]=\nabla f(x_t)$
- the mini-batch size $|S|$ is usually 64, 128, 256, etc.
- help escape saddle points due to noise in the gradient estimate
- help get the right mini-batch statistics for batch normalization
