# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 06 Linear Methods</span>

### <span style="color: #6ED3C5">Linear Regression</span>

- exact minimizer: $w=(X^TX)^{-1}X^Ty$
- (stochastic) gradient descent method: $L(w,X,y)=\dfrac{1}{2N}\sum_{i=1}^N(w^Tx_i-y_i)^2$
  $$\nabla_wL(w,X,y)=\frac{1}{N}\sum_{i=1}^N(w^Tx_i-y_i)x_i,\quad w_{t+1}=w_t-\frac{\eta}{b}\sum_{i\in B_t}(w_t^Tx_i-y_i)x_i$$

<span style="color: #6FA8FF">**Perceptron Algorithm:**</span>

- binary classification
- does not converge if data is not linearly separable

<figure class="algorithm">
<pre><code>
P = inputs with label +1
N = inputs with label -1
initialize w randomly
while not converged do
  pick random x from P ∪ N
  if x in P and w * x < 0 then
    w = w + x
  else if x in N and w * x >= 0 then
    w = w - x
</code></pre>
</figure>

<span style="color: #6FA8FF">**Theorem:**</span> Assume there exists a vector $w^*$ with $\|w^*\|_2=1$ such that for every $(x_i,y_i)$ in the training set, $y_i(w^{*T}x_i)\ge\gamma>0$. Then, the perceptron algorithm converges after at most $\dfrac{R^2}{r^2}$ mistakes, where $R=\max_i\|x_i\|_2$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

$$w_{t+1}^Tw^*=w_t^Tw^*+y_i(w^{*T}x_i)\ge w_t^Tw^*+\gamma\implies w_{t+1}^Tw^*\ge t\gamma\implies\|w_{t+1}\|=\|w_{t+1}\|\cdot\|w^*\|\ge w_{t+1}^Tw^*\ge t\gamma$$
$$\|w_{t+1}\|_2^2=\|w_t+y_ix_i\|_2^2=\|w_t\|_2^2+2y_iw_t^Tx_i+\|x_i\|_2^2\le\|w_t\|_2^2+R^2\implies\|w_{t+1}\|_2^2\le tR^2$$
Combining the above two inequalities, we have $t^2\gamma^2\le\|w_{t+1}\|_2^2\le tR^2\implies t\le\dfrac{R^2}{\gamma^2}$, which completes the proof.

</details>

### <span style="color: #6ED3C5">Logistic Regression</span>

<span style="color: #6FA8FF">**Cross Entropy**:</span> $XE(p,q)=-\sum_{c}p(c)\log q(c)\ge-\sum_{c}p(c)\log p(c)=H(p)$

### <span style="color: #6ED3C5">Rigid Regression</span>

$L=\dfrac{1}{2N}\sum_{i=1}^N(w^Tx_i-y_i)^2+\dfrac{\lambda}{2}\|w\|_2^2\implies\nabla_wL=\dfrac{1}{N}\sum_{i=1}^N(w^Tx_i-y_i)x_i+\lambda w$

For each gradient step: $\widehat{w_{t+1}}=w_t-\dfrac{\eta}{N}\sum_{i=1}^N(w_t^Tx_i-y_i)x_i,\quad w_{t+1}=(1-\eta\lambda)\widehat{w_{t+1}}$ until $w_{t+1}=w_t$

### <span style="color: #6ED3C5">LASSO Regression</span>

$L=\dfrac{1}{2N}\sum_{i=1}^N(w^Tx_i-y_i)^2+\lambda\|w\|_1\implies\nabla_wL=\dfrac{1}{N}\sum_{i=1}^N(w^Tx_i-y_i)x_i+\lambda\cdot\text{sign}(w)$

For each gradient step:
$\widehat{w_{t+1}}=w_t-\dfrac{\eta}{N}\sum_{i=1}^N(w_t^Tx_i-y_i)x_i,\quad (w_{t+1})_i=\begin{cases}(\widehat{w_{t+1}})_i-\eta\lambda&\text{if }(\widehat{w_{t+1}})_i>\eta\lambda\\0&\text{if }|(\widehat{w_{t+1}})_i|\le\eta\lambda\\(\widehat{w_{t+1}})_i+\eta\lambda&\text{if }(\widehat{w_{t+1}})_i<-\eta\lambda\end{cases}$

### <span style="color: #6ED3C5">Restricted Isometry Property (RIP) Condition</span>

$W\in\mathbb{R}^{n\times d}$ is <span style="color: #6FA8FF">**$(\epsilon,s)$-RIP**</span> if for every $x\in\mathbb{R}^d$ with $\|x\|_0\le s$, it holds that $(1-\epsilon)\|x\|_2^2\le\|Wx\|_2^2\le(1+\epsilon)\|x\|_2^2$.

<span style="color: #6FA8FF">**Theorem:**</span> Let $\epsilon<1$ and $W\in\mathbb{R}^{n\times d}$ be a $(\epsilon,2s)$-RIP matrix. Let $x\in\mathbb{R}^d$ be a vector with $\|x\|_0\le s$ and $y=Wx$ be the compression of $x$. Let $\tilde{x}=\arg\min_{v:Wv=y}\|v\|_0$ be the solution to the sparse recovery problem. Then $\tilde{x}=x$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

If not, then $\tilde{x}-x\ne 0$ and $\|\tilde{x}-x\|_0\le 2s$. Apply the RIP condition: $(1-\epsilon)\|\tilde{x}-x\|_2^2\le\|W(\tilde{x}-x)\|_2^2\le(1+\epsilon)\|\tilde{x}-x\|_2^2$. Since $W(\tilde{x}-x)=0$, we have $\|\tilde{x}-x\|_2^2=0$, which contradicts the assumption that $\tilde{x}\ne x$. Therefore, $\tilde{x}=x$.

</details>

<span style="color: #6FA8FF">**Proposition:**</span> Let $W\in\mathbb{R}^{n\times d}$ be a $(\epsilon,2s)$-RIP matrix. Then for any two disjoint sets $I,J$, both of size at most $s$, and for any vector $u\in\mathbb{R}^d$, we have $\langle Wu_I,Wu_J\rangle\le\epsilon\|u_I\|_2\|u_J\|_2$, where $u_I,u_J$ are the restrictions of $u$ to the coordinates in $I,J$, respectively.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

W.l.o.g., assume that $\|u_I\|_2=\|u_J\|_2=1$. Since $|I\cup J|\le 2s$, by the RIP condition, we have
$$\|W(u_I+u_J)\|_2^2\le(1+\epsilon)\|u_I+u_J\|_2^2=2(1+\epsilon),\quad \|W(u_I-u_J)\|_2^2\ge(1-\epsilon)\|u_I-u_J\|_2^2=2(1-\epsilon)$$
Subtracting the above two inequalities, we have $4\langle Wu_I,Wu_J\rangle\le 4\epsilon\implies\langle Wu_I,Wu_J\rangle\le\epsilon$, which completes the proof.

</details>

<span style="color: #6FA8FF">**Theorem:**</span> Let $\epsilon<\sqrt{2}-1$ and $W\in\mathbb{R}^{n\times d}$ be a $(\epsilon,2s)$-RIP matrix. Let $x\in\mathbb{R}^d$ be an arbitrary vector and denote $x_s\in\arg\min_{v:\|v\|_0\le s}\|x-v\|_1$ be the best $s$-sparse approximation of $x$. Let $y=Wx$ be the compression of $x$ and let $x^*=\arg\min_{v:Wv=y}\|v\|_1$ be the solution to the LASSO problem. Then $\|x^*-x\|_2\le\dfrac{2(1+\rho)}{\sqrt{s}(1-\rho)}\|x-x_s\|_1$, where $\rho=\dfrac{\sqrt{2}\epsilon}{1-\epsilon}$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

Let $h=x^*-x$. In $T_0$ we put the $s$ indices corresponding to the $s$ largest elements in absolute values of $x$. Let $T_0^c=[d]\backslash T_0$ be the complement of $T_0$. Next, $T_1$ will be the $s$ indices corresponding to the $s$ largest elements in absolute value of $h_{T_0^c}$. Let $T_{0,1}=T_0\cup T_1$ and $T_{0,1}^c=[d]\backslash T_{0,1}$. Similarly, we define $T_2,T_3,\ldots$.

<span style="color: #6FA8FF">**Claim 1:**</span> $\|h_{T_{0,1}^c}\|_2\le\|h_{T_0}\|_2+\dfrac{2}{\sqrt{s}}\|x-x_s\|_1$.

<span style="color: #6FA8FF">**Claim 2:**</span> $\|h_{T_{0,1}}\|_2\le\dfrac{2\rho}{\sqrt{s}(1-\rho)}\|x-x_s\|_1$.

Combining the two claims, we have
$$\|h\|_2\le\|h_{T_{0,1}}\|_2+\|h_{T_{0,1}^c}\|_2\le 2\|h_{T_{0,1}}\|_2+\frac{2}{\sqrt{s}}\|x-x_s\|_1\le\frac{2}{\sqrt{s}}\left(1+\frac{2\rho}{1-\rho}\right)\|x-x_s\|_1=\frac{2(1+\rho)}{\sqrt{s}(1-\rho)}\|x-x_s\|_1,$$
which completes the proof.

<span style="color: #6FA8FF">**Proof of Claim 1:**</span> Take $j>1$. For each $i\in T_j$ and $k\in T_{j-1}$, we have $|h_i|\le|h_k|$. Therefore, $\|h_{T_j}\|_2\le\sqrt{s}\|h_{T_j}\|_{\infty}\le\dfrac{\|h_{T_{j-1}}\|_1}{\sqrt{s}}$. Summing over all $j\ge 2$, we have $\|h_{T_{0,1}^c}\|_2\le\sum_{j\ge 2}\|h_{T_j}\|_2\le\dfrac{1}{\sqrt{s}}\sum_{j\ge 1}\|h_{T_j}\|_1=\dfrac{1}{\sqrt{s}}\|h_{T_0^c}\|_1$. Since $x^*$ is the solution to the LASSO problem, we have $\|x\|_1\ge\|x^*\|_1=\|x+h\|_1=\sum_{i\in T_0}|x_i+h_i|+\sum_{i\in T_0^c}|x_i+h_i|\ge\|x_{T_0}\|_1-\|h_{T_0}\|_1+\|h_{T_0^c}\|_1-\|x_{T_0^c}\|_1$.
$$\|x_{T_0^c}\|_1=\|x-x_s\|_1=\|x\|_1-\|x_{T_0}\|_1\implies\|h_{T_0^c}\|_1\le\|h_{T_0}\|_1+2\|x-x_s\|_1$$
$$\implies\|h_{T_{0,1}^c}\|_2\le\frac{1}{\sqrt{s}}\|h_{T_0^c}\|_1\le\frac{1}{\sqrt{s}}(\|h_{T_0}\|_1+2\|x-x_s\|_1)\le\|h_{T_0}\|_2+\frac{2}{\sqrt{s}}\|x-x_s\|_1$$

<span style="color: #6FA8FF">**Proof of Claim 2:**</span> By the RIP condition, we have
$$
\begin{aligned}
&(1-\epsilon)\|h_{T_{0,1}}\|_2^2\le\|Wh_{T_{0,1}}\|_2^2=\|Wh-Wh_{T_{0,1}^c}\|_2^2=\|Wh_{T_{0,1}^c}\|_2^2 \\
=&-\sum_{j\ge 2}\langle Wh_{T_{0,1}},Wh_{T_j}\rangle=-\sum_{j\ge 2}\langle Wh_{T_0}+Wh_{T_1},Wh_{T_j}\rangle=-\sum_{i=0,1}\sum_{j\ge 2}\langle Wh_{T_i},Wh_{T_j}\rangle.
\end{aligned}
$$
Since for every $i=0,1$ and $j\ge 2$, both $T_i$ and $T_j$ have size at most $s$ and are disjoint, by the previous lemma, we have
$$|\langle Wh_{T_i},Wh_{T_j}\rangle|\le\epsilon\|h_{T_i}\|_2\|h_{T_j}\|_2\implies(1-\epsilon)\|h_{T_{0,1}}\|_2^2\le\epsilon\sum_{i=0,1}\|h_{T_i}\|_2\sum_{j\ge 2}\|h_{T_j}\|_2$$
$$\sum_{i=0,1}\|h_{T_i}\|_2=\|h_{T_0}\|_2+\|h_{T_1}\|_2\le\sqrt{2}\|h_{T_{0,1}}\|_2,\quad \sum_{j\ge 2}\|h_{T_j}\|_2\le\frac{1}{\sqrt{s}}\sum_{j\ge 2}\|h_{T_{j-1}}\|_1=\frac{\|h_{T_0^c}\|_1}{\sqrt{s}}$$
$$\implies(1-\epsilon)\|h_{T_{0,1}}\|_2^2\le\epsilon\cdot\sqrt{2}\|h_{T_{0,1}}\|_2\cdot\frac{\|h_{T_0^c}\|_1}{\sqrt{s}}\implies\|h_{T_{0,1}}\|_2\le\frac{\sqrt{2}\epsilon}{1-\epsilon}\cdot\frac{\|h_{T_0^c}\|_1}{\sqrt{s}}$$
$$\|h_{T_0^c}\|_1\le\|h_{T_0}\|_1+2\|x-x_s\|_1\implies\|h_{T_{0,1}}\|_2\le\frac{\rho}{\sqrt{s}}(\|h_{T_0}\|_1+2\|x-x_s\|_1)\le\rho\|h_{T_0}\|_2+\frac{2\rho}{\sqrt{s}}\|x-x_s\|_1$$
$$\|h_{T_0}\|_2\le\|h_{T_{0,1}}\|_2\implies(1-\rho)\|h_{T_{0,1}}\|_2\le\frac{2\rho}{\sqrt{s}}\|x-x_s\|_1\implies\|h_{T_{0,1}}\|_2\le\frac{2\rho}{\sqrt{s}(1-\rho)}\|x-x_s\|_1$$

</details>

<span style="color: #6FA8FF">**Corollary:**</span> Let $\epsilon<\sqrt{2}-1$ and $W\in\mathbb{R}^{n\times d}$ be a $(\epsilon,2s)$-RIP matrix. Let $x\in\mathbb{R}^d$ be a vector with $\|x\|_0\le s$ and $y=Wx$ be the compression of $x$. Then $x=\arg\min_{v:Wv=y}\|v\|_0=\arg\min_{v:Wv=y}\|v\|_1$.

<span style="color: #6FA8FF">**Theorem:**</span> Let $\epsilon,\delta\in(0,1)$. Let $s\in[d]$ and $n\ge\dfrac{216s\log\dfrac{120d}{\epsilon\delta}}{\epsilon^2}$. Let $W\in\mathbb{R}^{n\times d}$ be a random matrix whose entries are i.i.d. samples from $\mathcal{N}\left(0,\dfrac{1}{n}\right)$. Then with probability of at least $1−\delta$ over the choice of $W$, the matrix $W$ is $(\epsilon,s)$-RIP.
