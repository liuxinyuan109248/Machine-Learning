# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 05 VC Dimension</span>

### <span style="color: #6ED3C5">No Free Lunch Theorem</span>

<span style="color: #6FA8FF">**No Free Lunch Theorem:**</span> Let $A$ be any learning algorithm for the binary classification problem with respect to $0-1$ loss over domain set $\mathcal{X}$. Let $m<\dfrac{|\mathcal{X}|}{2}$ be any training set size. Then there exists a distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$ such that

1. there exists a function $f:\mathcal{X}\to\{0,1\}$ with $L_{\mathcal{D}}(f)=0$,
2. with probability at least $\dfrac{1}{7}$ over the choice of training set $S\sim\mathcal{D}^m$, we have $L_{\mathcal{D}}(A(S))\ge\dfrac{1}{8}$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

Let $C\subseteq\mathcal{X}$ be any subset of size $2m$. There are $T=2^{2m}$ distinct functions from $C$ to $\{0,1\}$, denote them by $f_1,f_2,\ldots,f_T$. For each $i\in[T]$, define a distribution $\mathcal{D}_i$ over $C\times\{0,1\}$ as follows:

- for $x\in C$, $\mathbb{P}_{\mathcal{D_i}}[(x,f_i(x))]=\dfrac{1}{|C|}=\dfrac{1}{2m}$
- for $x\notin C$, $\mathbb{P}_{\mathcal{D_i}}[(x,y)]=0,\forall y\in\{0,1\}$

Note that for each $i\in[T]$, $L_{\mathcal{D}_i}(f_i)=0$. First we show that for every learning algorithm $A$ that receives a training set of $m$ samples from $C\times\{0,1\}$ and returns a function $A(S):C\to\{0,1\}$, it holds that
$$\max_{i\in[T]}\mathbb{E}_{S\sim\mathcal{D}_i^m}[L_{\mathcal{D}_i}(A(S))]\ge\frac{1}{4}.$$
There are $k=(2m)^m$ distinct training sets of size $m$ that can be drawn from $C\times\{0,1\}$ (since each of the $m$ samples can take any of the $2m$ values in $C$). Denote these training sets by $S_1,S_2,\ldots,S_k$. For every $i\in[T]$ and $j\in[k]$, if $S_j=(x_1,x_2,\ldots,x_m)$, then we denote by $S_j^i$ the training set labeled according to $f_i$:
$$S_j^i=\{(x_1,f_i(x_1)),(x_2,f_i(x_2)),\ldots,(x_m,f_i(x_m))\}.$$
If the distribution is $\mathcal{D}_i$, then the training set $S$ is drawn uniformly from $\{S_1^i,S_2^i,\ldots,S_k^i\}$. Therefore,
$$\mathbb{E}_{S\sim\mathcal{D}_i^m}[L_{\mathcal{D}_i}(A(S))]=\frac{1}{k}\sum_{j=1}^kL_{\mathcal{D}_i}(A(S_j^i))$$
$$\implies\max_{i\in[T]}\mathbb{E}_{S\sim\mathcal{D}_i^m}[L_{\mathcal{D}_i}(A(S))]\ge\frac{1}{T}\sum_{i=1}^T\frac{1}{k}\sum_{j=1}^kL_{\mathcal{D}_i(A(S_j^i))}=\frac{1}{k}\sum_{j=1}^k\frac{1}{T}\sum_{i=1}^TL_{\mathcal{D}_i}(A(S_j^i))\ge\min_{j\in[k]}\frac{1}{T}\sum_{i=1}^TL_{\mathcal{D}_i}(A(S_j^i)).$$
Fix some $j\in[k]$. Denote $S_j=(x_1,x_2,\ldots,x_m)$ and let $v_1,v_2,\ldots,v_p$ be the samples in $C$ that do not appear in $S_j$. Then $p\ge m$. Therefore for every function $h:C\to\{0,1\}$ and every $i\in[T]$, we have
$$L_{\mathcal{D}_i}(h)=\frac{1}{2m}\sum_{x\in C}\mathbf{1}[h(x)\ne f_i(x)]\ge\frac{1}{2p}\sum_{r=1}^p\mathbf{1}[h(v_r)\ne f_i(v_r)]$$
and therefore
$$
\begin{aligned}
&\frac{1}{T}\sum_{i=1}^TL_{\mathcal{D}_i}(A(S_j^i))\ge\frac{1}{T}\sum_{i=1}^T\frac{1}{2p}\sum_{r=1}^p\mathbf{1}[A(S_j^i)(v_r)\ne f_i(v_r)] \\
=&\frac{1}{2p}\sum_{r=1}^p\frac{1}{T}\sum_{i=1}^T\mathbf{1}[A(S_j^i)(v_r)\ne f_i(v_r)]\ge\frac{1}{2}\min_{r\in[p]}\frac{1}{T}\sum_{i=1}^T\mathbf{1}[A(S_j^i)(v_r)\ne f_i(v_r)]. \\
\end{aligned}
$$
Fix some $r\in[p]$. We can partition all functions $f_1,f_2,\ldots,f_T$ into $\dfrac{T}{2}$ disjoint pairs, where for a pair $(f_i,f_{i'})$, we have that for every $c\in C,f_i(c)\ne f_{i'}(c)$ if and only if $c=v_r$. For each such pair, $S_j^i=S_j^{i'}$ and therefore
$$\mathbf{1}[A(S_j^i)(v_r)\ne f_i(v_r)]+\mathbf{1}[A(S_j^{i'})(v_r)\ne f_{i'}(v_r)]=1,$$
which implies that $\dfrac{1}{T}\sum_{i=1}^T\mathbf{1}[A(S_j^i)(v_r)\ne f_i(v_r)]=\dfrac{1}{2}$. Combining the above inequalities, we have shown that
$$\max_{i\in[T]}\mathbb{E}_{S\sim\mathcal{D}_i^m}[L_{\mathcal{D}_i}(A(S))]\ge\frac{1}{4}.$$
This implies that for every learning algorithm $A$ that receives a training set of $m$ samples from $\mathcal{X}\times\{0,1\}$, there exists a function $f:\mathcal{X}\to\{0,1\}$ and a distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$ such that $L_{\mathcal{D}}(f)=0$ and $\mathbb{E}_{S\sim\mathcal{D}^m}[L_{\mathcal{D}}(A(S))]\ge\dfrac{1}{4}$. Therefore with probability at least $\dfrac{1}{7}$ over the choice of training set $S\sim\mathcal{D}^m$, we have $L_{\mathcal{D}}(A(S))\ge\dfrac{1}{8}$.

</details>

### <span style="color: #6ED3C5">Empirical Risk Minimization (ERM)</span>

The goal of <span style="color: #6FA8FF">**Empirical Risk Minimization (ERM)**</span> is to find a hypothesis $h$ that minimizes the empirical risk on the training data: $h_{ERM}=\arg\min_{h\in\mathcal{H}}\dfrac{1}{m}\sum_{i=1}^mL(h(x_i),y_i)$, where $\mathcal{H}$ is the hypothesis space, $L$ is the loss function, and $(x_i,y_i)$ are the samples.

### <span style="color: #6ED3C5">Realizability Assumption</span>

The <span style="color: #6FA8FF">**realizability assumption**</span> states that there exists a hypothesis $h^* \in \mathcal{H}$ such that $L_{(\mathcal{D},f)}(h^*)=0$. This means that with probability $1$ over random samples, $S$, where the instances are drawn i.i.d. from $\mathcal{D}$ and labeled according to $f$, we have $L_S(h^*)=0$.

$\mathcal{D}^m$: the probability over $m$ samples drawn i.i.d. from $\mathcal{D}$

<span style="color: #6FA8FF">**Theorem:**</span> Let $\mathcal{H}$ be a finite hypothesis class. Let $\delta\in(0,1)$ and $\epsilon>0$. Let $m$ be an integer that satisfies $m\ge\dfrac{\log(|\mathcal{H}|/\delta)}{\epsilon}$. Then, for any labeling function $f$ and any distribution $\mathcal{D}$, for which the realizability assumption holds, with probability at least $1-\delta$ over the choice of an i.i.d. sample $S$ of size $m$, the hypothesis $h_S$ returned by the ERM algorithm satisfies $L_{(\mathcal{D},f)}(h_S)\le\epsilon$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

Let $S|_x=(x_1,x_2,\ldots,x_m)$ be the instances in the training set $S$. We would like to upper bound
$$\mathcal{D}^m(\{S|_x:L_{(\mathcal{D},f)}(h_S)>\epsilon\}).$$
Let $\mathcal{H}_B$ be the set of "bad" hypotheses in $\mathcal{H}$: $\mathcal{H}_B=\{h\in\mathcal{H}:L_{(\mathcal{D},f)}(h)>\epsilon\}$ and $M=\{S|_x:\exists h\in\mathcal{H}_B\text{ s.t. }L_S(h)=0\}$ be the set of misleading samples. Since the realizability assumption holds, if $L_{(\mathcal{D},f)}(h_S)>\epsilon$, then there must exist some $h\in\mathcal{H}_B$ such that $L_S(h)=0$. Therefore, $\{S|_x:L_{(\mathcal{D},f)}(h_S)>\epsilon\}\subseteq M$, and it is sufficient to upper bound
$$\mathcal{D}^m(M)=\mathcal{D}^m(\cup_{h\in\mathcal{H}_B}\{S|_x:L_S(h)=0\})\le\sum_{h\in\mathcal{H}_B}\mathcal{D}^m(\{S|_x:L_S(h)=0\}).$$
Fix some $h\in\mathcal{H}_B$. The event $L_S(h)=0$ is equivalent to the event that $\forall i\in[m],h(x_i)=f(x_i)$. Since the samples in $S$ are drawn i.i.d. from $\mathcal{D}$, we have
$$\mathcal{D}^m(\{S|_x:L_S(h)=0\})=\prod_{i=1}^m\mathcal{D}(\{x_i:h(x_i)=f(x_i)\}).$$
For each $i\in[m]$, since $h\in\mathcal{H}_B$, we have $\mathcal{D}(\{x_i:h(x_i)=f(x_i)\})=1-\mathcal{D}(\{x_i:h(x_i)\ne f(x_i)\})=1-L_{(\mathcal{D},f)}(h)\le 1-\epsilon$. Therefore for every $h\in\mathcal{H}_B$,
$$\mathcal{D}^m(\{S|_x:L_S(h)=0\})\le(1-\epsilon)^m\le e^{-\epsilon m}\implies\mathcal{D}^m(\{S|_x:L_{(\mathcal{D},f)}(h_S)>\epsilon\})\le\mathcal{D}^m(M)\le|\mathcal{H}_B|e^{-\epsilon m}\le|\mathcal{H}|e^{-\epsilon m}.$$
Setting $|\mathcal{H}|e^{-\epsilon m}\le\delta\implies m\ge\dfrac{\log(|\mathcal{H}|/\delta)}{\epsilon}$, we have $\mathcal{D}^m(\{S|_x:L_{(\mathcal{D},f)}(h_S)>\epsilon\})\le\delta$, which completes the proof.

</details>

### <span style="color: #6ED3C5">Probabilistic Approximately Correct (PAC) Learnability</span>

A hypothesis class $\mathcal{H}$ is <span style="color: #6FA8FF">**PAC learnable**</span> if there exists a function $m_{\mathcal{H}}:(0,1)^2\to\mathbb{N}$ and a learning algorithm with the following property: for every $\epsilon,\delta\in(0,1)$, for every distribution $\mathcal{D}$ over $\mathcal{X}$, and for every labeling function $f:\mathcal{X}\to\{0,1\}$, if the realizability assumption holds, then when running the learning algorithm on $m\ge m_{\mathcal{H}}(\epsilon,\delta)$ i.i.d. samples generated by $\mathcal{D}$ and labeled according to $f$, the algorithm returns a hypothesis $h$ such that, with probability at least $1-\delta$ (over the choice of the training samples), $L_{(\mathcal{D},f)}(h)\le\epsilon$.

<span style="color: #6FA8FF">**Corollary:**</span> Every finite hypothesis class is PAC learnable with sample complexity $m_{\mathcal{H}}(\epsilon,\delta)\le\left\lceil\dfrac{\log(|\mathcal{H}|/\delta)}{\epsilon}\right\rceil$.

<span style="color: #6FA8FF">**Theorem:**</span> Let $\mathcal{X}$ be an infinite domain set and let $\mathcal{H}$ be the set of all functions from $\mathcal{X}$ to $\{0,1\}$. Then, $\mathcal{H}$ is not PAC learnable.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

Assume by contradiction that $\mathcal{H}$ is PAC learnable. Choose some $\epsilon<\dfrac{1}{8}$ and $\delta<\dfrac{1}{7}$. By the PAC learnability assumption, there must be some learning algorithm $A$ and an integer $m=m(\epsilon,\delta)$, such that for every data-generating distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$, if for some function $f:\mathcal{X}\to\{0,1\},L_{\mathcal{D}}(f)=0$, then with probability at least $1-\delta$ over the choice of an i.i.d. sample $S$ of size $m$, the hypothesis $h=A(S)$ satisfies $L_{\mathcal{D}}(h)\le\epsilon$.

However, by No Free Lunch Theorem, there exists a distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$ such that there exists a function $f:\mathcal{X}\to\{0,1\}$ with $L_{\mathcal{D}}(f)=0$, and with probability at least $\dfrac{1}{7}> \delta$ over the choice of an i.i.d. sample $S$ of size $m$, the hypothesis $h=A(S)$ satisfies $L_{\mathcal{D}}(h)\ge \dfrac{1}{8}>\epsilon$, which contradicts the PAC learnability assumption.

</details>

### <span style="color: #6ED3C5">The Bayes Optimal Predictor</span>

Given a distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$, the <span style="color: #6FA8FF">**Bayes optimal predictor**</span> is the function $f_{\mathcal{D}}:\mathcal{X}\to\{0,1\}$ defined as $f_{\mathcal{D}}(x)=\mathbf{1}\left[\mathbb{P}[y=1|x]\ge\dfrac{1}{2}\right]$, where $\mathbb{P}[y=1|x]$ is the conditional probability that the label is $1$ given the instance $x$.

<span style="color: #6FA8FF">**Proposition:**</span> For every distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$ and every function $g:\mathcal{X}\to\{0,1\}$, it holds that $L_{\mathcal{D}}(f_{\mathcal{D}})\le L_{\mathcal{D}}(g)$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

For every $x\in\mathcal{X}$, let $\eta(x)=\mathbb{P}[y=1|x]$. Then,
$$\mathbb{P}_{y\sim\mathcal{D}|x}[f_{\mathcal{D}}(x)\ne y]=\min(\eta(x),1-\eta(x))\le\eta(x)\mathbf{1}[g(x)=0]+(1-\eta(x))\mathbf{1}[g(x)=1]=\mathbb{P}_{y\sim\mathcal{D}|x}[g(x)\ne y]$$
$$\implies L_{\mathcal{D}}(f_{\mathcal{D}})=\mathbb{P}_{x\sim\mathcal{D}_x}[\mathbb{P}_{y\sim\mathcal{D}|x}[f_{\mathcal{D}}(x)\ne y]]\le\mathbb{P}_{x\sim\mathcal{D}_x}[\mathbb{P}_{y\sim\mathcal{D}|x}[g(x)\ne y]]=L_{\mathcal{D}}(g)$$

</details>

### <span style="color: #6ED3C5">Agnostic PAC Learnability</span>

A hypothesis class $\mathcal{H}$ is <span style="color: #6FA8FF">**agnostically PAC learnable**</span> if there exists a function $m_{\mathcal{H}}:(0,1)^2\to\mathbb{N}$ and a learning algorithm with the property: for every $\epsilon,\delta\in(0,1)$, for every distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$, when running the learning algorithm on $m\ge m_{\mathcal{H}}(\epsilon,\delta)$ i.i.d. samples generated by $\mathcal{D}$, the algorithm returns a hypothesis $h$ such that, with probability at least $1-\delta$ (over the choice of the training samples), it holds that $L_{\mathcal{D}}(h)\le\min_{h'\in\mathcal{H}}L_{\mathcal{D}}(h')+\epsilon$.

### <span style="color: #6ED3C5">Error Decomposition</span>

Let $h_S$ be an ERM hypothesis. Then we can decompose its error as follows:
$$L_{\mathcal{D}}(h_S)=\underbrace{L_{\mathcal{D}}(h^*)}_{\text{approximation error}}+\underbrace{(L_{\mathcal{D}}(h_S)-L_{\mathcal{D}}(h^*))}_{\text{estimation error}}$$
where $h^*=\arg\min_{h\in\mathcal{H}}L_{\mathcal{D}}(h)$.

### <span style="color: #6ED3C5">Infinite-Size Classes Can Be Learnable</span>

<span style="color: #6FA8FF">**Theorem:**</span> Let $\mathcal{H}$ be the set of all threshold functions on the real line: $\mathcal{H}=\{h_a:\mathbb{R}\to\{0,1\}|a\in\mathbb{R},h_a(x)=\mathbf{1}[x\le a]\}$. Then, $\mathcal{H}$ is PAC learnable with sample complexity
$m_{\mathcal{H}}(\epsilon,\delta)\le\left\lceil\dfrac{\log(2/\delta)}{\epsilon}\right\rceil$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

Let $a^*$ be a threshold such that $h^*(x)=\mathbf{1}[x\le a^*]$ achieves $L_{\mathcal{D}}(h^*)=0$. Let $\mathcal{D}_x$ be the marginal distribution over the domain $\mathcal{X}$ and let $a_0<a^*<a_1$ be such that
$$\mathbb{P}_{x\sim\mathcal{D}_x}[a_0<x<a^*]=\mathbb{P}_{x\sim\mathcal{D}_x}[a^*<x<a_1]=\epsilon.$$
(If no such $a_0$ or $a_1$ exists, we can set $a_0=-\infty$ or $a_1=+\infty$.) Given a training set $S$, let
$$b_0=\max\{x:(x,1)\in S\},b_1=\min\{x:(x,0)\in S\}.$$
(If no such $x$ exists, we can set $b_0=-\infty$ or $b_1=+\infty$.) Let $b_S$ be a threshold corresponding to the ERM hypothesis $h_S$, which implies that $b_0<b_S<b_1$. A sufficient condition for $L_{\mathcal{D}}(h_S)\le\epsilon$ is that $b_0\ge a_0$ and $b_1\le a_1$. The probability that this condition is not satisfied can be upper bounded as follows:
$$\mathbb{P}_{S\sim\mathcal{D}^m}[L_{\mathcal{D}}(h_S)>\epsilon]\le\mathbb{P}_{S\sim\mathcal{D}^m}[\text{not }(b_0\ge a_0\text{ and }b_1\le a_1)]\le\mathbb{P}_{S\sim\mathcal{D}^m}[b_0<a_0]+\mathbb{P}_{S\sim\mathcal{D}^m}[b_1>a_1].$$
The event $b_0<a_0$ happens if and only if all samples in $S$ are not in the interval $(a_0,a^*)$, whose probability mass is $\epsilon$. Therefore,
$$\mathbb{P}_{S\sim\mathcal{D}^m}[b_0<a_0]=\mathbb{P}_{S\sim\mathcal{D}^m}[\forall(x,y)\in S,x\notin(a_0,a^*)]=(1-\epsilon)^m\le e^{-\epsilon m}.$$
Since $m\ge\dfrac{\log(2/\delta)}{\epsilon}$, we have $e^{-\epsilon m}\le\dfrac{\delta}{2}$ and similarly $\mathbb{P}_{S\sim\mathcal{D}^m}[b_1>a_1]\le\dfrac{\delta}{2}$. Therefore,
$$\mathbb{P}_{S\sim\mathcal{D}^m}[L_{\mathcal{D}}(h_S)>\epsilon]\le\mathbb{P}_{S\sim\mathcal{D}^m}[b_0<a_0]+\mathbb{P}_{S\sim\mathcal{D}^m}[b_1>a_1]\le\frac{\delta}{2}+\frac{\delta}{2}=\delta,$$
which completes the proof.

</details>

<span style="color: #6FA8FF">**Remark:**</span> By using a "smarter" or more specific ERM algorithm—one that consistently picks a threshold from only one side of the version space (e.g., always choosing the smallest possible radius $r_0$ or the largest $r_1$ based on the samples)—we only need to bound a single failure event. In this case, the error region $L_{\mathcal{D}}(h_S) > \epsilon$ is covered by a single interval of probability mass $\epsilon$ rather than two. Consequently, the union bound is no longer necessary, yielding the tighter bound $m_{\mathcal{H}}(\epsilon,\delta)\le\left\lceil\dfrac{\log(1/\delta)}{\epsilon}\right\rceil$.

### <span style="color: #6ED3C5">VC Dimension</span>

<span style="color: #6FA8FF">**Restriction of $\mathcal{H}$ to $C$:**</span> Let $\mathcal{H}$ be a hypothesis class over a domain set $\mathcal{X}$ and let $C=\{c_1,c_2,\ldots,c_m\}\subseteq\mathcal{X}$ be a finite subset. The restriction of $\mathcal{H}$ to $C$ is defined as $\mathcal{H}_C=\{(h(x_1),h(x_2),\ldots,h(x_m)):h\in\mathcal{H}\}$, where we identify each function in $\mathcal{H}_C$ with a binary vector of length $m$.

<span style="color: #6FA8FF">**Shattering:**</span> A hypothesis class $\mathcal{H}$ shatters a finite subset $C\subseteq\mathcal{X}$ if $\mathcal{H}_C$ contains all binary vectors of length $|C|$, i.e., $|\mathcal{H}_C|=2^{|C|}$.

<span style="color: #6FA8FF">**Corollary:**</span> Let $\mathcal{H}$ be a hypothesis class of functions from a domain set $\mathcal{X}$ to $\{0,1\}$. Let $m$ be a training set size. Assume that there exists a subset $C\subseteq\mathcal{X}$ of size $2m$ that is shattered by $\mathcal{H}$. Then for any learning algorithm $A$, there exists a distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$ and a predictor function $h\in\mathcal{H}$ with $L_{\mathcal{D}}(h)=0$, but with probability at least $\dfrac{1}{7}$ over the choice of $S\sim\mathcal{D}^m$, $L_{\mathcal{D}}(A(S))\ge \dfrac{1}{8}$.

<span style="color: #6FA8FF">**VC Dimension:**</span> The VC dimension of a hypothesis class $\mathcal{H}$, denoted by VCdim$(\mathcal{H})$, is defined as the maximal size of a subset of $\mathcal{X}$ that is shattered by $\mathcal{H}$. If $\mathcal{H}$ can shatter sets of arbitrary large finite size, then VCdim$(\mathcal{H})=\infty$.

<span style="color: #6FA8FF">**Theorem:**</span> Let $\mathcal{H}$ be a class of infinite VC-dimension. Then, $\mathcal{H}$ is not PAC learnable.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>

Since $\mathcal{H}$ has an infinite VC-dimension, for any training set size $m$, there exists a shattered set of size $2m$. Then for any learning algorithm $A$, there exists a distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$ and a predictor function $h\in\mathcal{H}$ with $L_{\mathcal{D}}(h)=0$, but with probability at least $\dfrac{1}{7}$ over the choice of $S\sim\mathcal{D}^m$ we have $L_{\mathcal{D}}(A(S))\ge \dfrac{1}{8}$.

Assume by contradiction that $\mathcal{H}$ is PAC learnable. Choose some $\epsilon<\dfrac{1}{8}$ and $\delta<\dfrac{1}{7}$. By the PAC learnability assumption, there must be some learning algorithm $A$ and an integer $m=m(\epsilon,\delta)$, such that for every data-generating distribution $\mathcal{D}$ over $\mathcal{X}\times\{0,1\}$, if for some function $f:\mathcal{X}\to\{0,1\},L_{\mathcal{D}}(f)=0$, then with probability at least $1-\delta$ over the choice of an i.i.d. sample $S$ of size $m$, the hypothesis $h=A(S)$ satisfies $L_{\mathcal{D}}(h)\le\epsilon$, which contradicts the above result.

</details>

### <span style="color: #6ED3C5">The Fundamental Theorem of PAC Learning</span>

<span style="color: #6FA8FF">**The Fundamental Theorem of Statistical Learning:**</span> Let $\mathcal{H}$ be a hypothesis class of functions from a domain $\mathcal{X}$ to $\{0,1\}$ and let the loss function be the $0-1$ loss. Then, the following are equivalent:

1. $\mathcal{H}$ has the uniform convergence property.
2. Any ERM rule is a successful agnostic PAC learner for $\mathcal{H}$.
3. $\mathcal{H}$ is agnostic PAC learnable.
4. $\mathcal{H}$ is PAC learnable.
5. Any ERM rule is a successful PAC learner for $\mathcal{H}$.
6. $\mathcal{H}$ has a finite VC-dimension.

<span style="color: #6FA8FF">**Quantitative Version:**</span> Let $\mathcal{H}$ be a hypothesis class of functions from a domain $\mathcal{X}$ to $\{0,1\}$ and let the loss function be the $0−1$ loss. Assume that $\text{VCdim}(\mathcal{H})=d<\infty$. Then, there are absolute constants $C_1,C_2$ such that:

1. $\mathcal{H}$ is agnostically PAC learnable with sample complexity
   $$C_1\dfrac{d+\log(1/\delta)}{\epsilon^2}\le m_{\mathcal{H}}(\epsilon,\delta)\le C_2\dfrac{d+\log(1/\delta)}{\epsilon^2}.$$
2. $\mathcal{H}$ is PAC learnable with sample complexity
   $$C_1\dfrac{d+\log(1/\delta)}{\epsilon}\le m_{\mathcal{H}}(\epsilon,\delta)\le C_2\dfrac{d\log(1/\epsilon)+\log(1/\delta)}{\epsilon}.$$
