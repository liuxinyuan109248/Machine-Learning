# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 08 Decision Trees</span>

### <span style="color: #6ED3C5">Decision Trees</span>

- Gini Impurity: $\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$ where $p_i$ is the probability of class $i$ in a node
- pick the split that minimizes weighted Gini of child nodes

### <span style="color: #6ED3C5">Random Forests</span>

- for each tree:
  - randomly select subset of features
  - sample with replacement from training data and train tree $T_b$ on this sample $(x_b, y_b)$
  - each element will be left out with probability $\left(1 - \dfrac{1}{N}\right)^N \approx e^{-1} \approx 0.37$
- $\hat{f}_{rf}(x) = \dfrac{1}{B} \sum_{b=1}^{B} T_b(x)$, where $B$ is the number of trees

Random forests reduce variance by averaging multiple deep trees trained on different parts of the same training set

### <span style="color: #6ED3C5">AdaBoost</span>

- initialize weights $D_1(i) = \dfrac{1}{N}$ for $i = 1, \ldots, N$
- for $t = 1, \ldots, T$:
  - train weak classifier $h_t$ using distribution $D_t$
  - compute error $\epsilon_t = P_{i \sim D_t}[h_t(x_i) \neq y_i]$
  - compute classifier weight $\alpha_t = \dfrac{1}{2} \ln\left(\dfrac{1 - \epsilon_t}{\epsilon_t}\right)$
  - update weights: $D_{t+1}(i) = \dfrac{D_t(i) \exp(-\alpha_t y_i h_t(x_i))}{Z_t}$, where $Z_t$ is a normalization factor
- final classifier: $H(x) = \text{sign}\left(\sum_{t=1}^{T} \alpha_t h_t(x)\right)$

<span style="color: #6FA8FF">**Theorem:**</span> If we write $\epsilon_t = \dfrac{1}{2} - \gamma_t$, the training error of the final classifier $H(x)$ is bounded by:
$$\text{Training Error} \leq \exp\left(-2 \sum_{t=1}^{T} \gamma_t^2\right)$$

<span style="color: #6FA8FF">**Proof:**</span> Let $f(x) = \sum_{t=1}^{T} \alpha_t h_t(x)$, then $H(x) = \text{sign}(f(x))$. The training error can be bounded as follows:
$$\text{Training Error} = \dfrac{1}{N} \sum_{i=1}^{N} \mathbb{1}(H(x_i) \neq y_i) \leq \dfrac{1}{N} \sum_{i=1}^{N} \exp(-y_i f(x_i))$$
Unrolling the weight updates, we have:
$$D_{T+1}(i) = \dfrac{1}{N} \prod_{t=1}^{T} \dfrac{\exp(-\alpha_t y_i h_t(x_i))}{Z_t} = \dfrac{\exp(-y_i f(x_i))}{N \prod_{t=1}^{T} Z_t}$$
Next, we compute $Z_t$:
$$Z_t = \sum_{i=1}^{N} D_t(i) \exp(-\alpha_t y_i h_t(x_i)) = (1 - \epsilon_t) \exp(-\alpha_t) + \epsilon_t \exp(\alpha_t) = 2 \sqrt{\epsilon_t (1 - \epsilon_t)} = \sqrt{1 - 4 \gamma_t^2} \leq \exp(-2 \gamma_t^2)$$
Putting it all together:
$$\text{Training Error} \leq \prod_{t=1}^{T} Z_t \leq \exp\left(-2 \sum_{t=1}^{T} \gamma_t^2\right)$$

**Theorem:** Let $S$ be a sample of size $m$ drawn i.i.d. according to distribution $D$. Let $H$ be a finite hypothesis class. For any $\delta > 0$, with probability at least $1 - \delta$, every weighted average hypothesis $f$ satisfies for all $\theta > 0$:
$$P_D[y f(x) \leq 0] \leq P_S[y f(x) \leq \theta] + O\left(\sqrt{\dfrac{\log |H|}{m \theta^2}} + \sqrt{\dfrac{\log(1/\delta)}{m}}\right)$$

- modification of AdaBoost: $f(x) = \dfrac{\sum_{t=1}^{T} \alpha_t h_t(x)}{\sum_{t=1}^{T} \alpha_t}$

<span style="color: #6FA8FF">**Theorem:**</span> AdaBoost generates a sequence of hypotheses $h_1, \ldots, h_T$ with training errors $\epsilon_1, \ldots, \epsilon_T$. Then for any $\theta > 0$, we have:
$$P_S[y f(x) \leq \theta] \leq 2^T \prod_{t=1}^{T} \sqrt{\epsilon_t^{1 - \theta} (1 - \epsilon_t)^{1 + \theta}}$$

<span style="color: #6FA8FF">**Proof:**</span> Note that:
$$y f(x) \leq \theta \iff \sum_{t=1}^{T} \alpha_t y h_t(x) \leq \theta \sum_{t=1}^{T} \alpha_t \implies \exp\left(-\sum_{t=1}^{T} \alpha_t y h_t(x) + \theta \sum_{t=1}^{T} \alpha_t\right) \geq 1$$
Using Markov's inequality:
$$
\begin{aligned}
P_S[y f(x) \leq \theta] \leq& E_S\left[\exp\left(-\sum_{t=1}^{T} \alpha_t y h_t(x) + \theta \sum_{t=1}^{T} \alpha_t\right)\right] = \frac{\exp\left(\theta \sum_{t=1}^{T} \alpha_t\right)}{m} \sum_{i=1}^{m} \exp\left(-\sum_{t=1}^{T} \alpha_t y_i h_t(x_i)\right) \\
=& \exp\left(\theta \sum_{t=1}^{T} \alpha_t\right) \prod_{t=1}^{T} Z_t = \sqrt{\prod_{t=1}^{T} \left(\frac{1 - \epsilon_t}{\epsilon_t}\right)^{\theta}} \cdot 2^T \prod_{t=1}^{T} \sqrt{\epsilon_t (1 - \epsilon_t)} = 2^T \prod_{t=1}^{T} \sqrt{\epsilon_t^{1 - \theta} (1 - \epsilon_t)^{1 + \theta}} \\
\end{aligned}
$$
If $\epsilon_t \leq \dfrac{1}{2} - \gamma$ for all $t$, then:
$$P_S[y f(x) \leq \theta] \leq 2^T \left(\sqrt{\left(\dfrac{1}{2} - \gamma\right)^{1 - \theta} \left(\dfrac{1}{2} + \gamma\right)^{1 + \theta}}\right)^T = \left(\sqrt{(1 - 2\gamma)^{1 - \theta} (1 + 2\gamma)^{1 + \theta}}\right)^T$$
If $\theta < \gamma$, then $(1 - 2\gamma)^{1 - \theta} (1 + 2\gamma)^{1 + \theta} < 1$ and the training margin error decreases exponentially with $T$

### <span style="color: #6ED3C5">Coordinate Descent</span>

- let $\{g_1, \ldots, g_N\}$ be weak classifiers
- $L(\lambda) = \sum_{i=1}^{N} \exp\left(-y_i \sum_{j=1}^{N} \lambda_j g_j(x_i)\right)$
- initialize $\lambda_j = 0$ for $j = 1, \ldots, N$
- for $t = 1, \ldots, T$: pick coordinate $j$ and update $\lambda_j \leftarrow \alpha_j$

### <span style="color: #6ED3C5">Gradient Boosting</span>

- loss function: $L(y, f(x))$
- for $t = 1, \ldots, T$:
  - compute pseudo-residuals: $r_i^{(t)} = -\left.\dfrac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right|_{f = f^{(t-1)}}$
  - fit weak learner $h_t$ to pseudo-residuals $\{(x_i, r_i^{(t)})\}_{i=1}^{N}$
  - update model: $f^{(t)}(x) = f^{(t-1)}(x) + \eta h_t(x)$
