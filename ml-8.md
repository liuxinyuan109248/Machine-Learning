# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 08 Decision Trees</span>

### <span style="color: #6ED3C5">Decision Trees</span>

<span style="color: #6FA8FF">**Decision Trees**</span> are non-parametric supervised learning models that partition the feature space into a set of rectangular regions. The goal is to create a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.

For a classification task with $C$ classes, the <span style="color: #6FA8FF">**Gini Impurity**</span> measures the frequency with which a randomly chosen element from a node would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. At a specific node, it is defined as: $\text{Gini} = 1 - \sum_{i=1}^{C} p_i^2$, where $p_i$ is the probability (proportion) of class $i$ in that node.

* A Gini Impurity of **0** indicates a "pure" node (all elements belong to one class).
* A higher Gini value indicates a more diverse (impure) distribution of classes.

Decision trees are built using a **greedy, top-down approach**. At each node, the algorithm performs a split based on a single feature $k$ and a threshold $t$. For every internal node, the algorithm searches through all features $k \in \{1, \dots, p\}$ and all possible threshold values $t$ within each feature. It partitions the data $D$ into two disjoint subsets: $D_{left}(k, t) = \{x \mid x_k \leq t\}$ and $D_{right}(k, t) = \{x \mid x_k > t\}$.

To choose the "best" feature-threshold pair $(k, t)$, the algorithm minimizes a cost function $J(k, t)$, which represents the <span style="color: #6FA8FF">**Weighted Average Gini Impurity**</span> of the resulting child nodes: $J(k, t) = \dfrac{n_{left}}{n} \text{Gini}(D_{left}) + \dfrac{n_{right}}{n} \text{Gini}(D_{right})$, where $n_{left}$ and $n_{right}$ are the number of samples in the left and right children, respectively, and $n$ is the total number of samples in the parent node. The weights $\dfrac{n_{left}}{n}$ and $\dfrac{n_{right}}{n}$ ensure that larger child nodes have a greater influence on the decision than smaller ones.

Deep trees can capture noise in the training data, leading to high variance. This is typically mitigated by **pruning** or by using ensemble methods like <span style="color: #6FA8FF">**Random Forests**</span>.

### <span style="color: #6ED3C5">Random Forest Algorithm</span>

<span style="color: #6FA8FF">**Random Forest**</span> is an ensemble learning method that constructs a multitude of decision trees at training time and outputs the average prediction of the individual trees for regression tasks. It is designed to improve upon the high variance of individual decision trees through **Bagging** and **Feature Randomness**.

For an ensemble of $B$ trees, the training process for each tree $T_b$ follows these steps:

<span style="color: #6FA8FF">**A. Bootstrap Aggregation (Bagging):**</span> For each tree $b = 1, \dots, B$:

- Sample $N$ examples from the training data **with replacement**.
- This creates a bootstrap sample $(x_b, y_b)$ of the same size as the original dataset.

**The 0.37 Rule (Out-of-Bag Observations):**
For a dataset of size $N$, the probability that a specific data point is **not** selected in a single draw is $(1 - 1/N)$. Since we draw $N$ times, the probability that a point is left out of the bootstrap sample is: $\left(1 - \dfrac{1}{N}\right)^N \approx e^{-1} \approx 0.368$. Consequently, approximately **37%** of the data is "Out-of-Bag" (OOB) for any given tree, providing a built-in validation set for estimating generalization error.

<span style="color: #6FA8FF">**B. Feature Randomization:**</span> While growing the tree $T_b$, at each node split:

- Instead of considering all available features, **randomly select a subset of features**.
- Pick the best split among only that subset.
- This technique is the primary mechanism for **de-correlation**, ensuring that trees do not all rely on the same dominant features.

The final Random Forest estimator $\hat{f}_{\text{rf}}(x)$ is the average of all individual tree predictions: $\hat{f}_{\text{rf}}(x) = \dfrac{1}{B} \sum_{b=1}^{B} T_b(x)$.

Based on the statistical properties of the ensemble, the variance of the Random Forest is given by: $\text{Var}(\hat{f}_{\text{rf}}(x)) = \rho\sigma^2 + \dfrac{1-\rho}{B}\sigma^2$, where $\rho$ is the average correlation between individual trees and $\sigma^2$ is the variance of a single tree's prediction.

<span style="color: #6FA8FF">**Why Random Forests Outperform Simple Bagging**:</span>

In standard Bagging (using all features), $\rho$ remains high because different trees often choose the same strong predictors for the top-level splits. By using a **random subset of features**, Random Forests force the trees to be diverse, which **lowers $\rho$**.

As shown in the formula, as $B \to \infty$, the variance approaches the limit $\rho\sigma^2$. By aggressively reducing $\rho$ through feature randomness, Random Forests push this variance floor lower than simple bagging ever could.

| Parameter | Role | Effect on Model |
| :--- | :--- | :--- |
| **$B$ (n_estimators)** | Number of trees | Increasing $B$ reduces variance without increasing bias. |
| **$\rho$ (Tree Correlation)** | Correlation between trees | Lowering $\rho$ via feature randomness significantly reduces ensemble variance. |

### <span style="color: #6ED3C5">AdaBoost (Adaptive Boosting)</span>

<span style="color: #6FA8FF">**AdaBoost**</span> is an ensemble learning method that constructs a strong classifier by combining multiple **weak learners** (typically decision stumps). It is **adaptive** because it sequentially adjusts the importance of training samples:

- Misclassified samples are assigned higher weights in the next iteration.
- Correctly classified samples receive lower weights.

This forces each new weak learner to focus on the "hard" cases that previous learners missed.

<span style="color: #6FA8FF">**Algorithm Procedure:**</span>

1. **Initialize:** Set initial weights $D_1(i) = \dfrac{1}{N}$ for $i = 1, \ldots, N$.
2. **Iterate:** For $t = 1, \ldots, T$:
    * **Train:** Train a weak classifier $h_t$ using the current distribution $D_t$.
    * **Evaluate Error:** Compute the weighted error $\epsilon_t = P_{i \sim D_t}[h_t(x_i) \neq y_i]$.
    * **Assign Voting Power:** Compute the classifier weight (importance) $\alpha_t = \dfrac{1}{2} \ln\left(\dfrac{1 - \epsilon_t}{\epsilon_t}\right)$.
    * **Update Distribution:** Update weights for the next round: $D_{t+1}(i) = \dfrac{D_t(i) \exp(-\alpha_t y_i h_t(x_i))}{Z_t}$, where $Z_t$ is a normalization factor to ensure $\sum D_{t+1} = 1$.

3. **Final Decision:** The final strong classifier is a weighted majority vote: $H(x) = \text{sign}(f(x))$, where $f(x)=\sum_{t=1}^{T} \alpha_t h_t(x)$.

<span style="color: #6FA8FF">**Theorem:**</span> If we write $\epsilon_t = \dfrac{1}{2} - \gamma_t$, the training error of the final classifier $H(x)$ is bounded by $\text{Training Error} \leq \exp\left(-2 \sum_{t=1}^{T} \gamma_t^2\right)$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>
  
The training error can be bounded as follows:
$\text{Training Error} = \dfrac{1}{N} \sum_{i=1}^{N} \mathbb{1}(H(x_i) \neq y_i) \leq \dfrac{1}{N} \sum_{i=1}^{N} \exp(-y_i f(x_i))$.
Unrolling the weight updates, we have $D_{T+1}(i) = \dfrac{1}{N} \prod_{t=1}^{T} \dfrac{\exp(-\alpha_t y_i h_t(x_i))}{Z_t} = \dfrac{\exp(-y_i f(x_i))}{N \prod_{t=1}^{T} Z_t}$. Next, we compute $Z_t$:
$$Z_t = \sum_{i=1}^{N} D_t(i) \exp(-\alpha_t y_i h_t(x_i)) = (1 - \epsilon_t) \exp(-\alpha_t) + \epsilon_t \exp(\alpha_t) = 2 \sqrt{\epsilon_t (1 - \epsilon_t)} = \sqrt{1 - 4 \gamma_t^2} \leq \exp(-2 \gamma_t^2)$$
Putting it all together: $\text{Training Error} \leq \prod_{t=1}^{T} Z_t \leq \exp\left(-2 \sum_{t=1}^{T} \gamma_t^2\right)$.

</details>

<span style="color: #6FA8FF">**Theorem:**</span> Let $S$ be a sample of size $m$ drawn i.i.d. according to distribution $D$. Let $H$ be a finite hypothesis class. For any $\delta > 0$, with probability at least $1 - \delta$, every weighted average hypothesis $f$ satisfies for all $\theta > 0$:
$$P_D[y f(x) \leq 0] \leq P_S[y f(x) \leq \theta] + O\left(\sqrt{\dfrac{\log |H|}{m \theta^2}} + \sqrt{\dfrac{\log(1/\delta)}{m}}\right)$$

<span style="color: #6FA8FF">**Theorem:**</span> AdaBoost generates a sequence of hypotheses $h_1, \ldots, h_T$ with training errors $\epsilon_1, \ldots, \epsilon_T$. If $f(x) = \dfrac{\sum_{t=1}^{T} \alpha_t h_t(x)}{\sum_{t=1}^{T} \alpha_t}$, then for any $\theta > 0$, we have $P_S[y f(x) \leq \theta] \leq 2^T \prod_{t=1}^{T} \sqrt{\epsilon_t^{1 - \theta} (1 - \epsilon_t)^{1 + \theta}}$.

<details>
  <summary><b><font color="#6FA8FF">Proof:</font></b> (Click to expand)</summary>
  
Note that $y f(x) \leq \theta \iff \sum_{t=1}^{T} \alpha_t y h_t(x) \leq \theta \sum_{t=1}^{T} \alpha_t \implies \exp\left(-\sum_{t=1}^{T} \alpha_t y h_t(x) + \theta \sum_{t=1}^{T} \alpha_t\right) \geq 1$. By Markov's inequality:
$$
\begin{aligned}
P_S[y f(x) \leq \theta] \leq& E_S\left[\exp\left(-\sum_{t=1}^{T} \alpha_t y h_t(x) + \theta \sum_{t=1}^{T} \alpha_t\right)\right] = \frac{\exp\left(\theta \sum_{t=1}^{T} \alpha_t\right)}{m} \sum_{i=1}^{m} \exp\left(-\sum_{t=1}^{T} \alpha_t y_i h_t(x_i)\right) \\
=& \exp\left(\theta \sum_{t=1}^{T} \alpha_t\right) \prod_{t=1}^{T} Z_t = \sqrt{\prod_{t=1}^{T} \left(\frac{1 - \epsilon_t}{\epsilon_t}\right)^{\theta}} \cdot 2^T \prod_{t=1}^{T} \sqrt{\epsilon_t (1 - \epsilon_t)} = 2^T \prod_{t=1}^{T} \sqrt{\epsilon_t^{1 - \theta} (1 - \epsilon_t)^{1 + \theta}} \\
\end{aligned}
$$

</details>

<span style="color: #6FA8FF">**Corollary:**</span> If $\epsilon_t \leq \dfrac{1}{2} - \gamma$ for all $t$ and $\theta < \gamma$, then the training margin error decreases exponentially with $T$.

<span style="color: #6FA8FF">**Proof:**</span> $P_S[y f(x) \leq \theta] \leq \left(\sqrt{(1 - 2\gamma)^{1 - \theta} (1 + 2\gamma)^{1 + \theta}}\right)^T$ and $(1 - 2\gamma)^{1 - \theta} (1 + 2\gamma)^{1 + \theta} < 1$.

### <span style="color: #6ED3C5">Theoretical Connection: MWU and Mirror Descent</span>

**AdaBoost** can be viewed as a specific instance of the <span style="color: #6FA8FF">**Multiplicative Weights Update (MWU)**</span> algorithm, which in turn is equivalent to <span style="color: #6FA8FF">**Mirror Descent**</span> using KL divergence as the Bregman divergence.

* **MWU Equivalence:** The MWU update rule is defined as $w_i^{(t+1)} = \dfrac{w_i^{(t)} \exp(-\eta_t l_t^i)}{Z_t}$. By setting the learning rate $\eta_t = \alpha_t$ and the loss $l_t^i = y_i h_t(x_i)$, the AdaBoost weight update becomes identical to MWU.
* **Mirror Descent Perspective:** Since MWU is a form of Mirror Descent where the potential function generates the KL divergence: $V_x(y) = \sum_{i=1}^n y_i \log \dfrac{y_i}{x_i} = \mathrm{KL}(y \| x)$, AdaBoost effectively performs optimization in the simplex geometry, leveraging the properties of Mirror Descent to achieve fast convergence in high-dimensional feature spaces.

### <span style="color: #6ED3C5">Coordinate Descent</span>

In the context of boosting, <span style="color: #6FA8FF">**coordinate descent**</span> treats each weak classifier $g_j$ as a "coordinate" in a high-dimensional function space. The goal is to minimize the total exponential loss by optimizing the weights $\lambda_j$ assigned to each classifier.

* **Objective:** Minimize the exponential loss: $L(\lambda) = \sum_{i=1}^{N} \exp\left(-y_i \sum_{j=1}^{N} \lambda_j g_j(x_i)\right)$.
* **Procedure:**
  1. **Initialize:** Set all weights $\lambda_j = 0$ for $j = 1, \ldots, N$.
  2. **Iterate:** For $t = 1, \ldots, T$:
     * Select a specific coordinate $j$ (representing a weak learner $g_j$).
     * Update the weight $\lambda_j \leftarrow \alpha_j$ to reduce the loss along that direction.

### <span style="color: #6ED3C5">Gradient Boosting</span>

<span style="color: #6FA8FF">**Gradient boosting**</span> generalizes the concept of boosting to any differentiable loss function. Instead of just adjusting weights for a fixed set of classifiers, it fits new learners to the <span style="color: #6FA8FF">**negative gradient (pseudo-residuals)**</span> of the loss, moving the model $f(x)$ closer to the true labels.

**Algorithm Procedure:**

1. **Define Loss:** Choose a differentiable loss function $L(y, f(x))$.
2. **Iterate:** For $t = 1, \ldots, T$:
    * **Compute pseudo-residuals:** Calculate the negative gradient of the loss with respect to the current model's predictions: $r_i^{(t)} = -\left.\dfrac{\partial L(y_i, f(x_i))}{\partial f(x_i)}\right|_{f = f^{(t-1)}}$
    * **Fit weak learner:** Train a new weak learner $h_t$ to predict these residuals using the dataset $\{(x_i, r_i^{(t)})\}_{i=1}^{N}$.
    * **Update model:** Move the ensemble model in the direction of the new learner $f^{(t)}(x) = f^{(t-1)}(x) + \eta h_t(x)$.

### <span style="color: #6ED3C5">Connecting AdaBoost and Gradient Boosting</span>

**AdaBoost** is precisely **Gradient Boosting** using the <span style="color: #6FA8FF">**exponential loss**</span> function $L(y, f(x)) = \exp(-y f(x))$.

For a single observation, the negative gradient is $r_{i} = -\dfrac{\partial L(y_i, f(x_i))}{\partial f(x_i)} = y_i \exp(-y_i f(x_i))$. In AdaBoost, the sample weights are proportional to the exponential loss of the current ensemble: $D_t(i) \propto \exp(-y_i f^{(t-1)}(x_i))$. The weak learner $h_t$ is trained to maximize the correlation with these weights:
$h_t = \arg\max_{h} \sum_{i=1}^n y_i \exp(-y_i f^{(t-1)}(x_i)) h(x_i) = \arg\max_{h} \sum_{i=1}^n r_{it} h(x_i)$.

* **In AdaBoost:** $h_t$ is chosen to maximize **correlation** with the negative gradient: $\arg\max_{h} \sum r_{it} h(x_i)$.
* **In Gradient Boosting:** $h_t$ is chosen to minimize the **squared error** with the negative gradient: $\arg\min_{h} \sum (r_{it} - h(x_i))^2$.

For classification where $h(x_i) \in \{-1, 1\}$, these two objectives are mathematically equivalent, proving that AdaBoost is a functional gradient descent process.
