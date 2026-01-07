# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 07 Support Vector Machines</span>

### <span style="color: #6ED3C5">Hard-Margin SVM (Linearly Separable Case)</span>

The goal is to find the hyperplane that maximizes the margin between two perfectly separable classes.

<span style="color: #6FA8FF">**Primal Formulation:**</span> Minimize the norm of the weight vector subject to correct classification:
$$\min_{w, b} \frac{1}{2}\|w\|_2^2 \quad \text{s.t.} \quad y_i(w^Tx_i - b) \ge 1, \quad \forall i=1, \ldots, N$$

<span style="color: #6FA8FF">**Dual Formulation and Gradient Derivation:**</span> We construct the **Lagrangian** by introducing multipliers $a_i \ge 0$:
$$L(w, b, a) = \frac{1}{2}\|w\|_2^2 - \sum_{i=1}^N a_i [y_i(w^Tx_i - b) - 1]$$

To find the dual, we solve for the stationary points by calculating the **gradients**:

1. **Gradient w.r.t. $w$:** $\nabla_w L = w - \sum_{i=1}^N a_i y_i x_i = 0 \implies w = \sum_{i=1}^N a_i y_i x_i$
2. **Gradient w.r.t. $b$:** $\dfrac{\partial L}{\partial b} = \dfrac{\partial}{\partial b} \left( \sum_{i=1}^N a_i y_i b \right) = \sum_{i=1}^N a_i y_i = 0$

Substituting these results back into $L$ yields the <span style="color: #6FA8FF">**Dual Objective**</span>:
$$\max_{a} \sum_{i=1}^N a_i - \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N a_i a_j y_i y_j x_i^T x_j \quad \text{s.t.} \quad a_i \ge 0, \quad \sum_{i=1}^N a_i y_i = 0$$

<span style="color: #6FA8FF">**Decision Function:**</span> Using the weight relation $w = \sum a_i y_i x_i$, the standardized decision rule is: $f(x) = \text{sign}\left( \sum_{i=1}^N a_i y_i (x_i^T x) - b \right)$.

### <span style="color: #6ED3C5">Soft-Margin SVM (Relaxed Case)</span>

To handle non-separable data or noise, we introduce slack variables $\xi_i \ge 0$ and a regularization parameter $\lambda$.

<span style="color: #6FA8FF">**Primal Formulation**:</span>
$$\min_{w, b, \xi} \frac{1}{2}\|w\|_2^2 + \lambda \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(w^Tx_i - b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$

<span style="color: #6FA8FF">**Dual Formulation and Gradient Derivation:**</span> We introduce multipliers $a_i \ge 0$ (for the margin) and $\kappa_i \ge 0$ (for the slack constraints):
$$L(w, b, \xi, a, \kappa) = \frac{1}{2}\|w\|_2^2 + \lambda \sum_{i=1}^N \xi_i - \sum_{i=1}^N a_i [y_i(w^T x_i - b) - 1 + \xi_i] - \sum_{i=1}^N \kappa_i \xi_i$$

Setting the **gradients** to zero:

1. **Gradient w.r.t. $w$:** $\nabla_w L = w - \sum a_i y_i x_i = 0 \implies w = \sum_{i=1}^N a_i y_i x_i$
2. **Gradient w.r.t. $b$:** $\dfrac{\partial L}{\partial b} = \sum a_i y_i = 0$
3. **Gradient w.r.t. $\xi_i$:** $\dfrac{\partial L}{\partial \xi_i} = \lambda - a_i - \kappa_i = 0 \implies a_i + \kappa_i = \lambda$

Since $\kappa_i \ge 0$, the condition $a_i + \kappa_i = \lambda$ implies $a_i \le \lambda$, leading to the <span style="color: #6FA8FF">**Box Constraint**</span> $0 \le a_i \le \lambda$.

<span style="color: #6FA8FF">**Dual Objective:**</span>
Substituting these relations back into $L$ yields:
$$\max_{a} \sum_{i=1}^N a_i - \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N a_i a_j y_i y_j x_i^T x_j \quad \text{s.t.} \quad 0 \le a_i \le \lambda, \quad \sum_{i=1}^N a_i y_i = 0$$

<span style="color: #6FA8FF">**Decision Function:**</span> Using the weight relation $w = \sum a_i y_i x_i$, the standardized decision rule is: $f(x) = \text{sign}\left( \sum_{i=1}^N a_i y_i (x_i^T x) - b \right)$.

### <span style="color: #6ED3C5">Kernel SVM (Non-Linear Case)</span>

We map data to a high-dimensional feature space $\phi(x)$ to find a linear separation where one did not exist in the original space.

<span style="color: #6FA8FF">**The Kernel Trick:**</span> Instead of explicitly computing $\phi(x)$, we define a <span style="color: #6FA8FF">**Kernel Function**</span> $K(x_i, x_j) = \phi(x_i)^T \phi(x_j)$.

* <span style="color: #6FA8FF">**Mercer's Condition:**</span> For $K(x, z)$ to be a valid kernel, it must be symmetric ($K(x, z) = K(z, x)$) and the corresponding <span style="color: #6FA8FF">**Kernel Matrix**</span> must be positive semi-definite ($G \succeq 0$). This ensures there exists a mapping $\phi(\cdot)$ into a Hilbert space where the kernel represents an inner product.
* <span style="color: #6FA8FF">**Gaussian (RBF) Kernel:**</span> $K(x_i, x_j) = \exp\left(-\dfrac{\|x_i - x_j\|_2^2}{2\sigma^2}\right)$

<span style="color: #6FA8FF">**Dual Formulation with Kernels:**</span> By replacing $x_i^T x_j$ with $K(x_i, x_j)$ in the soft-margin dual, we get:
$$\max_{a} \sum_{i=1}^N a_i - \frac{1}{2}\sum_{i=1}^N \sum_{j=1}^N a_i a_j y_i y_j K(x_i, x_j) \quad \text{s.t.} \quad 0 \le a_i \le \lambda, \quad \sum_{i=1}^N a_i y_i = 0$$

<span style="color: #6FA8FF">**Decision Function with Kernels:**</span> The decision function is given by:
$f(x) = \text{sign}\left( \sum_{i=1}^N a_i y_i K(x_i, x) - b \right)$.

In high-dimensional space, the decision is $f(x) = \text{sign}(w^T \phi(x) - b)$. Substituting the optimal weight $w = \sum a_i y_i \phi(x_i)$, we get $\text{sign}\left( \left(\sum a_i y_i \phi(x_i)\right)^T \phi(x) - b \right)$. Replacing the dot product with the kernel $K(x_i, x)$, we obtain the final form. The kernelized decision function acts as a **weighted similarity comparison**. Each support vector $x_i$ influences the prediction based on its importance $a_i$, its class $y_i$, and its similarity to the test point $x$ as measured by $K(x_i, x)$.
