# <span style="color: #4FC3F7">Machine Learning</span>

## <span style="color: #F2A07B">Lec 07 Support Vector Machines</span>

<span style="color: #6FA8FF">**Hard-Margin SVM:**</span> $\min_{w,b}\dfrac{1}{2}\|w\|_2^2$ s.t. $y_i(w^Tx_i-b)\ge 1,\forall i=1,2,\ldots,N$

<span style="color: #6FA8FF">**Soft-Margin SVM:**</span> $\min_{w,b,\xi}\dfrac{1}{2}\|w\|_2^2+\lambda\sum_{i=1}^N\xi_i$ s.t. $y_i(w^Tx_i-b)\ge 1-\xi_i,\xi_i\ge 0,\forall i=1,2,\ldots,N$

<span style="color: #6FA8FF">**SVM (linearly separable case):**</span>

- primal: $\min_{w,b}\dfrac{1}{2}\|w\|_2^2$ s.t. $y_i(w^Tx_i-b)\ge 1,\forall i=1,2,\ldots,N$
- dual: $L(w,a)=\dfrac{1}{2}\|w\|_2^2-\sum_{i=1}^Na_i[y_i(w^Tx_i-b)-1], a_i\ge 0,\forall i=1,2,\ldots,N$
  $$\frac{\partial L}{\partial w}=0\implies w=\sum_{i=1}^Na_iy_ix_i,\quad \frac{\partial L}{\partial b}=0\implies \sum_{i=1}^Na_iy_i=0$$
  $$\implies L(w,a)=\sum_{i=1}^Na_i+b\sum_{i=1}^Na_iy_i-\dfrac{1}{2}\sum_{i=1}^N\sum_{j=1}^Na_ia_jy_iy_jx_i^Tx_j$$
  s.t. $a_i\ge 0,\forall i=1,2,\ldots,N,\sum_{i=1}^Na_iy_i=0$

<span style="color: #6FA8FF">**SVM (relaxed case):**</span>

- primal: $\min_{w,b,\xi}\dfrac{1}{2}\|w\|_2^2+\lambda\sum_{i=1}^N\xi_i$ s.t. $y_i(w^Tx_i-b)\ge 1-\xi_i,\xi_i\ge 0,\forall i=1,2,\ldots,N$
- dual: $L(w,b,\xi,a,\kappa)=\dfrac{1}{2}\|w\|_2^2+\lambda\sum_{i=1}^N\xi_i-\sum_{i=1}^Na_i[y_i(w^Tx_i-b)-1+\xi_i]-\sum_{i=1}^N\kappa_i\xi_i,a_i\ge 0,\kappa_i\ge 0$
  $$\frac{\partial L}{\partial w}=0\implies w=\sum_{i=1}^Na_iy_ix_i,\quad \frac{\partial L}{\partial \xi_i}=0\implies a_i+\kappa_i=\lambda,\quad \frac{\partial L}{\partial b}=0\implies \sum_{i=1}^Na_iy_i=0$$
  $$\implies L(a)=\sum_{i=1}^Na_i+b\sum_{i=1}^Na_iy_i-\dfrac{1}{2}\sum_{i=1}^N\sum_{j=1}^Na_ia_jy_iy_jx_i^Tx_j$$
  s.t. $0\le a_i\le \lambda,\forall i=1,2,\ldots,N,\sum_{i=1}^Na_iy_i=0$

<span style="color: #6FA8FF">**SVM with Kernel Trick (relaxed case):**</span>

- primal: $\min_{w,b,\xi}\dfrac{1}{2}\|w\|_2^2+\lambda\sum_{i=1}^N\xi_i$ s.t. $y_i(w^T\phi(x_i)-b)\ge 1-\xi_i,\xi_i\ge 0,\forall i=1,2,\ldots,N$
- dual: maximize $L(a)=\sum_{i=1}^Na_i+b\sum_{i=1}^Na_iy_i-\dfrac{1}{2}\sum_{i=1}^N\sum_{j=1}^Na_ia_jy_iy_j\phi(x_i)^T\phi(x_j)$
  
  s.t. $0\le a_i\le \lambda,\forall i=1,2,\ldots,N,\sum_{i=1}^Na_iy_i=0$, and $w=\sum_{i=1}^Na_iy_i\phi(x_i)$
- kernel function: $K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$
- if the kernel function satisfies Mercer's condition (the kernel matrix is positive semi-definite), then there exists a mapping function $\phi(\cdot)$ such that $K(x_i,x_j)=\phi(x_i)^T\phi(x_j)$
- Gaussian (RBF) kernel: $K(x_i,x_j)=\exp\left(-\dfrac{\|x_i-x_j\|_2^2}{2\sigma^2}\right)$
- dual: maximize $L(a)=\sum_{i=1}^Na_i+b\sum_{i=1}^Na_iy_i-\dfrac{1}{2}\sum_{i=1}^N\sum_{j=1}^Na_ia_jy_iy_jK(x_i,x_j)$
  
  s.t. $0\le a_i\le \lambda,\forall i=1,2,\ldots,N,\sum_{i=1}^Na_iy_i=0$
- decision function: $f(x)=\text{sign}\left(\sum_{i=1}^Na_iy_iK(x_i,x)-b\right)$
