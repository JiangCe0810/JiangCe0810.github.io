# Weak2
本周主要介绍了神经网络中forward和backward的一般实现和向量实现。一般实现较为简单，向量实现中存在一些疑点
\begin{align}
\boldsymbol{X}
\end{align}是一个由训练集组成的矩阵，每一列代表一个数据，列数为数据的大小
$ \boldsymbol{\omega}$ 是训练参数，大小与 $\boldsymbol{X}$一列的大小一致
$b$ 为偏差(bias)，为一个标量
## forward
$\boldsymbol{Z} = np.dot(\boldsymbol{\omega}.T,\;\boldsymbol{X}) + b$ 
$\boldsymbol{A} = \sigma(\boldsymbol{Z})$，其中 $\sigma(\boldsymbol{x}) = \frac{1}{1-e^{-\boldsymbol{x}}}$
通过编程实现为 $1/(1-np.exp(-\boldsymbol{X}))$
Cost Function(Loss Function)通过矩阵实现时应该注意Cost Function是将所有的预测误差相加取平均得到的，**不可以直接用矩阵乘法使其变为标量**
\begin{align}
 L = 1/m*np.sum((-\boldsymbol{Y}*np.log(\boldsymbol{A})+(1-\boldsymbol{Y})*np.log(1-\boldsymbol{A})))
\end{align}，其中m为样本的个数
## backward
backward实际上是一个链式求导的过程，backward最根本的式子是通过梯度下降法来更新w和b
\begin{align}
\frac{\partial L}{\partial \boldsymbol{A}} = -(\frac{\boldsymbol{Y}}{\boldsymbol{A}}-\frac{1-\boldsymbol{Y}}{1-\boldsymbol{A}})
\end{align}
\begin{align}
\frac{\partial \boldsymbol{A}}{\partial \boldsymbol{\boldsymbol{Z}}} = \boldsymbol{A}(1-\boldsymbol{A})
\end{align}
\begin{align}
\frac{\partial \boldsymbol{Z}}{\partial \boldsymbol{\omega}} = \boldsymbol{X}
\end{align}
所以，我们可以表示 $d\omega$ 为$ d\omega = np.dot(\boldsymbol{X},\;(\boldsymbol{A} - \boldsymbol{Y}).T)$，这个求解出来为m个样本训练出w的变化总和，因此应该除以m，所以为$ d\omega =1/m* np.dot(\boldsymbol{X},\;(\boldsymbol{A} - \boldsymbol{Y}).T)$。
同理可求$db$，但是由于b为标量，因此需要对求出的m次训练的b求和，即 $db = 1/m*np.sum(\boldsymbol{A}-\boldsymbol{Y})$。根据这两个值即可以更新 $\omega$ 和 $b$



