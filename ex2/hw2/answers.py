r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 (Backprop) answers

part1_q1 = r"""
**Your answer:**
## 1
* A. When $\boldsymbol{Y}$ is considers a function of $\boldsymbol{X}$, then for a single sample, $\pderiv{\mat{Y}}{\mat{X}}$  would have the shape $ (512,1024) $. Since we have $N=64$ samples, then $\pderiv{\mat{Y}}{\mat{X}}$  would have the shape $(64,512,1024)$.
* B.  No, the above Jacobian is not (generally) sparse. This is because: consider a single sample  $x$  and a single sample output $y$. Then $\frac{\partial y_i}{\partial x_j} = W_{ij}$, so the Jacobian is the weight matrix $W$. Since the weight matrix is not sparse, the Jacobian is not sparse as well.
* C. No, Using the chain rule, we get:
$$\delta \boldsymbol{X} = \delta \boldsymbol{Y} \frac{\partial \boldsymbol{Y}}{\partial \boldsymbol{X}} $$
Notice we don't even have to store the entire Jacobian in memory, but rather, we can compute each column of $\frac{\partial \boldsymbol{Y}}{\partial \boldsymbol{X}} $, and evalute the inner product of that column with $\delta \boldsymbol{Y}$, and thus compute $\delta \boldsymbol{X}$ increcementally, without storing the entire Jacobian matrix.
## 2
* A. Let's start with a single sample. When $\boldsymbol{Y}$ is considered a function of $\boldsymbol{W}$, then $\pderiv{\mat{Y}}{\mat{W}}$ has the shape $(512,1024, 512)$ . This is because for each component $y_i$ is a scalar-valued function, that gets an input $\boldsymbol{W}$ of dimensions $(512,1024)$, and so $\pderiv{y_i}{\mat{W}}$  is of shape $(512,1024)$, and since $\vec{Y}$ is of dim $512$, we have that $\frac{\partial \boldsymbol{Y}}{\partial \boldsymbol{W}}$ is a  3-tensor of dimensions $(512,1024,512)$. For $N=64$ samples, we have  that the shape of $\frac{\partial \boldsymbol{Y}}{\partial \boldsymbol{W}}$  is $(64,512,1024,512)$.

* B. The above Jacobian we've described would be mostly sparse. Since $\vec{Y}=\vec{X}\mat{W}^T$,  $y_i$ only depends on the $i$-th row of $\mat{W}$, and zeroes for all other rows.
* C. No, Using the chain rule, we get:
$$\delta \boldsymbol{W} = \delta \boldsymbol{Y} \frac{\partial \boldsymbol{Y}}{\partial \boldsymbol{W}} $$
Notice we don't even have to store the entire Jacobian in memory, but rather, we can compute $\frac{\partial \boldsymbol{Y}}{\partial \boldsymbol{W}} $, and evalute the inner product of that column with $\delta \boldsymbol{Y}$, and thus compute $\delta \boldsymbol{W}$ increcementally, without storing the entire Jacobian matrix.
"""

part1_q2 = r"""
**Your answer:**
**NO**, back-propagation is not necessary for descent-based optimization, it is just an efficient way to compute the gradients of the loss function with respect to the weights, using the chain rule. There are other ways, like evolutionary algorithms, that optimize the weights using feedback. There's also numerical differentiation, which approxiamtes the gradient by averaging the loss over multiple points.
"""


# ==============
# Part 2 (Optimization) answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp():
    wstd, lr_vanilla, lr_momentum, lr_rmsprop, reg, = (
        0,
        0,
        0,
        0,
        0,
    )

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different learning rates for each optimizer.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        wstd=wstd,
        lr_vanilla=lr_vanilla,
        lr_momentum=lr_momentum,
        lr_rmsprop=lr_rmsprop,
        reg=reg,
    )


def part2_dropout_hp():
    wstd, lr, = (
        0,
        0,
    )
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part2_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============


# ==============
# Part 3 (MLP) answers


def part3_arch_hp():
    n_layers = 0  # number of layers (not including output)
    hidden_dims = 0  # number of output dimensions for each hidden layer
    activation = "none"  # activation function to apply after each hidden layer
    out_activation = "none"  # activation function to apply at the output layer
    # TODO: Tweak the MLP architecture hyperparameters.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(
        n_layers=n_layers,
        hidden_dims=hidden_dims,
        activation=activation,
        out_activation=out_activation,
    )


def part3_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part3_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""
# ==============
# Part 4 (CNN) answers


def part4_optim_hp():
    import torch.nn
    import torch.nn.functional

    loss_fn = None  # One of the torch.nn losses
    lr, weight_decay, momentum = 0, 0, 0  # Arguments for SGD optimizer
    # TODO:
    #  - Tweak the Optimizer hyperparameters.
    #  - Choose the appropriate loss function for your architecture.
    #    What you returns needs to be a callable, so either an instance of one of the
    #    Loss classes in torch.nn or one of the loss functions from torch.nn.functional.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 5 (CNN Experiments) answers


part5_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part5_q4 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""


part6_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part6_bonus = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""