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
    wstd = 2
    lr = 0.01
    reg = 0.01
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
    wstd = 0.01
    lr_vanilla = 0.03
    lr_momentum = 3e-3
    lr_rmsprop = 1.6e-4
    reg = 0.01
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
    wstd = 0.1
    lr = 0.003
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""
**Your answer:**

1. The graph of no-dropout vs dropout does match what I expected to see. The no-dropout graph shows a clear overfitting, 
    with training accuracy close to 100% and poor test accuracy (around 26% at the peak).
The dropout graph shows a better generalization (test accuracy), and the train accuracy is closer to the test 
    accuracy (though in the low-dropout also much higher).

2. The low-dropout setting still shows signs of overfitting, because the trainning accuracy is still much higher than the
    test accuracy (~70% vs. ~30%), though the gap is smaller than in the no-dropout case.
In the high-dropout setting, the trainning accuracy is much similar to the test accuracy, which means there is no overfitting,
    but still the test accuracy is lower (but very close) to the low-dropout setting. This might be due to the fact that
    the dropout hyperparameter is too high, and the model is not learning enough.
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
    n_layers = 3
    hidden_dims = 64
    activation = "relu"
    out_activation = "none"
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
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 1e-3
    weight_decay = 1e-4
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part3_q1 = r"""
**Your answer:**
1. Optimization error is the difference between the loss of the model and the best possible loss for a function from our hypothesis loss. Looking at the decision boundary plot, it doesn't seem we could've trained a signinifcantly better function from this hypotehsis class, so I don't think this error is high.

2. Generealization error is the difference between the error on the training and the error on the test. Looking the at the test loss and accuracy, there's some generealization error, and I'd definitely try to do early stopping, in order to prevent overfitting.

3. Approximation error is the error due to the difference between the best possible function in the hypothesis space, and the true function. It's indicative of how expressive our hypothesis space is. In our case, looking at the decision boundary, we see points that cross the boundary, and so inevitably we will have some approximation error. Knowing that this is caused by noise in our data, and not internal features, I'd be heistatnt to try to improve on this (i.e. I don't think this error is high), since reducing this error means overfitting the noise. 
"""

part3_q2 = r"""
**Your answer:**
When comparing the FPR and NPR across the test set and the validation set, we notice significant differences On the test set:
FNR: 6.92%, FPR: 5.46%
And on the validation set:
FNR: 20.68%, FPR: 2.53%

This difference disappears when we generate our dataset by shuffling. Therefore, my explanation for this difference is: When we don't shuffle our data, our test and validation set represent different distributions. Specifically in our case, the test set will contain more samples from moon1, and the validation test will contain more samples from moon2. This causes class imbalance, which means our model will be biased towards the class that is more represented in the test set. Seeing that this difference disapperas When we shuffle our data is a good indication, is a good indication that this explanation is correct.

Note - the shuffling of the dataloader doesn't change, because that is shuffling within the already biased data.
"""

part3_q3 = r"""
**Your answer:**
Currently we choose a threshold that maximizes TPR-FPR. This tries to find a balance optimizing both sensitivity and specificity. In the scenarios presented, we wish to optimize for different metric. 

A) In this case, the cost of False Negative is low, because those cases later can be solved easily - symptomps do devlop, are evident, and a cure is cheap. Therefore we will choose a higher threshold, that will reduce the number of False Positives, and increase the number of False Negatives.

B) Here, the cost of a FN is much higher. Therefore we will choose a lower threshold. This will reduce the number of False Negatives, which is what we want to optimize for.

In both cases, The specific optimal threshold can be derived for a specific cost function, probably by differentiating some function.
"""


part3_q4 = r"""
**Your answer:**
1. The most striking difference is the shape of the line. A width of 2 allong only for roughy linear boundary decissions boundary,  while a higher width allows more curved boundaries. As expected, this allows the model to fit the training data better, since it comes from a moon distribution, and since the test distribution is also "moon-like" we get that the test accuracy is also higher for higher width. 

2. This difference is less significant than the width. Usually, more depth allows for more complicated feature extraction. However, since
the data is relatively simple, the model doesn't benefit much from the extra depth. This is why we see that test accuracy is not that different when increasing the depth.

3. In our case, the results are failry similar. (depth=4,width=8) has 87.6% valid_acc, and 85.8% test_acc, whereas (depth=1, width=32) has 87.4% valid_acc, and 87% test_acc. This shows us that this model doesn't benefit too greatly from extra depth, and is able to generate a sufficient hypothesis function with a single layer.

4. Explain the effect of threshold selection on the validation set: did it improve the results on the test set? why?
In our case, when using threshold=0.5, we get 87.6% test accuracy (with loss of 0.267), and when using optimal threshold of approx 0.200, we get 89.1% test accuracy, with (Avg. loss of 0.266). This results makes sense, since the threshold we choose is the one that is guaranteed to separate between the two classes the best, and so it makes sense that it will improve the results on the test set, i.e. that will indeed generalize better.

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
    loss_fn = torch.nn.CrossEntropyLoss()
    lr = 1e-2
    weight_decay = 1e-3
    momentum = 0.9
    # ========================
    return dict(lr=lr, weight_decay=weight_decay, momentum=momentum, loss_fn=loss_fn)


part4_q1 = r"""
**Your answer:**
1. The formula for the number of parameters for a convolutional layer is $(C_{in} \cdot K^2 + 1) \cdot C_{out}$,
    Where $C_{in}$ is the number of input channels, $C_{out}$ is the number of output channels, 
    $K$ is the size of the kernel and the plus 1 comes from the bias term.
    
    For the regular residual block, we have 2 layers, each with 3x3 kernel and 256 input and output channels.
    Therefore, the number of parameters is $(256 \cdot 3^2 + 1) \cdot 256 \cdot 2 = 1180160$.

    For the bottleneck residual block, we have 3 layers -
    the first with 1x1 kernel, 256 input and 64 output channels,
    the second with 3x3 kernel and 64 input and output channels,
    and the third with 1x1 kernel, 64 input and 256 output channels.
    Therefore, the number of parameters is
    $(256 \cdot 1^2 + 1) \cdot 64 + (64 \cdot 3^2 + 1) \cdot 64 + (64 \cdot 1^2 + 1) \cdot 256 = 70016$.

    We see that the bottleneck block has significantly less parameters, and so it is more efficient.

2. The number of floating point operations is proportional to the number of parameters, and so the bottleneck
    block will also be more efficient in terms of FLOPs.

3. Spatially: The regular block have 2 layers of 3x3 kernel, meaning receptive field of 5x5, while the bottleneck block 
    has 3 layers of 1x1, 3x3, 1x1 kernel, meaning receptive field of 3x3. This means that the regular block can
    capture more complex features.

    Across feature maps: The regular block combines information across all 256 features maps directly in both 3x3 
    convolutional layers. On the other hand, the bottleneck Block Combines information primarily through the middle
    3x3 convolution, which operates on only 64 channels. However, some limited information exchange still occurs
    during the 1x1 convolutions due to their linear transformations.
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
