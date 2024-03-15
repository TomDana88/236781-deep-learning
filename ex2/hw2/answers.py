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

1. The graph of no-dropout vs dropout does match what we expected to see. The no-dropout graph shows a clear overfitting, 
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

Yes, it's possible, because the cross-entropy loss and the accuracy measure different things. The loss is a measure of
how confident the model is in its predictions, while the accuracy is a measure of how many predictions the model got right.
During training, the model might make changes that will help it correcly classify example from the test set, which will 
lead to an increase in test accuracy. However, these changes might also make the model become overconfident in its wrong 
predictions for other examples, which will lead to an increase in the loss.
"""

part2_q3 = r"""
**Your answer:**

1. Gradient descent is an optimization algorithm, which tells the model how to learn in each iteraion. i.e. how to 
    update the weights.
    Back-propagation is a method for computing the gradient of the loss function with respect to the weights of the model.
    In other words, gradient descent is the algorithm that uses the gradients computed by back-propagation to update the weights.

2. Both GD and SGD are optimization algorithms used train models by minimizing a cost function.
    GD uses the entire training dataset in each iteration to calculate the gradient, while uses only a single data point
    or a small mini-batch randomly chosen from the training dataset in each iteration.
    This means that GD is generally more accurate, and will always converge to a (global or local) minimum, as opposed 
    to SGD, which is not guaranteed to converge to a minimum because of the noise introduced by the random sampling.
    However, GD is much slower than SGD, requiring a lot of memory and computational power, and might not be feasible for
    large datasets.

3. There are few reasons why SGD is used more often in the practice of deep learning (DL):
    * Scalability: DL often deals with massive datasets that wouldn't fit in memory entirely. Processing the 
    entire dataset (as GD does) becomes impossible computationally. SGD, by using just a single data point or a 
    mini-batch, requires significantly less memory, making it feasible to train deep models on large datasets.
    * Efficiency: SGD iterates through the training data much faster than GD. Because it calculates the gradient for 
    only one data point. This speed advantage allows for quicker training of complex deep networks, which wouldn't be
    feasible with GD.
    * Regularization: The inherent noise introduced by SGD's randomness is acting as a regularizer. Because SGD 
    calculates the gradient of only one random example each time, it's not following the exact gradient direction like 
    GD, which can help the model avoid getting stuck in local minima and potentially converge to a better solution.

4.  A. Yes, this approach would produce a gradient equivalent to GD, because the gradient operator is linear, and so
    the gradient of the sum of the losses is the sum of the gradients of the losses. 
    Mathematically:
    $$
    ∇ L(\theta; X) = ∇ \sum_{i=1}^N L(\theta; x_i) = \sum_{i=1}^{n_1} ∇ L(\theta; x_i) + \cdots + \sum_{i=n_{m-1}+1}^{n_m} ∇ L(\theta; x_i)
    $$
    Where $n_i$ is the number of examples in the $i$-th batch, and $m$ is the number of batches.
    B. We will get an out of memory error because we are storing the gradients of all the batches in memory for the 
    backword pass, so we're still saving a lot of data for each batch, even though we are not saving the entire dataset.


"""

part2_q4 = r"""
**Your answer:**

1. 
    **Forward mode AD**:
    In the tutorial we saw that the formula for calculatin the gradient of $v_{j+1}$ is:
    $$
    v_{j+1}.grad \leftarrow v_{j+1}.fn.derivative(v_j.val) \cdot v_{j}.grad
    $$
    Instead of storing the gradients of all nodes, we can use a single variable that will accumulate all the gradients,
    and a single variable that will store the previous node.
    In each iteration we'll use the following formula:
    $$
    \text{acc\_grad} \leftarrow \text{acc\_grad} \cdot f_i.derivative(\text{prev\_val})
    $$
    $$
    \text{prev\_val} \leftarrow f_i(\text{prev\_val})
    $$
    Where $\text{acc\_grad}$ is initialized to $1$ and $\text{prev\_cal}$ is initialized to $x$.\
    Memory complexity is $\mathcal{O}(1).$

    **Backward mode AD**:
    Same as before, we'll use two variables and the following formulas:
    $$
    \text{acc\_grad} \leftarrow \text{acc\_grad} \cdot f_{i-1}.derivative(\text{prev\_val})
    $$
    $$
    \text{prev\_val} \leftarrow f_i(\text{prev\_val})
    $$
    Where $\text{acc\_grad}$ is initialized to $1$ and $\text{prev\_val}$ is initialized to $v_n.val$. \
    Memory complexity is $\mathcal{O}(1)$. \
    Note that this time we're iterating the graph in reverse order.

2. These techniques are not directly generalizable to arbitrary computational graphs because they rely on the assumption
    that the given function is constructed of sequential compositions of functions.
    However, it can be applied to specific sub-graphs within a larger network if they exhibit this sequential structure.

3. The back-propagation algorithm can consume a lot of memory when working on deep networks, because it's memory complexity
    is linear with the depth of the network. By using techniques that reduce the memory consumption, we'll get a significant
    reduction in memory usage in deep networks.

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

1. We can see that the greater the depth, the better the accuracy for both `K=[32]` and `K=[64]`, although for `K=[32]`
   there is no big difference between `L=8` and `L=16` as they both have similar accuracy.
   The best results are achieved with `L=16`, which is the highest depth we tested. This is because the higher the depth,
   the more complex features the network can learn, and so it can better capture the complexity of the data.
2. We did encounter a situation where the network wasn't trainable with `L=16`, but we managed to resolve it by using
   the LeakyReLU activation function instead of regular ReLU. The problem we were facing was probably vanishing gradients,
   and the LeakyReLU activation function allows for a non-zero gradient, which helped solve it.
   This could have also been resolved by using skipping connections, like in ResNet.
   For `K=[64]` and `L=16` we can see a big drop in the test accuracy for one epoch, which also might caused by vanishing
   gradients, but the network overcame it and continued the training.
"""

part5_q2 = r"""
**Your answer:**

For `K=[32]` and `K=[64]`, abviously the results are very similar to the results of experiment 1.1 (with corresponding 
`L` values), since we used the same hyper-parameters. For the case of `K=[128]`, we see that as we increased `L`, we got better
results. The difference between `L8_K128` and `L8_K64` is not very significant, which indicates that increasing the number
of filters (`K`) drastically without increasing the depth (`L`) doesn't help much. We can also see that insight by 
comparing `L16_K64` from experiment 1.1 and `L8_K128` from experiment 1.2, where the former achieved better results.
"""

part5_q3 = r"""
**Your answer:**

As before, we can see that for fixed `K` value, increasing `L` improves the results.
Compared to the results of the previous experiments with the same `L` values, we can see that the results are much better,
which indicates that adding another convolutional block to the network improves the results significantly.
"""

part5_q4 = r"""
**Your answer:**

exp1.1:
* `K=[32]` fixed with varying `L=2,4,8,16`
* `K=[64]` fixed with varying `L=2,4,8,16`

exp1.3:
* `K=[64, 128]` fixed with varying `L=2,3,4`.

exp1.4:
* `K=[32]` fixed with varying `L=8,16,32`.
* K=[64, 128, 256] fixed with varying L=2,4,8.

The best results in experiment 4 (and in total) were achieved with `L8_K64-128-256`. This tells us that
increasing the number of convolutional blocks in the network (adding values to `K`) is more beneficial than
increasing the number of filters filters in each layer (value of items in `K`) or increasing the depth of each
block (value of `L`).
That being said, we did get good results also with `L32_K32`, which are similar (but more stable) to the results
of `L16_K64` from exp1.1 and `L4_K64-128` from exp1.3. This tells us that increasing `L` or values in `K` can
also improve the results to some extent.
Another thing we see from the results is that the results of `L2_K64-128` and `L4_K64-128` from exp1.3 are better 
than the results of `L2_K64-128-256` and `L4_K64-128-256` from exp1.4 respectively, which tells us that adding 
more blocks without increasing the depth of each block can actually hurt the results.
However, we ran exp1.4 with hyper-parameter `pool_every=5`, compare to `pool_every=3` in exp1.3. We did get 
better results with `pool_every=3` in exp1.4 for `L=2,4` but got an error for `L=8`, so we can't tell of the
results of exp1.3 were better because of the hyper-parameter or because of the added block.
This also applies to the first part of exp1.4, which we ran with `pool_every=8` because of error occured with
`L32_K32`.
"""


# ==============

# ==============
# Part 6 (YOLO) answers


part6_q1 = r"""
**Your answer:**
1. The model did relatively bad. For the dolphins image, all three dolphins were misclassified, two dolphins were classified as "person", and 1 dolphin was classified as a surfboard. For the 1 cat and 3 dogs image, only 1 dog was classified correctly, 1 dog was bounded with the cat, and one dog was misclassified as a cat. It is noteworthy that it did detect the correct number of objects in both images. 

-------------
2. Here are several possible reasons why the model failed:  

a) As for the misclassification of the dolphins, it might be that the model was never trained on dolphins, and hence couldn't recognize them. A solution to this problem would be to train the model on a larger dataset, that includes dolphins. This doesn't even necessarily requires a full re-train, but possibly can be achieved using fine-tuning of the last layers of the model. 

b) For the misclassification of the dog as a cat, due to object overlap, this is can be caused because of a  lack of such cases in the training data, or that the model is too simple, and possibly couldn't cannot capture such level of difference. Possible solutions are data augmentation, that will increase the model ability to generalize. Another possible solution is to increase the model architecture, which would help the model capture more fine-level details. 
"""


part6_q2 = r"""
**Your answer:** There was no question

"""


part6_q3 = r"""
**Your answer:** 
The model mostly failed to detect the objects

The model misclassified the blurry cat as a person, but it did provide a correct bounding box. This indicates that the parts of the model related to grouping pixels were able to handle the blurriness, however when it came to extracting features specific to classification, those parts did not handle the bluriness. A possible solution would be to train the model on blurrier images using data augmentation. 

I've also tested occlusion. I've provided a clear image with only a fork to demonstrate that the model is able to detect a fork, and in that simple image the model succeeded. However in every other case it didn't detect the fork, and misclassified many other things. Even in the last image, where the fork is visible, the model isn't able to detect it. Another interesting case is the textured background. Here, we get an image of only a fork, however the model misclassifies it as a "scissor", probably due to the different, unusual background. An hypothesis is that such a background usually occurs more in cases where scissors are involved than with forks, and that could have tipped the model classification in that direction.
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
