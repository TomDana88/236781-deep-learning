r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 1 answers

part1_q1 = r"""
**Your answer:**

**1. False**

The in-sample error is the error on the data used to train the model, and we use the 
    validation set to estimate it.
The test set is used to estimate the out-of-sample error.

**2. False**

The train-test split should be random and not arbitrary. 
For example, if the data is sorted by the label, it's possible that one of the sets will
    contain only examples of one class.

**3. True**

The cross-validation is done during the training phase, to estimate the model's 
    generalization error and to choose the best hyperparameters.
The test set shouldn't be used during the training phase at all, to avoid overfitting
    the model to the test set, and to allow us to evaluate the model's generalization
    error on unseen data.

**4. True**

The purpose of splitting the training set into a train and validation sets is to estimate
    the model's generalization error, and use it to choose the best hyperparameters.
Note that we're using the validation error as a proxy for the generalization error only
    for choosing the hyperparameters and tuning the model, and not for evaluating the
    model's performance, which is done using the test set.
"""

part1_q2 = r"""
**Your answer:**

"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
"""

part2_q2 = r"""
**Your answer:**
"""

part2_q3 = r"""
**Your answer:**
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**

The ideal pattern to see is all the points scattered around the horizontal axis (where $y-\hat{y}=0$).
This means that the model's predictions are very close to the true values.

Based on the plots above, we can see that the model's predictions where improved after performing the CV compared to 
the top-5 features model, but the model still not performing well enough, since we can see that the points are not
scattered around the horizontal axis and the dashed lines representing the $\pm \text{std}$ are not very close to the horizontal
axis.


"""

part3_q2 = r"""
**Your answer:**

1. Yes, this is still a linear regression model, since the model is still linear in the parameters $\mat{W}$,
    even though the features are non-linear.

2. No. While adding non-linear features to the data can help us fit non-linear functions, it's not guaranteed
    that we can fit any non-linear function of the original features with this approach. For example, if we have
    a non-linear function that is not a polynomial (e.g. log), we can't fit it using polynomial features.
    Beacause there are infinite non-linear functions, and we can only add a finite number of features, we can't
    fit all non-linear functions.

3. Adding non-linear features will probably change the decision boundary. In the original feature space the
    decision boundary may no longer be a hyperplane, because we now can fit non-linear functions between the 
    original features. In the new feature space the decision boundary will still be a hyperplane, since
    the model is still linear in the transformed features. 
"""

part3_q3 = r"""
Regarding the cross-validation:

1. When defining the range for $\lambda$ the in the above CV code, why do you think we used
   `np.logspace` instead of `np.linspace`? Explain the advantage for CV.
1. How many times in total was the model fitted to data (with the parameters as given, and not including the 
   final fit on the entire training set)?

**Your answer:**

1. Using `np.logspace` allows us to sample a wider range of values for $\lambda$ with different scales more 
    efficiently. It's useful when because we don't know the scale of the best value for $\lambda$.

2. There are 3 possible values for `degree` and 20 possible values for `lambda`.
    We used `GridSearchCV` which fits the model with all possible combinations of the parameters.
    We used `k_folds=3` which means that for each combination of parameters the model will be fitted 3 times
    (each time with $\frac{1}{3}$ of the samples as validation set).
    Therefore, the model was fitted 3 * 20 * 3 = 180 times.
"""

# ==============

# ==============
