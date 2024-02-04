import torch
from torch import Tensor
from collections import namedtuple
from torch.utils.data import DataLoader

from .losses import ClassifierLoss

def to_scalar(obj):
    if isinstance(obj, torch.Tensor):
        return obj.item()
    else:
        return float(obj)
class LinearClassifier(object):
    def __init__(self, n_features, n_classes, weight_std=0.001):
        """
        Initializes the linear classifier.
        :param n_features: Number or features in each sample.
        :param n_classes: Number of classes samples can belong to.
        :param weight_std: Standard deviation of initial weights.
        """
        self.n_features = n_features
        self.n_classes = n_classes

        # TODO:
        #  Create weights tensor of appropriate dimensions
        #  Initialize it from a normal dist with zero mean and the given std.
        # ====== YOUR CODE: ======
        self.weights = torch.normal(0, weight_std, (n_features, n_classes)) # (D,C)
        # ========================

    def predict(self, x: Tensor):
        """
        Predict the class of a batch of samples based on the current weights.
        :param x: A tensor of shape (N,n_features) where N is the batch size.
        :return:
            y_pred: Tensor of shape (N,) where each entry is the predicted
                class of the corresponding sample. Predictions are integers in
                range [0, n_classes-1].
            class_scores: Tensor of shape (N,n_classes) with the class score
                per sample.
        """

        # TODO:
        #  Implement linear prediction.
        #  Calculate the score for each class using the weights and
        #  return the class y_pred with the highest score.

        y_pred, class_scores = None, None
        # ====== YOUR CODE: ======
        class_scores = torch.mm(x,self.weights)
        y_pred = torch.argmax(class_scores, dim=1)
        # ========================

        return y_pred, class_scores

    @staticmethod
    def evaluate_accuracy(y: Tensor, y_pred: Tensor):
        """
        Calculates the prediction accuracy based on predicted and ground-truth
        labels.
        :param y: A tensor of shape (N,) containing ground truth class labels.
        :param y_pred: A tensor of shape (N,) containing predicted labels.
        :return: The accuracy in percent.
        """

        # TODO:
        #  calculate accuracy of prediction.
        #  Do not use an explicit loop.

        acc = None
        # ====== YOUR CODE: ======
        num_samples = y.shape[0]
        num_correct = torch.sum(y == y_pred)
        acc = num_correct / num_samples
        # ========================

        return acc * 100

    def train(
        self,
        dl_train: DataLoader,
        dl_valid: DataLoader,
        loss_fn: ClassifierLoss,
        learn_rate=0.1,
        weight_decay=0.001,
        max_epochs=100,
    ):

        Result = namedtuple("Result", "accuracy loss")
        train_res = Result(accuracy=[], loss=[])
        valid_res = Result(accuracy=[], loss=[])

        print("Training", end="")
        for epoch_idx in range(max_epochs):
            total_correct = 0
            average_loss = 0

            # TODO:
            #  Implement model training loop.
            #  1. At each epoch, evaluate the model on the entire training set
            #     (batch by batch) and update the weights.
            #  2. Each epoch, also evaluate on the validation set.
            #  3. Accumulate average loss and total accuracy for both sets.
            #     The train/valid_res variables should hold the average loss
            #     and accuracy per epoch.
            #  4. Don't forget to add a regularization term to the loss,
            #     using the weight_decay parameter.

            # ====== YOUR CODE: ======
            for _ in range(max_epochs):
                # evaluate on the entire training set
                total_correct_train, total_loss_train = 0. , 0.
                for x_train_batch, y_train_batch in dl_train:
                    y_pred_train, class_scores_train = self.predict(x_train_batch)
                    loss_train = to_scalar(loss_fn.loss(x_train_batch, y_train_batch, class_scores_train, y_pred_train))
                    loss_train += to_scalar(weight_decay * torch.sum(self.weights ** 2))
                    total_loss_train += loss_train
                    total_correct_train += torch.sum(y_train_batch == y_pred_train)
                    # update the weights
                    grad = loss_fn.grad()
                    self.weights -= learn_rate * grad
                # compute average loss and accuracy on training set after training epoch
                average_loss_train = to_scalar(total_loss_train / len(dl_train))
                accuracy_train = to_scalar(total_correct_train / len(dl_train.dataset))
                train_res.accuracy.append(accuracy_train); train_res.loss.append(average_loss_train)

                # evaluate on the validation set
                total_correct_valid, total_loss_valid = 0. , 0.
                for x_valid_batch, y_valid_batch in dl_valid:
                    y_pred_valid, class_scores_valid = self.predict(x_valid_batch)
                    loss_valid = to_scalar(loss_fn.loss(x_valid_batch, y_valid_batch, class_scores_valid, y_pred_valid))
                    loss_valid += to_scalar(weight_decay * torch.sum(self.weights ** 2))
                    total_loss_valid += loss_valid
                    total_correct_valid += torch.sum(y_valid_batch == y_pred_valid)
                    # No need to update the weights, because in validation
                # compute average loss and accuracy on validation set(!) after training epoch
                average_loss_valid = to_scalar(total_loss_valid / len(dl_valid))
                accuracy_valid = to_scalar(total_correct_valid / len(dl_valid.dataset))
                valid_res.accuracy.append(accuracy_valid); valid_res.loss.append(average_loss_valid)
            # ========================
            print(".", end="")

        print("")
        return train_res, valid_res

    
    def weights_as_images(self, img_shape, has_bias=True):
        """
        Create tensor images from the weights, for visualization.
        :param img_shape: Shape of each tensor image to create, i.e. (C,H,W).
        :param has_bias: Whether the weights include a bias component
            (assumed to be the first feature).
        :return: Tensor of shape (n_classes, C, H, W).
        """

        # TODO:
        #  Convert the weights matrix into a tensor of images.
        #  The output shape should be (n_classes, C, H, W).

        # ====== YOUR CODE: ======
        weight_matrix = self.weights
        if has_bias:
            weight_matrix = weight_matrix[1:]
        w_images = weight_matrix.view(self.n_classes, *img_shape)
        # ========================

        return w_images


def hyperparams():
    hp = dict(weight_std=0.0, learn_rate=0.0, weight_decay=0.0)

    # TODO:
    #  Manually tune the hyperparameters to get the training accuracy test
    #  to pass.
    # ====== YOUR CODE: ======
    hp = dict(weight_std=0.001, learn_rate=0.001, weight_decay=0.0001)
    # ========================

    return hp
