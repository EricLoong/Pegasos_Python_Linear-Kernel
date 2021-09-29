import numpy as np
from random import randint
from sklearn.metrics import hinge_loss
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def pegasos(x, y, weights=None, max_iterations=2000, lam=1):
    if type(weights) == type(None): weights = np.zeros(x[0].shape)
    n_samples = len(y)
    loss = list()
    for i in range(max_iterations):
        sl = randint(0, n_samples - 1)
        step = 1 / (lam * (i + 1))
        decision = y[sl] * weights @ x[sl].T
        if decision < 1:
            weights = (1 - step * lam) * weights + step * y[sl] * x[sl]
        else:
            weights = (1 - step * lam) * weights
        pred_decision = np.dot(x, weights.T)
        # Note that the loss is for binary classification hinge loss rather than multi-class loss
        temp_loss = hinge_loss(y_true=y, pred_decision=np.array(pred_decision))
        loss.append(temp_loss)
        # Constrain the weights into a sphere.
        # weights = min(1, (1/math.sqrt(lam))/(np.linalg.norm(weights)))*weights
    return weights, loss


def preprocess_binary(y, target_i):
    y = np.array(y)
    y[np.where(y != target_i)] = -1
    y[np.where(y == target_i)] = 1
    return y

# If wanna use binary classifier, refer to another pegasos.py

class pegasos_linear_multiclass():
    # We record the binary classification loss, the number of class detected, the learning rate
    # lambda and the weights for one-vs-all classifiers.
    def __init__(self, lmd=0.0001, max_iter=10000):
        self.W = np.ndarray
        self.lmd = lmd
        self.Cl = None
        self.max_iter = max_iter
        self.loss = None
    def fit(self, x: np.ndarray, y: np.ndarray, W=None):
        num_class = len(np.unique(y))
        self.Cl = num_class
        dim = x.shape[1]
        if type(W) == type(None):
            W = np.zeros((num_class, dim))
        self.W = W
        self.loss = np.zeros((self.Cl, self.max_iter))
        for i in range(num_class):
            temp_y = preprocess_binary(y, target_i=i)
            temp_w, self.loss[i,:]  = pegasos(x, y=temp_y, weights=W[i, :], lam=self.lmd, max_iterations=self.max_iter)
            W[i, :] = temp_w
        self.W = W

    def predict_eval(self, x_new, metric=accuracy_score, y_true=None):
        weight = self.W
        n_s = x_new.shape[0]
        pred_c = list()
        for i in range(n_s):
            # select maximum value as the predicted class, since the decision value shows
            # the probability of class, which is also used by sklearn package
            decision_list = np.array(np.dot(x_new[i], weight.T))
            pred_class = np.where(decision_list==max(decision_list))[0][0]
            pred_c.append(pred_class)
        output = np.array(pred_c)

        if y_true is not None:
            acc = metric(y_pred=output, y_true=y_true)
            return output, acc
        else:
            return output


from sklearn.datasets import make_classification
import numpy as np

X, Y = make_classification(n_classes=4, n_samples=4000, n_clusters_per_class=1, random_state=120, n_features=5,
                           n_informative=2, n_redundant=0)

clf = pegasos_linear_multiclass(lmd=1, max_iter=2000)
clf.fit(x=X, y=Y)
loss_multiclass = clf.loss
acc = clf.predict_eval(x_new=X, y_true=Y)[1]
print('Pegasos SVM classification accuracy:', acc)

from sklearn.linear_model import SGDClassifier

clf_sgd = SGDClassifier()
clf_sgd.fit(X, Y)
clf_sgd.score(X, Y)

pred = clf_sgd.predict(X)
print('sklearn SVM classification accuracy', accuracy_score(y_true=Y, y_pred=pred))




