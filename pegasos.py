from typing import List, Any

from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
import random

# trivial linearly seperable dataset
X, Y = make_classification(n_classes=2, n_samples=400, n_clusters_per_class=1, random_state=120, n_features=2,
                           n_informative=2, n_redundant=0)
Y[Y == 0] = -1
print(Y)
plt.scatter(X[:, 0], X[:, 1], c=Y)


class Pegasos_Linear_SVC():

    def __init__(self, C=1.0):
        self.C = C
        self.W = 0
        self.b = 0

    def HingeLoss(self, W, b, x, y):
        loss = 0
        loss += 0.5 * np.dot(W, W.T)
        m = x.shape[0]
        for i in range(m):
            t = y[i]*(np.dot(W, x[i].T)+b)
            loss += self.C*max(0, (1-t))
        return loss[0][0]

    def fit_model(self, x, y, Max_Iter = 100):
        n_features = x.shape[1]
        n_samples = x.shape[0]
        c = self.C
        W = np.zeros((1, n_features))
        bias = 0

        loss = list()

        ids = np.arange(n_samples)
        np.random.shuffle(ids)

        for j in range(1, Max_Iter):
            eta = 1/(c*j)
            g_w = 0
            g_b = 0
            k = ids[j]
            t = y[k] * (np.dot(W, x[k].T) + bias)
            if t > 1:
                g_w += 0
                g_b += 0
            else:
                g_w += c * y[k] * x[k]
                g_b += c * y[k]
            W = W - eta * W + eta * g_w
            bias = bias + eta * g_b
            I = self.HingeLoss(W, bias, X, Y)
            loss.append(I)
        self.W = W
        self.b = bias
        return W, bias, loss

    def predict(self, x):
        m, p = x.shape
        output = np.ones((m,1))

        if self.W.shape[1] != p:
            print('Dimension not matched')
        else:
            output = np.dot(x, self.W.T)+ np.ones((m,1))*self.b
        result = output>0
        result = result*1
        result[result == 0] = -1
        return result


def RBF(x,y,sigma = 10):
    above = np.linalg.norm(x-y)**2
    below = 2*(sigma**2)
    result = np.exp(-above/below)
    return result

def linear(x,y):
    return np.dot(x,y.T)

def polynomial(x,y,degree=3,alpha=1.0,Const=1.0):
    result = (Const+alpha*np.dot(x,y.T))**degree
    return result


class Pegasos_Kernelized_SVC_Batch(object):

    def __init__(self, kernel=RBF,Max_Iter=20, C=1.0, sigma=10,S=30):
        # kernel: the kernel function used
        # C is the penalty parameter for SVM
        # S is the mini-Batch Size
        # sigma is the variance of RBF function used
        # model store the trained model in a dictionary

        self.kernel = kernel
        self.C = C
        self.sigma = sigma
        self.model = dict()
        self.Max_Iter = Max_Iter
        self.S = S

    def __repr__(self):
        return (f"{self.__class__.__name__}("
                f"kernel={self.kernel.__name__}, "
                f"C={self.C}, "
                f"max_iterations={self.Max_Iter}, "
                f"sigma={self.sigma}"
                ")")
    def HingeLoss_kernel(self, alpha, K, y):
        loss = 0
        loss += 0.5 * np.dot(alpha.T,np.dot(K,alpha))*self.C
        m = K.shape[0]
        for i in range(m):
            t2 = y[i]*(1/m)*np.dot(alpha.T,K[:,i])
            loss += max(0, (1-t2))
        return loss

    def fit_model(self,x,y,Max_Iter=20,kernel=RBF,C=0.0001):
        self.kernel = kernel
        self.C = C
        start_time = time.time()
        # x and y are training examples and their corresponding labels respectively
        self.Max_Iter = Max_Iter
        m = x.shape[0]
        p = x.shape[1]
        x = np.insert(x, p, 1, axis=1)
        self.model['train'] = x
        self.model['label'] = y
        #x = np.hstack((x,np.ones((m,1))))
        # default Y has been transformed to -1 and 1

        # Compute the kernel matrix with sklearn
        if self.kernel.__name__ == 'linear':
            print(f'Computing {self.kernel.__name__} Kernel Matrix')
            K = pairwise_kernels(x, metric='linear')

        elif self.kernel.__name__ == 'RBF':
            print(f'Computing {self.kernel.__name__} Kernel Matrix')
            K = pairwise_kernels(x, metric='rbf',gamma = 1/(2 * (self.sigma ** 2)))

        else:
            print(f'Computing {self.kernel.__name__} Kernel Matrix')
            K = pairwise_kernels(x, metric='poly')

        # Initialize alpha which store the model information of Pegasos Kernalized version
        # the intercept of decision function is already included in w
        alpha = np.zeros((self.Max_Iter+1,m))
        losses: List[Any] = list()
        for i in range(1, Max_Iter):
            index = np.random.randint(0,m)
            eta = (1/ (self.C * i))
            y_it = y[index]
            for j in range(m):
                if j!= index:
                    alpha[i + 1, j] = alpha[i, j]
            sum_it = 0
            for j in range(m):
                sum_it += alpha[i,j] * K(index,x[j]) * y[j]
            sum_it *= y_it*eta
            if sum_it < 1:
                alpha[i+1,index] = alpha[i,index]+1
            else:
                alpha[i+1,index] = alpha[i,index]
            temp_alpha = np.array(alpha[i+1]).T
            loss = self.HingeLoss_kernel(alpha= temp_alpha,K=K,y=y)
            losses.append(loss)
        self.model['kernelFunction'] = self.kernel
        self.model['alpha'] = alpha[self.Max_Iter]
        alpha = alpha[Max_Iter]
        print("Training Time --- %s seconds ---" % (time.time() - start_time))
        return alpha, losses[1:]

    def predict(self, x):
        x = np.insert(x, x.shape[1], 1, axis=1)
        alpha = self.model['alpha']
        train_x = self.model['train']
        train_y = self.model['label']
        kernel_function =self.kernel
        m = alpha.shape[0]
        n = x.shape[0]
        pred = np.zeros((n,1))
        for i in range(n):
            decision = 0
            for j in range(m):
              # if use other kernel function, you should change the RBF
                decision += alpha[j] *train_y[j] * RBF(train_x[j].reshape(-1,1), x[i].reshape(-1,1))
            if decision < 0:
                prediction = -1
            else:
                prediction = 1
            pred[i] = prediction
        return pred
   
  
####Simple Test Part
import time
start_time = time.time()
mySVM = Pegasos_Linear_SVC()
W,b, loss = mySVM.fit_model(X,Y,Max_Iter=10)
print("--- %s seconds ---" % (time.time() - start_time))

print(loss[0])
print(loss[-1])
prediction = mySVM.predict(x=X)
plt.plot(range(2,10),loss[1:])
plt.show()




kSVM = Pegasos_Kernelized_SVC_Batch()
alpha, loss = kSVM.fit_model(x=X,y=Y,C=0.0001)
plt.plot(range(2,20),loss)
plt.show()
prediction = kSVM.predict(X)
def accuracy(y,y_pre):
    right = 0
    for i in range(len(y)):
        if y[i] == y_pre[i]:
            right +=1
    acc = right/len(y)
    return acc

print(accuracy(Y,prediction))

