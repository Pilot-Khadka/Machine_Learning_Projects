import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

"""
Margin -> shortest distance between observation and threshodl
Maximal margin classifier -> sensetive to outliers

Support vector machines:
    -> start with data in relatively low dimension
    -> move the data into higher dimension
    -> find support vector classfier that reperates the higher dim data into two groups
    
uses kernel functions to find support vecotr classifers in higher dim:
    -> Eg: Polynomial kernel (axb+r)^d
        -> computes relationship between each pair of observations
        -> d=1 1-d relation between each pair of observation
        -> d=2 2-d relation between each pair of observation
    
    -> Radial kernel:
        -> finds SVC in infinite dimension
        -> behaves like weighted nearest neighbors 
        
Kernels calculate the high dim relationship, but dont actually transform data to higher dim
    This is called Kernel trick (reduces computation)
    
Classification:
    -> Hyperplane(y =mx+c)
    -> W -> parameters of line (m,c)
    -> W transpose x is used to compute whether the point lies in left or right side of hyperplane
    -> Eg: W = [-1,0] , x = (3,3), Wtx + b = [-1 ; 0] [3,3] = -3 (negative)

Which best hyperplane?
    -> Maximize margin value (X1 - X2)
    -> hyperplane (Wtx+b= label), for points, Wtx2 +b=-1, Wtx1+b=1
    -> Wt(x1-x2) = 2
    -> wt (x1-x2) / ||w|| = 2 / ||w||
    -> x1-x2 = 2/ ||w||
    -> label = -1 if wtx1+b <=-1
    -> label = 1 if wtx1+b >=1
    -> max(||w||/2)
    
"""

class SVM:
    def __init__(self, learning_rate=0.01, num_iteration=500, lambda_parm=0.001, kernel_type='linear'):
        self.learning_rate = learning_rate
        self.num_iteration = num_iteration
        self.lambda_parm = lambda_parm
        self.kernel_type = kernel_type

    def fit(self,x,y):
        self.num_data = x.shape[0]
        self.num_features = x.shape[1]
        self.w = np.zeros(self.num_features)
        self.b = 0
        self.x = x
        self.y = y
        self.y_label = np.where(self.y <=0,-1,1)

        for _ in range(self.num_iteration):
            self.update()

    def update(self):
        dw, db = self.calculate_gradients()
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db

    def calculate_gradients(self):
        dw = 0
        db = 0

        for x_i, label in zip(self.x, self.y_label):
            margin = label * (np.dot(x_i, self.w) - self.b) >= 1
            condition = margin >=1
            dw += self.lambda_parm * (2 * self.w) if condition else (2 * self.lambda_parm * self.w - label * self.kernel(x_i))
            db += 0 if condition else label

        dw /= self.num_data
        db /= self.num_data
        return dw,db

    def predict(self,x):
        out = np.dot(x, self.w) - self.b
        predicted_labels = np.sign(out)
        y_hat = np.where(predicted_labels <=-1, 0,1)
        return y_hat

    def kernel(self,x, r=1, d=2):
        if self.kernel_type == 'polynomial':
            return (np.dot(x, self.x.T)+r) ** d
        elif self.kernel_type == 'radial':
            return (np.exp(-np.linalg.norm(x-self.x,axis=1)**2 / (2*self.d **2)))
        else:
            return x

    def hinge_loss(self,y_true,y_pred):
        """
            -> used for maximum margin classification models
        :return:loss
        """
        loss = max(0, 1 - y_true * y_pred)
        return loss

    @staticmethod
    def calculate_accuracy(pred, actual):
        if len(pred) != len(actual):
            raise ValueError("Arrays must have the same length")

        correct = sum(1 for a, b in zip(pred, actual) if a == b)
        total = len(pred)

        accuracy = correct / total
        return accuracy

def generate_data(n_samples=1000):
    x,y = make_classification(n_samples=n_samples, n_features=2,n_informative=2, n_redundant=0, n_clusters_per_class=1)
    return x,y


if __name__ == "__main__":
    model = SVM(learning_rate=0.001, num_iteration=100, lambda_parm=0.01)
    x,y = generate_data()
    print(len(x))
    print(len(y))
    plt.scatter(x[:,0], x[:,1], c=y)
    # model.fit(data[:-50],target[:-50])
    #
    # plt.scatter(data[0], data[1])
    # pred = model.predict(data[-50:])
    # print(pred)
    # print(target[-50:])
    # loss = model.hinge_loss(pred,target[-50:])