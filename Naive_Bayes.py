#Importing the required libraries:
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from scipy.stats import norm

#Creating the Data:
X, y = make_blobs(n_samples = 10000, n_features = 2, centers = 2, random_state = 1)

print(X.shape, y.shape)

#Class that Performs Naive Bayes:
class NaiveBayes:
    """Class that performs Naive Bayes
    
    Parameters:
    ---------------
    X- Features
    y- Target variable
    splitRatio- Train/Test Split Ratio
    
    """
    
    def __init__(self, X, y, splitRatio = 0.3):
        """Initialising the parameters"""
        self.X = X
        self.y = y
        self.splitRatio = splitRatio
        
    def splitTrainTest(self):
        """Function to Split the data into Training set and Test set"""
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, 
                                                            train_size = self.splitRatio,
                                                            random_state = 0)
        return X_train, X_test, y_train, y_test
    
    def fitDistribution(self, X):
        
        mu = np.mean(X)
        std = np.std(X)
        dist = norm(mu, std)
        
        return dist
    
    def probability(self, X, prior, dist1, dist2):
        """Function that calculates the probabilities"""
        return prior * dist1.pdf(X[0]) * dist2.pdf(X[1])
    
    def runModel(self):
        """Funtion that runs the model"""
        #Splitting the data:
        self.X_train, self.X_test, self.y_train, self.y_test = self.splitTrainTest()
        
        self.X0_train = self.X_train[self.y_train == 0]
        self.X1_train = self.X_train[self.y_train == 1]
        
        #Calculating the priors:
        self.prior_y0 = self.X0_train.shape[0] / self.y_train.shape[0]
        self.prior_y1 = self.X1_train.shape[0] / self.y_train.shape[0]
        
        #Fitting the distributions:
        self.dist_X0y0 = self.fitDistribution(self.X0_train[:, 0])
        self.dist_X0y1 = self.fitDistribution(self.X0_train[:, 1])
        
        self.dist_X1y0 = self.fitDistribution(self.X1_train[:, 0])
        self.dist_X1y1 = self.fitDistribution(self.X1_train[:, 1])
        
    def predict(self):
        """Function that predicts the class of the input data"""
        for sample, target in zip(self.X_test, self.y_test):
            
            py0 = self.probability(sample, self.prior_y0, self.dist_X0y0, self.dist_X1y0)
            py1 = self.probability(sample, self.prior_y1, self.dist_X0y1, self.dist_X1y1)
            
            print("P(y=0) | %s) = %.3f" % (sample, py0*100))
            print("P(y=1) | %s) = %.3f" % (sample, py1*100))
            print("Model predicted class {} and the true class was {}".format(np.argmax([py0*100,
                                                                                         py1*100]), target))


#Running the model:
nb = NaiveBayes(X, y)

nb.runModel()

#Predictions:
nb.predict()