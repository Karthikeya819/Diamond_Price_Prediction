import random
import numpy as np
def Normalize(data,columns):
    for column in columns:
        mini = data[column].min()
        maxi = data[column].max()
        data[column] = (data[column] - mini)/(maxi - mini)
    return data
def train_test_split(data,test_size,random_state=None):
    if random_state is not None:
        random.seed(random_state)

    data_indices = list(range(len(data)))
    random.shuffle(data_indices)

    test_size = int(test_size * len(data))
    test_indices = data_indices[:test_size]
    train_indices = data_indices[test_size:]

    train_data = data.iloc[train_indices]
    test_data = data.iloc[test_indices]

    return train_data, test_data
class Logistic_Regression():
    def __init__(self,Learnin_Rate,No_Iterations):
        self.Learnin_Rate = Learnin_Rate
        self.No_Iterations = No_Iterations

    def fit(self,X,Y):

        self.X = X
        self.Y = Y

        self.m,self.n = X.shape

        self.w = np.zeros(self.n)
        self.b = 0

        for i in range(self.No_Iterations):
            self.update_weights()

    def update_weights(self):
        y_cap = 1/(1+np.exp(-(self.X.dot(self.w) + self.b)))

        dw = (1/self.m)*np.dot(self.X.T, (y_cap - self.Y))
        db = (1/self.m)*np.sum(y_cap - self.Y)

        self.w = self.w - self.Learnin_Rate * dw
        self.b = self.b - self.Learnin_Rate * db

    def predict(self,X):
        Y_pred = 1 / (1 + np.exp( - (X.dot(self.w) + self.b ) ))     
        Y_pred = np.where( Y_pred > 0.5, 1, 0)
        return Y_pred
class Linear_Regression():

   def __init__(self,learning_rate,no_of_iterations ):
        self.learning_rate =learning_rate
        self.no_of_iterations =no_of_iterations

   def fit(self,X,Y) :
        self.m, self.n = X.shape
        self.w = np.zeros( self.n )
        self.b = 0
        self.X = X
        self.Y = Y  
        for i in range(self.no_of_iterations) :
            self.update_weights()
              
   def update_weights(self) :
        Y_prediction = self.predict(self.X)
        dw = - ( 2 * (self.X.T).dot(self.Y - Y_prediction)) / self.m
        db = - 2 * np.sum(self.Y - Y_prediction) / self.m 
        self.w = self.w - self.learning_rate * dw
        self.b = self.b - self.learning_rate * db
          
   def predict(self,X) :
        return X.dot(self.w) + self.b
def Linear_Accuracy(result, Y_test):
    RMSE = 0.0
    for i in range(len(result)):
        RMSE += (Y_test.iloc[i] - result.iloc[i]) ** 2 
    RMSE = (RMSE / len(result)) ** 0.5
    VarY =0.0
    mean = Y_test.mean()
    for i in range(len(result)):
        VarY += (Y_test.iloc[i]-mean)**2
    R2 = 1 - ((RMSE**2)/VarY)
    return R2
def R2(result,Y_test):
    RMSE = np.sum((Y_test-result)**2)
    RMSE = (RMSE / len(result)) ** 0.5
    mean = Y_test.mean()
    VarY = np.sum((Y_test-mean)**2)
    return 1 - ((RMSE**2)/VarY)
    

