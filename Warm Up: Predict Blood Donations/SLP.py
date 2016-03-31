import gc
import numpy as np
from scipy import optimize

class ANN(object):
    def __init__(self,inputLayerSize,hiddenLayerSize,outputLayerSize):
        #Define Hyperparameters
        gc.collect()
        self.inputLayerSize = inputLayerSize
        self.outputLayerSize = outputLayerSize
        self.hiddenLayerSize = hiddenLayerSize

        # Weights (parameters)
        # self.W1 = np.random.randn(self.inputLayerSize,self.hiddenLayerSize)
        # self.W2 = np.random.randn(self.hiddenLayerSize,self.outputLayerSize)
        self.W1 = np.random.uniform(-0.5,0.5,(self.inputLayerSize,self.hiddenLayerSize))
        self.W2 = np.random.uniform(-0.5,0.5,(self.hiddenLayerSize,self.outputLayerSize))

        try:
            f = open('myfile','r')
            for i in range(self.inputLayerSize):
                for j in range(self.hiddenLayerSize):
                    temp = f.readline()
                    self.W1[i][j] = float(temp)
            for i in range(self.hiddenLayerSize):
                for j in range(self.outputLayerSize):
                    temp = f.readline()
                    self.W2[i][j] = float(temp)
            f.close()
        except Exception, e:
            print("File not Found")

        gc.collect()

    def forward(self, X):
        #Propogate inputs though network
        gc.collect()
        return self.sigmoid(np.dot(self.sigmoid(np.dot(X, self.W1)), self.W2))

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matrix
        gc.collect()
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self,z):
        #Gradient of sigmoid
        gc.collect()
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        gc.collect()
        yHat = self.forward(X)
        return -sum((y*np.log(yHat))+((np.ones((len(yHat),1))-y)*np.log(np.ones((len(yHat),1))-yHat)))/len(y)
        # return 0.5*sum((y-self.forward(X))**2)

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W and W2 for a given X and y:
        gc.collect()

        z2 = np.dot(X, self.W1)
        a2 = self.sigmoid(z2)
        z3 = np.dot(a2, self.W2)
        yHat = self.sigmoid(z3)

        delta3 = ((y - yHat)/(yHat*(np.ones((len(y),1)))))*self.sigmoidPrime(z3)
        # delta3 = np.multiply(-(y-yHat), self.sigmoidPrime(z3))
        dJdW2 = -np.dot(a2.T, delta3)/len(y)

        delta2 = np.dot(delta3,self.W2.T)*self.sigmoidPrime(z2)
        # delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(z2)
        dJdW1 = -np.dot(X.T, delta2)/len(y)

        del z2,a2,z3,delta3,delta2,yHat
        gc.collect()

        return dJdW1, dJdW2

    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        gc.collect()
        return np.concatenate((self.W1.ravel(), self.W2.ravel()))

    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        gc.collect()
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize , self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

        del W1_start , W1_end , W2_end
        gc.collect()

    def computeGradients(self, X, y):
    	gc.collect()
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def callbackF(self, params):
        gc.collect()
        print(self.i)
        self.i = self.i + 1
        self.setParams(params)
        # self.J.append(self.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        gc.collect()
        self.setParams(params)
        return self.costFunction(X, y), self.computeGradients(X,y)

    def fit(self, X, y):
        #Make an internal variable for the callback function:
        gc.collect()
        # self.X = X
        # self.y = y
        self.i = 1
        #Make empty list to store costs:
        # self.J = []

        params0 = self.getParams()

        options = {'maxiter': 200, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackF)
        self.setParams(_res.x)
        self.optimizationResults = _res

        f = open('myfile','w')
        for i in range(self.inputLayerSize):
            for j in range(self.hiddenLayerSize):
                f.write(str(self.W1[i][j]) + '\n')
        for i in range(self.hiddenLayerSize):
            for j in range(self.outputLayerSize):
                f.write(str(self.W2[i][j]) + '\n')
        f.close()

    def predict(self, X):
        gc.collect()
        return self.forward(X)



#     def computeNumericalGradient(N, X, y):
#         gc.collect()
#         paramsInitial = N.getParams()
#         numgrad = np.zeros(paramsInitial.shape)
#         perturb = np.zeros(paramsInitial.shape)
#         e = 1e-4

#         for p in range(len(paramsInitial)):
#             #Set perturbation vector
#             perturb[p] = e
#             N.setParams(paramsInitial + perturb)
#             loss2 = N.costFunction(X, y)

#             N.setParams(paramsInitial - perturb)
#             loss1 = N.costFunction(X, y)

#             #Compute Numerical Gradient
#             numgrad[p] = (loss2 - loss1) / (2*e)

#             #Return the value we changed to zero:
#             perturb[p] = 0

#         #Return Params to original value:
#         N.setParams(paramsInitial)

#         return numgrad



# class trainer(object):
#     def __init__(self, N):
#         #Make Local reference to network:
#         gc.collect()
#         self. = N

#     def callbackF(self, params):
#     	gc.collect()
#         print("I am here")
#         self.setParams(params)
#         #self.J.append(self.costFunction(self.X, self.y))

#     def costFunctionWrapper(self, params, X, y):
#     	gc.collect()
#         self.setParams(params)
#         return self.costFunction(X, y), self.computeGradients(X,y)

#     def train(self, X, y):
# 		#Make an internal variable for the callback function:
# 		gc.collect()
# 		# self.X = X
# 		# self.y = y

# 		#Make empty list to store costs:
# 		#self.J = []

# 		params0 = self.getParams()

# 		options = {'maxiter': 1, 'disp' : True}
# 		_res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackF)
# 		self.setParams(_res.x)
# 		self.optimizationResults = _res

#     def predict(self, X):
#     	gc.collect()
#         return self.forward(X)