import numpy as np
import matplotlib as plt
from sklearn.utils.validation import check_random_state
import sys
import time

class MultiLayerPerceptron (object):
    def func_id (self, x):
        return x
    
    def func_sigmoid (self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def func_relu (self, x):
        return np.maximum(x, 0)
    
    def __init__ (self, nInputNeurons = 12,
                        nHiddenNeurons = 16,
                        nOutputNeurons = 4,
                        weights = None,
                        random_state = 41,
                        iterations = 100,
                        eta = 0.1,
                        *args, **kwargs):

        self.nInputNeurons = nInputNeurons
        self.nHiddenNeurons = nHiddenNeurons
        self.nOutputNeurons = nOutputNeurons
        self.weights = weights
        self.iterations = iterations
        self.eta = eta

        self.random_state_ = check_random_state(random_state)

        self.network = []

        self.inputLayer = np.zeros((self.nInputNeurons + 1, 5))
        self.inputLayer[0] = 1.0
        self.network.append(self.inputLayer)

        if weights:
            W_IH = self.weights[0]
        else:
            W_IH = 2 * self.random_state_.random_sample((self.nHiddenNeurons + 1, self.nInputNeurons + 1)) - 1
        self.network.append(W_IH)

        self.hiddenLayer = np.zeros((self.nHiddenNeurons + 1, 5))
        self.hiddenLayer[0] = 1.0
        self.network.append(self.hiddenLayer)

        if weights:
            W_HO = weights[1]
        else:
            W_HO = 2 * self.random_state_.random_sample((self.nOutputNeurons + 1, self.nHiddenNeurons + 1)) - 1
        self.network.append(W_HO)

        self.outputLayer = np.zeros((self.nOutputNeurons + 1, 5))
        self.outputLayer[0] = 0.0
        self.network.append(self.outputLayer)
        
    def print (self):
        print("MLP - Multi-Layer-Perceptron")
        np.set_printoptions(formatter={"float": lambda x: "{0:0.1f}".format(x)})
        for nn_part in enumerate (self.network):
            print(nn_part)
            print("-----\\/-----")

    def predict (self, x):
        self.network[0][:,2] = x

        self.network[2][1:,0] = np.dot(self.network[1][1:,:], self.network[0][:,0])
        self.network[2][1:,1] = self.func_sigmoid(self.network[2][1:,0])
        self.network[2][1:,2] = self.func_id(self.network[2][1:,1])
        self.network[2][1:,3] = self.network[2][1:,2] * ( 1.0 - self.network[2][1:,2])

        self.network[4][1:,0] = np.dot(self.network[3][1:,:], self.network[2][:,2])
        self.network[4][1:,1] = self.func_sigmoid(self.network[4][1:,0])
        self.network[4][1:,2] = self.func_id(self.network[4][1:,1])
        self.network[4][1:,3] = self.network[4][1:,2] * (1.0  - self.network[4][1:,2])

        return self.network[4][:,2]

    def learn (self, cursor):
        sql_command = "SELECT * FROM trainingExamples"
        cursor.execute(sql_command)
        rows = cursor.fetchall()
        X = []
        Y = []
        for row in rows:
            x = []
            x.append(1)
            for i in range(14):
                if i > 1:
                    x.append(row[i])
            X.append(x)
            direction = row[14]
            if direction == 0:
                y = [0, 1, 0, 0, 0]
            if direction == 1:
                y = [0, 0, 1, 0, 0]
            if direction == 2:
                y = [0, 0, 0, 1, 0]
            if direction == 3:
                y = [0, 0, 0, 0, 1]
            Y.append(y)

        print(str(X) + "...")
        print("...")

        self.errors = []
        print("")
        print("")
        print("STARTED LEARNING...")
        print("")
        print("")
        for iteration in range(self.iterations):
            error = 0.0

            print(iteration)


            for x,y in zip(X, Y):
                yCalc = self.predict(x)

                diff = y - yCalc

                # square with numpy because of vectors in diff
                error += 0.5 * np.sum(diff * diff)

                self.network[4][:,4] = self.network[4][:,3] * diff
                self.network[2][:,4] = self.network[2][:,3] * np.dot(self.network[3][:].T, self.network[4][:,4])

                deltaWjk = self.eta * np.outer(self.network[4][:,4], self.network[2][:,2].T)
                detlaWij = self.eta * np.outer(self.network[2][:,4], self.network[0][:,2].T)

                self.network[1][:,:] += detlaWij
                self.network[3][:,:] += deltaWjk
                
            self.errors.append(error)

        print(self.errors)
        
        print("")
        print("")
        print("LEARNING DONE!!")
        print("")
        print("")



# ----------------------------------------------------------------------------------------------
# W_HI = np.matrix([[0.0,0.0,0.0],[-10.0,20.0,20.0],[30.0,-20.0,-20.0]])
# W_HO = np.matrix([[0.0,0.0,0.0],[-30.0,20.0,20.0]])

# W_HI = np.matrix([[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]])
# W_HO = np.matrix([[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]])

# weights = []
# weights.append(W_HI)
# weights.append(W_HO)

# nn = MultiLayerPerceptron(weights=weights)
# nn.print()

# X = np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]])
# y = np.array([0,1.0,1.0,0])

# print("Predict function:")
# for idx,x in enumerate (X):
#     print("{} {} => {}".format(x,y[idx],nn.predict(x)))