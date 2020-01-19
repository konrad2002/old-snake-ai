import numpy as np
import matplotlib as plt

class MultiLayerPerceptron (object):
    def func_id (self, x):
        return x
    
    def func_sigmoid (self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def func_relu (self, x):
        return np.maximum(x, 0)
    
    def __init__ (self, nInputNeurons = 12, nHiddenNeurons = 16, nOutputNeurons = 4, weights = None, *args, **kwargs):
        self.nInputNeurons = nInputNeurons
        self.nHiddenNeurons = nHiddenNeurons
        self.nOutputNeurons = nOutputNeurons
        self.weights = weights

        self.network = []

        self.inputLayer = np.zeros((self.nInputNeurons + 1, 1))
        self.inputLayer[0] = 1.0
        self.network.append(self.inputLayer)

        if weights:
            W_HI = self.weights[0]
        else:
            W_HI = np.zeros(self.nHiddenNeurons + 1, self.nInputNeurons + 1)
        self.network.append(W_HI)

        self.hiddenLayer = np.zeros((self.nHiddenNeurons + 1, 3))
        self.hiddenLayer[0] = 1.0
        self.network.append(self.hiddenLayer)

        if weights:
            W_HO = weights[1]
        else:
            W_HO = np.zeros(self.nOutputNeurons + 1, self.nHiddenNeurons + 1)
        self.network.append(W_HO)

        self.outputLayer = np.zeros((self.nOutputNeurons + 1, 3))
        self.outputLayer[0] = 0.0
        self.network.append(self.outputLayer)
        
    def print (self):
        print("MLP - Multi-Layer-Perceptron")
        np.set_printoptions(formatter={"float": lambda x: "{0:0.1f}".format(x)})
        for nn_part in enumerate (self.network):
            print(nn_part)
            print("-----\\/-----")

    def predict (self, x):
        self.network[0][:,0] = x

        self.network[2][1:,0] = np.dot(self.network[1][1:,:], self.network[0][:,0])
        self.network[2][1:,1] = self.func_sigmoid(self.network[2][1:,0])
        self.network[2][1:,2] = self.func_id(self.network[2][1:,1])

        self.network[4][1:,0] = np.dot(self.network[3][1:,:], self.network[2][:,2])
        self.network[4][1:,1] = self.func_sigmoid(self.network[4][1:,0])
        self.network[4][1:,2] = self.func_id(self.network[4][1:,1])

        return self.network[4][1:,2]


# ----------------------------------------------------------------------------------------------
W_HI = np.matrix([[0.0,0.0,0.0],[-10.0,20.0,20.0],[30.0,-20.0,-20.0]])
W_HO = np.matrix([[0.0,0.0,0.0],[-30.0,20.0,20.0]])

W_HI = np.matrix([[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]])
W_HO = np.matrix([[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5],[1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5,1.5]])

weights = []
weights.append(W_HI)
weights.append(W_HO)

nn = MultiLayerPerceptron(weights=weights)
nn.print()

X = np.array([[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]])
y = np.array([0,1.0,1.0,0])

print("Predict function:")
for idx,x in enumerate (X):
    print("{} {} => {}".format(x,y[idx],nn.predict(x)))