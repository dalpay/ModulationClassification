import numpy as np


class FeedforwardNetwork(object):

    def __init__ (self, model_path, num_layers):
        
        self.num_layers = num_layers
        self.weights = list()
        self.biases = list()

        for i in range(self.num_layers):

            weights = np.genfromtxt(model_path + '_weights' + str(i) + '.gz')
            biases = np.genfromtxt(model_path + '_biases' + str(i) + '.gz')
            self.weights.append(weights)
            self.biases.append(biases)

    def predict(self, vec):

        activations = vec.reshape(1, -1)
        
        for i in range(self.num_layers):
        
            activations = np.dot(activations, self.weights[i])
            activations += self.biases[i]

            if (i != self.num_layers - 1):
                activations = self.relu(activations)

        activations = self.softmax(activations)
        
        return np.argmax(activations)

    def relu(self, vec):
        
        np.clip(vec, 0, np.finfo(vec.dtype).max, out=vec)

        return vec

    def softmax(self, vec):

        tmp = vec - vec.max(axis=1)[:, np.newaxis]
        np.exp(tmp, out=vec)
        vec /= vec.sum(axis=1)[:, np.newaxis]

        return vec
    
