import numpy as np

class Activation_relu:
    def predictions(self, outputs):
        return outputs

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0, inputs)
    
    def backprop(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_softmax:
    def predictions(self, outputs):
        return np.argmax(outputs, axis=1)

    def forward(self, inputs):
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.outputs = probabilities
    
    def backprop(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for index, (single_output, single_dvalues) in enumerate(zip(self.outputs, dvalues)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)

class Activation_Sigmoid:
    def predictions(self, outputs):
        return (outputs > 0.5) * 1

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))
    
    def backprop(self, dvalues):
        self.dinputs = dvalues * (1 - self.outputs) * self.outputs

class Activation_linear:
    def predictions(self, outputs):
        return outputs

    def forward(self, inputs):
        self.inputs = inputs 
        self.outputs = inputs

    def backprop(self, dvalues):
        self.dinputs = dvalues.copy()
