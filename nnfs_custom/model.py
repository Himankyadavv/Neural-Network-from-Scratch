class layer_inputs:
    def forward(self, inputs):
        self.outputs = inputs

class Model:
    def __init__(self):
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def forward(self, X):
        self.input_layer.forward(X)
        for layer in self.layers:
            layer.forward(layer.prev.outputs)
        return layer.outputs

    def finalize(self):
        self.input_layer = layer_inputs()
        layer_count = len(self.layers)
        self.trainable_layers = []

        for i in range(layer_count):
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i+1]
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.layers[i+1]
            else:
                self.layers[i].prev = self.layers[i-1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        self.loss.remember_trainable_layers(self.trainable_layers)

    def train(self, X, y, *, epochs=1, print_every=1, validation_data=None):
        self.accuracy.init(y)

        for epoch in range(1, epochs + 1):
            outputs = self.forward(X)
            data_loss, regularization_loss = self.loss.calculate(outputs, y, include_regularization=True)
            loss = data_loss + regularization_loss
            predictions = self.output_layer_activation.predictions(outputs)
            accuracy = self.accuracy.calculate(predictions, y)

            self.backprop(outputs, y)
            self.optimizer.pre_update_params()

            for layer in self.trainable_layers:
                self.optimizer.update_params(layer)
            self.optimizer.post_update_params()

            if not epoch % print_every:
                print(f'epoch: {epoch}, acc: {accuracy:.3f}, loss: {loss:.3f}, lr: {self.optimizer.current_learning_rate}')
        
        if validation_data is not None:
            X_val, y_val = validation_data
            outputs = self.forward(X_val)
            loss = self.loss.calculate(outputs, y_val)
            predictions = self.output_layer_activation.predictions(outputs)
            accuracy = self.accuracy.calculate(predictions, y_val)
            print(f'validation, acc: {accuracy:.3f}, loss: {loss:.3f}')

    def backprop(self, outputs, y):
        self.loss.backprop(outputs, y)
        for layer in reversed(self.layers):
            layer.backprop(layer.next.dinputs)
