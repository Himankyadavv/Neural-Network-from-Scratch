import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
import numpy as np

from nnfs_custom import *

nnfs.init()

X, y = spiral_data(samples=100, classes=2)
X_test, y_test = spiral_data(samples=100, classes=2)

y = y.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

model = Model()

model.add(layer_dense(2, 64, weight_regularizer_l2=5e-4, bias_regularizer_l2=5e-4))
model.add(Activation_relu())
model.add(layer_dense(64, 1))
model.add(Activation_Sigmoid())

model.set(
    loss=Loss_BinaryCrossEntropy(),
    optimizer=Optimizer_ADAM(decay=5e-7),
    accuracy=Accuracy_Classification()
)

model.finalize()
model.train(X, y, validation_data=(X_test, y_test), epochs=1000, print_every=100)
