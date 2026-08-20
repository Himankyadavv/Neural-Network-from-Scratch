import nnfs
from nnfs.datasets import sine_data
import matplotlib.pyplot as plt
import numpy as np

from nnfs_custom import *

nnfs.init()

X, y = sine_data()

model = Model()
model.add(layer_dense(1, 64))
model.add(Activation_relu())
model.add(layer_dense(64, 1))
model.add(Activation_linear())

model.set(
    loss=Loss_MeanSquaredError(),
    optimizer=Optimizer_ADAM(),
    accuracy=Accuracy_Regression()
)

model.finalize()
model.train(X, y, epochs=1000, print_every=100)
