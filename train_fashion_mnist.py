import os
import urllib.request
import gzip
import numpy as np
import nnfs
from nnfs_custom import *

nnfs.init()

def load_fashion_mnist():
    base_url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/'
    files = [
        'train-images-idx3-ubyte.gz',
        'train-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz'
    ]
    
    dataset_dir = 'fashion_mnist'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        
    paths = []
    for file in files:
        filepath = os.path.join(dataset_dir, file)
        if not os.path.exists(filepath):
            print(f'Downloading {file}...')
            urllib.request.urlretrieve(base_url + file, filepath)
        paths.append(filepath)
        

    with gzip.open(paths[0], 'rb') as f:
        X_train = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    with gzip.open(paths[1], 'rb') as f:
        y_train = np.frombuffer(f.read(), np.uint8, offset=8)
    with gzip.open(paths[2], 'rb') as f:
        X_test = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 28*28)
    with gzip.open(paths[3], 'rb') as f:
        y_test = np.frombuffer(f.read(), np.uint8, offset=8)
        
    return X_train, y_train, X_test, y_test

print("Loading Fashion MNIST dataset...")
X, y, X_test, y_test = load_fashion_mnist()


X = (X.astype(np.float32) - 127.5) / 127.5
X_test = (X_test.astype(np.float32) - 127.5) / 127.5

# Shuffle the training data
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]

# We will use a smaller subset of the dataset for training to make it faster
# since the custom Model class trains on the full batch.
# Uncomment these lines to train on the full dataset:
X = X[:1000]
y = y[:1000]
X_test = X_test[:100]
y_test = y_test[:100]

print(f"Training on {X.shape[0]} samples, validating on {X_test.shape[0]} samples.")

model = Model()

# Add layers
# Input shape is 784 (28x28 flattened images)
model.add(layer_dense(784, 128))
model.add(Activation_relu())
model.add(layer_dense(128, 128))
model.add(Activation_relu())
model.add(layer_dense(128, 10))
model.add(Activation_softmax())

# Set loss, optimizer and accuracy objects
model.set(
    loss=Loss_CategoricalCrossEntropy(),
    optimizer=Optimizer_ADAM(learning_rate=0.001, decay=1e-3),
    accuracy=Accuracy_Classification()
)


model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=100, print_every=10)
