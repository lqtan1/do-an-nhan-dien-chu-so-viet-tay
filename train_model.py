import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import idx2numpy
import os

# Load MNIST data from local files
def load_mnist_data():
    X_train = idx2numpy.convert_from_file('train-images.idx3-ubyte')
    y_train = idx2numpy.convert_from_file('train-labels.idx1-ubyte')
    X_test = idx2numpy.convert_from_file('t10k-images.idx3-ubyte')
    y_test = idx2numpy.convert_from_file('t10k-labels.idx1-ubyte')
    
    # Normalize pixel values [0, 255] to [0, 1]
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    # Reshape images to (28, 28, 1) for CNN
    X_train = X_train.reshape((-1, 28, 28, 1))
    X_test = X_test.reshape((-1, 28, 28, 1))
    
    return (X_train, y_train), (X_test, y_test)

def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    print("Loading data...")
    (X_train, y_train), (X_test, y_test) = load_mnist_data()
    
    print("Building model...")
    model = build_model()
    
    print("Training model...")
    model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))
    
    print("Evaluating model...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"\nTest accuracy: {test_acc}")
    
    print("Saving model...")
    model.save('mnist_cnn_model.h5')
    print("Model saved as mnist_cnn_model.h5")
