# neural-network-from-scratch

## Overview
This project implements a simple feedforward neural network from scratch using NumPy. It includes:
- Custom implementation of forward and backward propagation
- Support for different activation functions (Tanh, Sigmoid, Softmax)
- L2 regularization and early stopping
- Training with mini-batch gradient descent

## Features
- Multi-layer architecture with customizable hidden layers
- Supports classification tasks with Softmax output
- Implements early stopping for optimal convergence
- Regularization to prevent overfitting
- Training loss visualization

## Requirements
Ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib opencv-python scikit-learn tensorflow
```
Dataset Preparation
The dataset should be preprocessed and split into training and testing sets. Ensure that X_train and y_train are correctly formatted before training.

Usage
Training the Model
Modify the hyperparameters as needed:
```bash
hidden_units = [64, 32]  # Adjust the network architecture
alpha = 0.001  # Learning rate
weights, biases, loss_history = train(X_train, y_train, hidden_units=hidden_units, alpha=alpha)
```
Predicting
Use the trained model to make predictions:
```bash
y_pred = predict(X_test, weights, biases)
```
Visualizing Loss Curve
Plot the loss curve to observe training progress:
```bash
import matplotlib.pyplot as plt
plt.plot(loss_history)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Curve')
plt.show()
```
Customization
Modify activation functions (tanH or logistic)
Adjust regularization (lambda_reg)
Tune learning rate (alpha)
Enable/disable early stopping (early_stopping=True)
Future Improvements
Implement batch normalization
Add support for different optimization techniques (e.g., Adam, RMSprop)
Extend to convolutional layers for image processing




## Other notebooks
This project also holds some labs notebooks  
- MNIST_conv.ipynb
- Autodiff.ipynb
- Layers_and_Models.ipynb
- Segmentation.ipynb


License
This project is open-source and free to use for educational purposes. I did this as part of an assignment for my AI Msc. 
