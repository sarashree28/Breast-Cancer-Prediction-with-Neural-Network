# Breast-Cancer-Prediction-with-Neural-Network


This code snippet demonstrates how to build and train a simple neural network for predicting breast cancer using a binary classification approach in PyTorch. Here’s a brief explanation of how this code works and how it applies to breast cancer prediction:

Overview:
The goal of this code is to train a neural network to predict whether a given set of features (such as cell measurements from a biopsy) indicate a malignant (cancerous) or benign (non-cancerous) tumor. The prediction is binary, meaning the model will output either 0 (benign) or 1 (malignant).

Neural Network Architecture:

Input Layer (input_size): The number of features in the dataset (e.g., measurements from the breast cancer dataset).
Hidden Layer (hidden_size): A fully connected layer with 64 neurons and ReLU activation. The hidden layer processes the input features to extract important patterns.
Output Layer (output_size): A single neuron with a Sigmoid activation function. The Sigmoid function outputs a probability between 0 and 1, indicating the likelihood of the tumor being malignant.
Loss Function:

Binary Cross-Entropy Loss (nn.BCELoss): Used as the loss function because this is a binary classification problem. It measures the difference between the predicted probability and the actual label (0 or 1).
Optimizer:

Adam Optimizer (optim.Adam): An optimization algorithm that adjusts the weights of the neural network to minimize the loss. The learning rate is set to 0.001, which controls the step size during weight updates.
Training Loop:

Epochs (num_epochs): The number of times the entire training dataset is passed through the network. Here, it is set to 100.

Forward Pass: The input features (X_train) are passed through the network to produce predictions.

Loss Calculation: The loss is calculated by comparing the predicted values to the actual labels (y_train).

Backward Pass: The gradients are computed using backpropagation, and the optimizer updates the model's weights.

Accuracy Calculation: After every epoch, the model's predictions are compared to the actual labels to calculate accuracy.
