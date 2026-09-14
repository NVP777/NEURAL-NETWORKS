DEBUGGING MISTAKES:
During implementation and checking, I made these 3 mistakaes:
1) MSE Normalisation factor: The MSE gradient accounts for the averaging used in the losses, but I initially fdorgot the division by N. As a result the gradient had the corrrect direction but incorrect magnitude, due to which parameter updates were too lalrge and didnt match with the pytorch gradient.
2) Wrong axis during bias gradient calculation: The correct bias gradient is 'self.db = np.sum(dZ, axis=0, keepdims=True)'. I initially summed it along the wrong direction, using axis=1. Instead of summming the columns, it summed up the rows. This is because each bias belongs to an output feature/neuron and is shared across all examples in the batch, so we must accumulate gradients across the batch dimension.

Manual Neural Network from Scratch
Description
The purpose of this project was to implement and train a small neural network from scratch using NumPy. The goal was to learn the full pipeline of training a neural network, including forward propagation, loss computation, backpropagation, gradient descent, and finally validating the gradients.
The implemented network does not use an automatic differentiation library for training. Instead, the gradients of each operation were computed manually.
The implementation was validated by comparing the manually implemented gradients to PyTorch's automatic differentiation.
Network
The implemented network had the following structure:

Input (2 features)
↓
Linear Layer (2 → 8)
↓
ReLU Activation
↓
Linear Layer (8 → 1)
↓
Prediction
↓
MSE Loss
Project Layout
manual-neural-network/
│
├── src/
│   ├── layers.py
│   ├── activations.py
│   ├── losses.py
│   └── network.py
│
├── tests/
│   ├── gradientcheck.py
│   └── activation_loss_check.py
│
├── train.py
├── training_loss.png
├── requirements.txt
├── README.md
└── WRITEUP.md
Implementation
Linear Layer
The linear layer performs the following operation:

Z = XW + b

where:
$X$ is the input
$W$ contains the weights
$b$ contains the biases
$Z$ is the output
The gradients during backpropagation are computed manually as:

dW = X^T dZ

db = sum of all dZ

dX = dZW^T

Notice that we have to save the original X input during the forward pass to compute the weight gradient.
ReLU
The ReLU activation function is defined as:

operatorname{ReLU}(x) = max(0, x)

With the derivative:

frac{d}{dx}operatorname{ReLU}(x) = begin{cases} 1, & x > 0  0, & x leq 0 end{cases}

Notice that during the backward pass, we only propagate the gradients through the neurons that had a positive value during the forward pass.
Mean Squared Error
We want to minimize the mean squared error loss:

L = frac{1}{N}sum_{i=1}^{N}(hat{y}_i-y_i)^2

The gradient of this loss with respect to the prediction is simply:

frac{partial L}{partial hat{y}} = frac{2}{N}(hat{y}-y)

This gradient is then backpropagated through the network.
Training
The network was trained on a synthetic dataset that had 2 features with a non-linear target function.
During each epoch of training, the following steps are performed:

Forward Pass
↓
Calculate Prediction
↓
Calculate MSE Loss
↓
Backward Pass
↓
Calculate Gradients
↓
Update Weights and Biases
↓
Repeat
The parameters of the network are updated using gradient descent:

theta = theta - eta nabla_theta L

where $eta$ is the learning rate.
Once the network started to train, the loss decreased, which indicated that the manually implemented neural network was able to learn the pattern in this data.
Gradient Validation
The manually implemented gradients were validated against PyTorch's automatic differentiation.
The following components underwent validation:

Linear layer gradients
ReLU gradients
Mean Squared Error gradients
Linear Layer
The maximum difference between the manually calculated and PyTorch weight gradients was:

2.7755575615628914e-17
The bias gradients were the same.

ReLU
The manual ReLU gradients matched PyTorch exactly for the tested inputs.

MSE Loss
The manually implemented MSE loss and gradients were both exactly the same as PyTorch for the tested inputs.
Debugging and Implementation Pitfalls
During implementation and checking, I made these 2 mistakaes:
1) MSE Normalisation factor: The MSE gradient accounts for the averaging used in the losses, but I initially fdorgot the division by N. As a result the gradient had the corrrect direction but incorrect magnitude, due to which parameter updates were too lalrge and didnt match with the pytorch gradient.
2) Wrong axis during bias gradient calculation: The correct bias gradient is 'self.db = np.sum(dZ, axis=0, keepdims=True)'. I initially summed it along the wrong direction, using axis=1. Instead of summming the columns, it summed up the rows. This is because each bias belongs to an output feature/neuron and is shared across all examples in the batch, so we must accumulate gradients across the batch dimension.


How to Run
Install dependencies
pip install -r requirements.txt
Train the network
python train.py
This will train the network and generate:

training_loss.png
Run the Linear gradient check
python tests/gradientcheck.py
Run the ReLU and MSE checks
python tests/activation_loss_check.py
What I Learned
By working on this project, I was able to make connections between the mathematical expressions and their implementation in code.
The main concepts I was able to explore were:

Forward propagation
Linear transformation
Activation function
Mean Squared Error
Chain rule
Backpropagation
Gradient descent
Batch gradients
Gradient validation
The most important lesson I learned through this project was that the training of a neural network is an iterative procedure that follows these steps:

Predict
↓
Measure Error
↓
Calculate Gradients
↓
Update Parameters
↓
Repeat
By manually implementing these steps, I was able to really understand how these concepts connected to each other.


