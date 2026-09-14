# Project Write-Up

## Approach

When I started working on the task, my objective was to dissect the process of learning for a neural network into separate steps that illustrate the input-output process.

For this purpose, I separated the process into four steps: the Linear layer, activation function, loss function, and the whole network itself. By performing separate steps for the forward and backward passes, it became easier to follow what contributes to what in the process of learning.

The following steps are required to train the network:

1. Initialize inputs and perform forward inference to predict outputs.

2. Calculate the loss function given the predicted and ground-truth values.

3. Compute gradients for all learnable parameters using backward propagation.

4. Update the parameters using the gradient descent algorithm.

5. Repeat the process until the loss converges to a minimum point over epochs.

The loop above, where one epoch equals one full training iteration over all the data points, was illustrative of the general mechanism of minimizing loss.

---

## Backpropagation Mechanism

What I found particularly interesting about the implementation was the concept of propagating gradients backward.

In essence, the forward pass computes certain variables that are later necessary for evaluating gradients. This implies that one cannot discard information from the forward pass when performing the backward propagation step. The easiest example is that the gradient of weights in the Linear layer is computed using the original input matrix.

I think this was a crucial realization that illustrates how neural network frameworks work. After defining the loss, its gradient is equal to the sum of derivatives of all parameters multiplied by their gradients.

This, however, is done automatically by the frameworks, and the backpropagation process is just a matter of applying the chain rule of calculus to each individual operation.

---

## Implementation Details

At the same time, the process of implementing these mathematical operations in code taught me some important details of the implementation.

In particular, I learned that one needs to make sure that the calculated gradient corresponds to the analytical expression of the chosen loss function.

For instance, Mean Squared Error involves a division of the sum of errors by the number of elements in the batch. As a consequence, one cannot ignore this detail when calculating gradients.

Furthermore, I realized that proper account should be taken of the dimensions of the tensors when calculating the gradients for learnable parameters.

The reason for this is that parameters are used repeatedly for every element in the batch. Therefore, the gradients need to be accumulated with the same dimensionality as the parameter matrices themselves.

I think the details mentioned above are illustrative of how one usually approaches the process of building a neural network from scratch.

---

## Validation

In this project, I validated my implementation not only by checking if the training loss converges but also by comparing PyTorch's built-in automatic differentiation.

I think this is a very important step and could not be ignored to ensure that my implementation was correct.

The reason for this is that, most of the time, one would not know for sure whether a neural network is trained correctly because it is possible for it to converge to a local minimum.

At the same time, checking the gradients using an independent algorithm helped me build confidence in my validation procedures.

The results of such tests showed that the gradient of the Linear layer differed from PyTorch's result only due to floating-point precision, while ReLU and MSE tests revealed no differences at all.

---

## Final Thoughts

The most important lesson that I learned from the project was the connection between mathematical concepts and their implementation in code.

It is one thing to know the definitions of weights, matrix multiplication, activation functions, loss functions, their gradients, and derivatives. It is another thing to implement all of those together and see how they complement each other in a deep learning procedure.

As such, the project not only helped me practice mathematics but also made me realize how the concepts that I already knew fitted together to build a statistical model.

In my opinion, the biggest conceptual insight was the connection between forward and backward passes. In essence, the former helps estimate outputs, while the latter estimates how weights contribute to the final error, thus enabling parameter updates.

The project also helped me understand the importance of automatic differentiation and how it simplifies the process of building neural networks.

While it was not hard to implement the mathematical operations in code, doing that for a large-scale neural network would become tedious. From this project, I saw the importance of automatic differentiation and how it is implemented in PyTorch, TensorFlow, and other machine learning frameworks.

Overall, this project was an important exercise for me that demonstrated how deep learning works from scratch. In essence, it is all about using fundamental mathematical concepts to train a model.

Matrix multiplication is used to estimate new data points, activation functions add non-linearity to the model, the loss function allows one to estimate the accuracy of the model, the chain rule of calculus is used to propagate gradients, and gradient descent updates the weights.