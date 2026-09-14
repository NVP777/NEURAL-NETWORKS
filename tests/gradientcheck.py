import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.layers import Linear


np.random.seed(42)
torch.manual_seed(42)


#small random input
X_np = np.random.randn(4, 3)

#our manual linear layer
layer = Linear(3, 2)

#forward pass
output_np = layer.forward(X_np)

#use a simple loss function
#sum(output^2)
loss_np = np.sum(output_np ** 2)

#manual gradient of the loss with respect to the output
dOutput = 2 * output_np

#manual backward pass
layer.backward(dOutput)

#PYTORCH VERSION


X_torch = torch.tensor(X_np, dtype=torch.float64)

W_torch = torch.tensor(
    layer.W,
    dtype=torch.float64,
    requires_grad=True
)

b_torch = torch.tensor(
    layer.b,
    dtype=torch.float64,
    requires_grad=True
)

output_torch = X_torch @ W_torch + b_torch
loss_torch = torch.sum(output_torch ** 2)

loss_torch.backward()


#comparing gradients


print("Checking gradients...\n")

weight_difference = np.max(
    np.abs(layer.dW - W_torch.grad.detach().numpy())
)

bias_difference = np.max(
    np.abs(layer.db - b_torch.grad.detach().numpy())
)


print("Weight gradient max difference:", weight_difference)
print("Bias gradient max difference:", bias_difference)


tolerance = 1e-7

if weight_difference < tolerance and bias_difference < tolerance:
    print("\nSUCCESS: Manual gradients match PyTorch autograd!")
else:
    print("\nFAILED: Gradients do not match.")