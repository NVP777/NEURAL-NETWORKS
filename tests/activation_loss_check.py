import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.activations import ReLU
from src.losses import MSELoss


np.random.seed(42)
torch.manual_seed(42)


print("ReLU GRADIENT CHECK")


#ReLU CHECK


X_np = np.array([
    [-2.0, -0.5, 1.0],
    [3.0, 0.0, -4.0]
])

relu = ReLU()

#output of our ReLU
output_np = relu.forward(X_np)

#upstream gradient from the next layer 
upstream_gradient = np.ones_like(output_np)

#oiur manual backward pass
manual_gradient = relu.backward(upstream_gradient)


#pyTorch version
X_torch = torch.tensor(
    X_np,
    dtype=torch.float64,
    requires_grad=True
)

output_torch = torch.relu(X_torch)

#Same scalar loss: sum of outputs
loss_torch = torch.sum(output_torch)

loss_torch.backward()

torch_gradient = X_torch.grad.detach().numpy()


difference = np.max(
    np.abs(manual_gradient - torch_gradient)
)

print("\nManual gradient:")
print(manual_gradient)

print("\nPyTorch gradient:")
print(torch_gradient)

print("\nMaximum difference:", difference)

if difference < 1e-8:
    print("SUCCESS: ReLU gradients match PyTorch!")
else:
    print("FAILED: ReLU gradients do not match.")


#MSE LOSS CHECK


print("MSE LOSS GRADIENT CHECK")



prediction_np = np.array([
    [1.5],
    [2.0],
    [4.5],
    [3.0]
])

target_np = np.array([
    [2.0],
    [1.0],
    [5.0],
    [4.0]
])


#our manual MSE loss and gradient
mse = MSELoss()

loss_np = mse.forward(prediction_np, target_np)
manual_gradient = mse.backward()


#pyTorch MSE
prediction_torch = torch.tensor(
    prediction_np,
    dtype=torch.float64,
    requires_grad=True
)

target_torch = torch.tensor(
    target_np,
    dtype=torch.float64
)

loss_torch = torch.mean(
    (prediction_torch - target_torch) ** 2
)

loss_torch.backward()

torch_gradient = prediction_torch.grad.detach().numpy()


difference = np.max(
    np.abs(manual_gradient - torch_gradient)
)


print("\nManual MSE loss:", loss_np)
print("PyTorch MSE loss:", loss_torch.item())

print("\nManual gradient:")
print(manual_gradient)

print("\nPyTorch gradient:")
print(torch_gradient)

print("\nMaximum difference:", difference)

if difference < 1e-8:
    print("SUCCESS: MSE gradients match PyTorch!")
else:
    print("FAILED: MSE gradients do not match.")