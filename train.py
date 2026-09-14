import numpy as np
import matplotlib.pyplot as plt

from src.networks import NeuralNetwork
from src.losses import MSELoss


#make results reproducible
np.random.seed(42)


#creating a simple dataset


#this is a simple regression problem where we want to learn a nonlinear function
X = np.random.randn(200, 2)

#Tthe target variable is a nonlinear combination of the input features
y = (
    2 * X[:, 0]
    - 3 * X[:, 1]
    + 0.5 * np.sin(3 * X[:, 0])
).reshape(-1, 1)



#CREATE MODEL AND LOSS FUNCTION


model = NeuralNetwork()
loss_function = MSELoss()

learning_rate = 0.01
epochs = 1000

losses = []



#TRAINING LOOP


for epoch in range(epochs):

    #FORWARD PASS
    predictions = model.forward(X)

    #CALCULATE LOSS
    loss = loss_function.forward(predictions, y)

    #BACKWARD PASS
    gradient = loss_function.backward()
    model.backward(gradient)

    #UPDATE PARAMETERS
    model.update(learning_rate)

    #Save loss
    losses.append(loss)

    #Print progress
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}")



#FINAL RESULT


print("\nTraining complete!")
print(f"Initial loss: {losses[0]:.6f}")
print(f"Final loss: {losses[-1]:.6f}")



#PLOT LOSS


plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Over Time")
plt.grid(True)

plt.savefig("training_loss.png", dpi=150)
plt.show()
