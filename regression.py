import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read  data
data = pd.read_csv("medical_insurance.csv")

# only using age and what they are charging
x = data["age"].values
y = data["charges"].values

# set parameters
m = 0   # slope
b = 0   # intercept
alpha = 0.0001  # learning rate
epochs = 100  # number of iterations

print("Starting values:")
print("m =", m)
print("b =", b)
print("learning rate =", alpha)
print("epochs =", epochs)

# functions for predictions and costs
def predict(x):
    return m * x + b

def cost(x, y):
    return np.mean((predict(x) - y) ** 2) / 2

# gradient decent 
all_costs = []
shots = {}
print("Type of shots:", type(shots))

for i in range(epochs):
    y_pred = predict(x)
    dm = np.mean((y_pred - y) * x)
    db = np.mean(y_pred - y)

    # change the parameters
    m = m - alpha * dm
    b = b - alpha * db

    current_cost = cost(x, y)
    all_costs.append(current_cost)

    # save some stuff so you can make a line
    if i in [0, 10, 100, 500, epochs - 1]:
        shots[i] = (m, b)

# results get printed
print("\nFinal values:")
print("m =", m)
print("b =", b)
print("Final cost =", all_costs[-1])

# cost vs time plot 
plt.plot(range(epochs), all_costs)
plt.xlabel("Epoch")
plt.ylabel("Cost")
plt.title("Cost over iterations")
plt.show()

# plot data... there's a lot, also the regression lines
plt.scatter(x, y, label="Data", color="blue")

for i, (m_snap, b_snap) in shots.items():
    plt.plot(x, m_snap * x + b_snap, label=f"Epoch {i}")

plt.title("How the line changes during training")
plt.xlabel("Age")
plt.ylabel("Insurance Charges")
plt.legend()
plt.show()
