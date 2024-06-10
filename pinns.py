#!/usr/bin/env python
# coding: utf-8
# %%

# %%
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


# %%
# Definin the neural network model for PINN
class PINN(tf.keras.Model):
    def __init__(self):
        super(PINN, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
        self.hidden2 = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
        self.hidden3 = tf.keras.layers.Dense(20, activation=tf.nn.tanh)
        self.output_layer = tf.keras.layers.Dense(1)
    def call(self, inputs):
        x = self.hidden1(inputs)
        x = self.hidden2(x)
        x = self.hidden3(x)
        return self.output_layer(x)
# Generating random training data within the unit square
def generate_training_data(N):
    x = np.random.rand(N, 1)
    y = np.random.rand(N, 1)
    X = np.hstack((x, y))
    return X
# Computing the loss for the PINN model
def compute_loss(model, X):
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float32)
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(X_tensor)
        u = model(X_tensor)
        u_x = tape.gradient(u, X_tensor)[:, 0]
        u_y = tape.gradient(u, X_tensor)[:, 1]
        u_xx = tape.gradient(u_x, X_tensor)[:, 0]
        u_yy = tape.gradient(u_y, X_tensor)[:, 1]
        f = tf.zeros_like(u)
        residual = u_xx + u_yy - f
    loss_residual = tf.reduce_mean(tf.square(residual))
    return loss_residual
# Training the PINN model
def train_model(model, X, epochs=1000, lr=0.001):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            loss = compute_loss(model, X)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.numpy()}')
# Applying the boundary conditions
def apply_boundary_conditions(model, X):
    boundary_mask = np.logical_or.reduce((X[:, 0] == 0, X[:, 0] == 1, X[:, 1] == 0, X[:, 1] == 1))
    boundary_values = np.zeros_like(boundary_mask, dtype=float)
    boundary_values[X[:, 1] == 1] = 1
    boundary_values[X[:, 1] == 0] = 1
    def boundary_loss():
        boundary_pred = model(tf.convert_to_tensor(X[boundary_mask], dtype=tf.float32))
        return tf.reduce_mean(tf.square(boundary_pred - tf.convert_to_tensor(boundary_values, dtype=tf.float32)))
    return boundary_loss


# %%
# Generating a grid of test points over the unit square
def generate_test_points(num_points):
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    X, Y = np.meshgrid(x, y)
    points = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))
    return X, Y, points
# Computing model predictions at test points
def predict(model, points):
    points_tensor = tf.convert_to_tensor(points, dtype=tf.float32)
    predictions = model(points_tensor)
    return predictions.numpy()
def plot_results(X, Y, predictions, exact):
    plt.figure(figsize=(15, 8))
    # Plotting PINN predictions
    plt.subplot(1, 2, 1)
    plt.contourf(X, Y, predictions.reshape(X.shape), levels=50, cmap='viridis')
    plt.colorbar()
    plt.title("PINN Predictions")
    plt.xlabel("x")
    plt.ylabel("y")
    # Plotting exact solution
    plt.subplot(1, 2, 2)
    plt.contourf(X, Y, exact.reshape(X.shape), levels=50, cmap='viridis')
    plt.colorbar()
    plt.title("Exact Solution (Zero)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()
# Computing errors between exact solution and predictions
def compute_errors(exact, predictions):
    return np.abs(exact - predictions)
# Ploting the error graph
def plot_error(error, X_grid, Y_grid):
    plt.figure(figsize=(8, 6))
    plt.contourf(X_grid, Y_grid, error.reshape(X_grid.shape), levels=50, cmap='viridis')
    plt.colorbar()
    plt.title("Error")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


# %%
N = 1000
X = generate_training_data(N)
pinn_model = PINN()
train_model(pinn_model, X)


# %%
num_points = 100
X_grid, Y_grid, test_points = generate_test_points(num_points)
predictions = predict(pinn_model, test_points)
exact = np.zeros((num_points, num_points))
plot_results(X_grid, Y_grid, predictions.reshape((num_points, num_points)), exact)
errors = compute_errors(exact, predictions.reshape((num_points, num_points)))
plot_error(errors, X_grid, Y_grid)


# %%
# Check boundary values
print("Boundary condition values at x=0, x=1, y=0, and y=1")
boundary_points_x0 = test_points[test_points[:, 0] == 0]
boundary_points_x1 = test_points[test_points[:, 0] == 1]
boundary_points_y0 = test_points[test_points[:, 1] == 0]
boundary_points_y1 = test_points[test_points[:, 1] == 1]
print("Predictions at x=0:", predict(pinn_model, boundary_points_x0))
print("Predictions at x=1:", predict(pinn_model, boundary_points_x1))
print("Predictions at y=0:", predict(pinn_model, boundary_points_y0))
print("Predictions at y=1:", predict(pinn_model, boundary_points_y1))


# %%

