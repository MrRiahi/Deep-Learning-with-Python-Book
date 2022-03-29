import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


def model(inputs, W, b):
    """
    This is a linear classifier model.
    :param inputs:
    :param W:
    :param b:
    :return:
    """

    outputs = tf.matmul(inputs, W) + b
    return outputs


def square_loss(true_labels, predictions):
    """
    This function gets the loss of model based on predictions and labels.
    :param true_labels:
    :param predictions:
    :return:
    """

    loss_per_sample = tf.square(predictions - true_labels)
    return tf.reduce_mean(loss_per_sample)


def fit(X, y, W, b, learning_rate=0.1):
    """
    Fit the model.
    :param X:
    :param y:
    :param W:
    :param b:
    :param learning_rate:
    :return:
    """

    with tf.GradientTape() as tape:
        predictions = model(inputs=X, W=W, b=b)
        loss = square_loss(true_labels=y, predictions=predictions)

    gradients_dW, gradients_db = tape.gradient(target=loss, sources=[W, b])  # Get gradients
    W.assign_sub(gradients_dW * learning_rate)  # Update W
    b.assign_sub(gradients_db * learning_rate)  # Update b

    return loss, W, b


# Generate random data
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(mean=[0, 3], cov=[[1, 0.5], [0.5, 1]],
                                                 size=num_samples_per_class)
negative_labels = np.zeros(shape=(num_samples_per_class, 1))

positive_samples = np.random.multivariate_normal(mean=[3, 0], cov=[[1, 0.5], [0.5, 1]],
                                                 size=num_samples_per_class)
positive_labels = np.ones(shape=(num_samples_per_class, 1))

X = np.vstack((negative_samples, positive_samples)).astype('float32')
y = np.vstack((negative_labels, positive_labels)).astype('float32')

# Initialize the model parameters
input_dims = 2
output_dims = 1
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dims, output_dims)))
b = tf.Variable(initial_value=tf.zeros(shape=(output_dims,)))

# Train model
epochs = 50
loss = np.inf

for i_epoch in range(epochs):
    loss, W, b = fit(X=X, y=y, W=W, b=b)

    print(f'Loss at step {i_epoch}: {loss:.4f}')

# Get predictions
predictions = model(inputs=X, W=W, b=b)

# Plot results
x = np.linspace(-1, 4, 100)

weights = W.numpy()
bias = b.numpy()
hyper_line = -weights[0] / weights[1] * x + (0.5 - bias) / weights[1]
plt.plot(x, hyper_line, '-r')
plt.scatter(X[:, 0], X[:, 1], c=predictions[:, 0] >= 0.5)
plt.show()
