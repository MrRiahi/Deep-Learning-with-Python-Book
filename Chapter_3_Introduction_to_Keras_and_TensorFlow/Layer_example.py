import tensorflow as tf


class SimpleDense(tf.keras.layers.Layer):

    def __init__(self, units, activation=None):
        super().__init__()

        self.units = units
        self.activation = activation

    def build(self, input_shape):
        """
        Weight creation takes place in the build() method.
        :param input_shape:
        :return:
        """

        self.W = self.add_weight(shape=(input_shape[-1], self.units), initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,), initializer='zeros',
                                 trainable=True)

    def call(self, inputs):
        """
        Define the forward pass of the SimpleDense layer.
        :param inputs:
        :return:
        """

        y = tf.matmul(inputs, self.W) + self.b

        if self.activation is not None:
            y = self.activation(y)

        return y


dense_layer = SimpleDense(units=32, activation=tf.keras.activations.relu)
input_tensor = tf.ones(shape=(2, 784))
output_tensor = dense_layer(input_tensor)
print(f'The output_tensor shape is {output_tensor.shape}')




