import tensorflow as tf

x = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))

with tf.GradientTape() as tape:
	y = tf.square(x)

gradient = tape.gradient(y, x)

print(f'Input x is {x} \nThe gradient of y respect to x is {gradient}')
