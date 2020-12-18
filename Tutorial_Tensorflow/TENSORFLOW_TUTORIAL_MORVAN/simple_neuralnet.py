import tensorflow as tf
import numpy as np

def print_element_in_tensor(element):
    # return 1
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        tensor = sess.run(element)
        print("Print out Tensor form: ", tensor)
        print('-------------------------------------')

def add_layer(inputs, input_size, output_size, activation_function=None):
    print("inside add_layer function")

    Weights = tf.Variable(tf.random_normal([input_size,output_size]))
    print("dtype of Weights: ", Weights.dtype)
    biases = tf.Variable(tf.zeros([1,output_size])+0.1)
    print("dtype of biases: ", biases.dtype)

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs

# x_data = np.linspace(-1,1,300)
# print("x_data: ", x_data)
# print("x_data length: ",len(x_data))
# x_data = np.linspace(-1,1,300)[:]
# print("x_data[:]: ", x_data)
# print("x_data[:] length: ",len(x_data))
x_data = np.linspace(-1,1,300)[:,np.newaxis]
x_data = x_data.astype(np.float32)
print("x_data: ", x_data)
print("x_data length: ",len(x_data))

noise = np.random.normal(0,0.05,x_data.shape)
print("x_data.shape: ", x_data.shape)
print("noise.shape: ", noise.shape)

y_data=np.square(x_data)-0.5 + noise
print("y_data: ",y_data)
print("y_data.shape: ",y_data.shape)
print('\n')

x_placeholder = tf.placeholder(tf.float32,[None,1])
print("x_placeholder: ", x_placeholder)
y_placeholder = tf.placeholder(tf.float32,[None,1])
print("y_placeholder: ", y_placeholder)

print("Type of x_data: ", type(x_data))
print("dtype of x_data: ", x_data.dtype)
print('\n')
# layer_1 = add_layer(x_data, 1,10,activation_function=tf.nn.relu)
layer_1 = add_layer(x_placeholder, 1,10,activation_function=tf.nn.relu)
# print("layer_1: ", layer_1)
# print_element_in_tensor(layer_1)

output_layer = add_layer(layer_1,10,1,activation_function= None)
print("output_layer: ", output_layer)
# print_element_in_tensor(output_layer)

# square_difference_array = tf.square(y_data-output_layer)
square_difference_array = tf.square(y_placeholder-output_layer)
# print("square_difference_array: ", square_difference_array)
# print_element_in_tensor(square_difference_array)

sum_all_elem = tf.reduce_sum(square_difference_array, reduction_indices=[1])
print("sum_all_elem: ",sum_all_elem)
# print_element_in_tensor(sum_all_elem)

loss = tf.reduce_mean(sum_all_elem)
# loss = tf.reduce_mean(sum_all_elem)
print("loss: ",loss)
# print_element_in_tensor(loss)

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
print("train_step: ", train_step)
# print_element_in_tensor(train_step)

init = tf.initialize_all_variables()
# init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(2000):
    sess.run(train_step,feed_dict={x_placeholder:x_data, y_placeholder:y_data})
    if i % 50 == 0:
        print("loss: {}".format(sess.run(loss,feed_dict={x_placeholder: x_data, y_placeholder:y_data})))
