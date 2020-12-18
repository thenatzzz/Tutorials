import tensorflow as tf
import numpy as np

#### Creating some data
x_data = np.random.rand(100).astype(np.float32)
print("x_data: ",x_data)
print("Length of x_data: ",len(x_data))
y_data = x_data*0.1 + 0.3
print("y_data: ",y_data)

print("1st x_data:",x_data[0])
print("1st y_data:",y_data[0])
print('\n')
######## Creating Tensorflow Structure Start
# sess = tf.InteractiveSession()

Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
print("var of Weight: {}".format(Weights))
biases = tf.Variable(tf.zeros([1]))
print("var of Bias: %s"%(biases))

y = Weights*x_data + biases
print("y: ", y)

loss = tf.reduce_mean(tf.square(y-y_data))
print("loss: {}".format(loss))
optimizer = tf.train.GradientDescentOptimizer(0.5)
print("optimizer: ", optimizer)
train = optimizer.minimize(loss)
print("train: ", train)

print('\n')

init = tf.initialize_all_variables() # MUST RUN THIS FUNCTION !!!

####### Creating Tensorflow Structure End
sess = tf.Session()
print("sess: ",sess)
sess.run(init)

''' # if we want to print detail in Tensorflow object we need sess.run()'''
print("var of Weight: {}".format(sess.run(Weights)))
print("var of Bias: %s"%(sess.run(biases)))
print("y: ", sess.run(y))
print("loss: {}".format(sess.run(loss)))
print("train: {}".format(sess.run(train)))
print('\n')

for step in range(301):
    sess.run(train)
    if step % 20 == 0:
        print("step: {} , Weights: {}, biases: {} ".format(step,sess.run(Weights),sess.run(biases)))
