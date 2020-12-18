import tensorflow as tf

input_1 = tf.placeholder(tf.float32)
input_2 = tf.placeholder(tf.float32)

output = tf.multiply(input_1,input_2)

sess = tf.Session()
x = sess.run(output,feed_dict={input_1:[3], input_2:[55]})
print(x)
sess.close()

with tf.Session() as sess:
    y = sess.run(output, feed_dict={input_1:[4], input_2:[2]})
    print(y)
