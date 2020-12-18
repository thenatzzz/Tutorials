import tensorflow as tf

matrix1 = tf.constant([[3,3]])
print(matrix1)
matrix2 = tf.constant([[2],[2]])
print(matrix2)
print('\n')

product = tf.matmul(matrix1, matrix2)

#### no need to initialize_all_variables because we do not use any tf.Variable
# start_execution = tf.initialize_all_variables()

############      METHOD 1 to run session   ###############################
sess = tf.Session()
print(sess.run(product))
sess.close()

 ############     METHOD 2 to run session   ##############################
with tf.Session() as sess:
    # sess.run(start_execution)
    print(sess.run(product))
