import tensorflow as tf

state = tf.Variable(0,name='counter')
print("state variable: ", state)
print("state name: ",state.name)

one = tf.constant(1)
print("one: ",one)

new_value = tf.add(state,one)
print("new_value tensor: ",new_value)
test = state+one
print(".. still be a tensor: ",test)

update = tf.assign(state,new_value)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    print(sess.run(one))
    print(sess.run(new_value))
    print("----------------------------------")
    for _ in range(3):
        sess.run(update)
        print("sess.run(state): ",sess.run(state))
