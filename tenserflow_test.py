import tensorflow as tf

a = tf.constant([1.0, 2.0], name='a')
b = tf.constant([2.0, 3.0], name='b')

result = a + b

# Page 40
print(result)

# Page 41
print(a.graph is tf.get_default_graph())

g1 = tf.Graph()
with g1.as_default():
    tf.get_variable(
        'V', shape=[1], initializer=tf.zeros_initializer)

g2 = tf.Graph()
with g2.as_default():
    tf.get_variable(
        'V', shape=[1], initializer=tf.ones_initializer)

with tf.Session(graph=g1) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("V")))

with tf.Session(graph=g2) as sess:
    tf.global_variables_initializer().run()
    with tf.variable_scope("", reuse=True):
        print(sess.run(tf.get_variable("V")))
