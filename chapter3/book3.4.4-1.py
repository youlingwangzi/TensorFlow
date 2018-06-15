# page55

import tensorflow as tf

w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 使用placeholder作为输入数据存放的地方，维度不一定要定义，但是降低出错的概率
# x = tf.constant([[0.7, 0.9]])

# 只输入一组数据
# x = tf.placeholder(tf.float32, shape=(1, 2), name="input")
# 同时输入多组数据
x = tf.placeholder(tf.float32, shape=(3, 2), name="input")

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 使用初始化函数来初始化变量

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # print(sess.run(y, feed_dict={x: [[0.7, 0.9]]}))
    print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
