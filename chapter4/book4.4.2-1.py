# page89 过拟合问题

import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 回归问题一般置有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义一个简单加权和的单层神经网络前向传播过程
w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))
y = tf.matmul(x, w1)

# 定义损失函数为 正则化的损失函数 P88
loss_less = 10
loss_more = 1
''' 未正则化的损失函数
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                              (y - y_) * loss_more,
                              (y_ - y) * loss_less))
'''
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                              (y - y_) * loss_more,
                              (y_ - y) * loss_less)) + tf.contrib.layers.l2_regularizer(0.5)(w1)

global_step = tf.Variable(0)
# 通过 exponential_decey 函数生成学习率
learning_rate = tf.train.exponential_decay(
    0.1, global_step, 100, 0.96, staircase=True
)

# 使用指数衰减的学习率。在minimize函数中传入global_step将自动更新
# global_step 参数，使学习率相应更新。
learning_step = tf.train.GradientDescentOptimizer(learning_rate)\
    .minimize(loss, global_step=global_step)

# 通过随机数生成一个模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1, x2) in X]

# 训练神经网络
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 5000
    for i in range(STEPS):
        start = (i*batch_size) % dataset_size
        # end = (i*batch_size) % 128 + batch_size
        end = min(start+batch_size, dataset_size)
        sess.run(learning_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            print(sess.run(w1))
    print("Final:\n", sess.run(w1))

'''
output:
=======================================
[[0.77205235]
 [3.6075373 ]]
[[1.6326492]
 [2.04698  ]]
[[0.88967645]
 [0.9673948 ]]
[[1.6437817]
 [2.0231419]]
[[1.1279787]
 [1.2870868]]
Final:
 [[0.93396765]
 [1.0631682 ]]
========================================
'''