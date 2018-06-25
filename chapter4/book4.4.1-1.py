# page87 学习率的设置

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

# 定义预测多了和预测少了的成本
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),
                              (y - y_) * loss_more,
                              (y_ - y) * loss_less))
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
        if i % 100 == 0:
            print("learing rate: ", sess.run(learning_rate))
            # print(sess.run(w1))
    # print("Final:\n", sess.run(w1))

'''
output:
=======================================
learing rate:  0.1
learing rate:  0.096
learing rate:  0.09216
learing rate:  0.088473596
learing rate:  0.084934644
learing rate:  0.08153726
learing rate:  0.07827577
learing rate:  0.07514474
learing rate:  0.07213895
learing rate:  0.069253385
learing rate:  0.06648325
learing rate:  0.063823916
learing rate:  0.06127096
learing rate:  0.058820125
learing rate:  0.056467313
learing rate:  0.054208618
learing rate:  0.052040275
learing rate:  0.04995866
learing rate:  0.047960315
learing rate:  0.046041902
learing rate:  0.044200223
learing rate:  0.042432215
learing rate:  0.040734924
learing rate:  0.039105527
learing rate:  0.037541308
learing rate:  0.03603965
learing rate:  0.034598064
learing rate:  0.03321414
learing rate:  0.031885576
learing rate:  0.03061015
learing rate:  0.029385746
learing rate:  0.028210316
learing rate:  0.027081901
learing rate:  0.025998626
learing rate:  0.02495868
learing rate:  0.023960331
learing rate:  0.023001917
learing rate:  0.02208184
learing rate:  0.021198567
learing rate:  0.020350624
learing rate:  0.019536598
learing rate:  0.018755134
learing rate:  0.018004928
learing rate:  0.01728473
learing rate:  0.01659334
learing rate:  0.015929608
learing rate:  0.015292423
learing rate:  0.0146807255
learing rate:  0.014093496
learing rate:  0.013529755
=======================================
'''