# page78 实现了softmax回归之后的经典损失函数

import tensorflow as tf

from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))

# 使用placeholder作为输入数据存放的地方，维度不一定要定义，但是降低出错的概率
x = tf.placeholder(tf.float32, shape=(None, 2), name="X-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义神经网络前向传播过程-未加入偏置项和非线性激活函数
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)

# 加入偏置项和非线性激活函数的前向传播过程
biases1 = -0.5
biases2 = -0.1
a = tf.nn.relu(tf.matmul(x, w1) + biases1)
y = tf.nn.relu(tf.matmul(a, w2) + biases2)

# 损失函数，反向传播算法，这里使用交叉熵
# tf.clip_by_value()函数可以将张量中的数值限定在一定的范围内，避免数值运算错误
# 注意区别 * 乘法和 tf.matmul()的矩阵乘法
# tf.clip_by_value函数可以控制矩阵值在一定的安全范围内
y = tf.sigmoid(y)
# cross_entropy = -tf.reduce_mean(
# 替换为使用softmax回归之后的交叉熵
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
    + (1-y)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))
)

# 随机数模拟数据集
rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)

# 定义规则样本，使用0表示负样本，使用1表示正样本
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

learning_rate = 0.001
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# 使用初始化函数来初始化变量

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 训练之前的参数值
    print(sess.run(w1))
    print(sess.run(w2))

    STEPS = 5000
    for i in range(STEPS):
        # 每次选区batch_size个样本进行训练
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # 通过选取的样本训练神经网络并更新参数
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            # 每个一段时间计算所有数据上的交叉熵并输出
            total_croee_entropy = sess.run(
                cross_entropy, feed_dict={x: X, y_: Y}
            )
            print("%d 次训练之后，所有数据的交叉熵是 %g" %(i, total_croee_entropy))

    # 训练之后的参数集
    print(sess.run(w1))
    print(sess.run(w2))

'''
the output data:
没有偏置项和激活函数
================================================
[[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
[[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]
0 次训练之后，所有数据的交叉熵是 0.314006
1000 次训练之后，所有数据的交叉熵是 0.0684551
2000 次训练之后，所有数据的交叉熵是 0.033715
3000 次训练之后，所有数据的交叉熵是 0.020558
4000 次训练之后，所有数据的交叉熵是 0.0136867
[[-2.548655   3.0793087  2.8951712]
 [-4.1112747  1.6259071  3.3972702]]
[[-2.3230937]
 [ 3.3011687]
 [ 2.4632082]]
=================================================
有偏执项和激活函数
=================================================
[[-0.8113182   1.4845988   0.06532937]
 [-2.4427042   0.0992484   0.5912243 ]]
[[-0.8113182 ]
 [ 1.4845988 ]
 [ 0.06532937]]
0 次训练之后，所有数据的交叉熵是 0.655775
1000 次训练之后，所有数据的交叉熵是 0.226569
2000 次训练之后，所有数据的交叉熵是 0.119352
3000 次训练之后，所有数据的交叉熵是 0.0823214
4000 次训练之后，所有数据的交叉熵是 0.0615102
[[-0.8113182  3.6808345  3.3656993]
 [-2.4427042  2.718267   4.0405617]]
[[-0.8113182]
 [ 3.8388839]
 [ 3.288155 ]]
=================================================

question：
加入偏置项和激活函数后，交叉熵反而变大了，这是为啥？有待进一步确认原因。
'''
