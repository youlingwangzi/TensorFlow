# page80 一个完成的训练程序展示损失函数对模型训练的结果的影响

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

train_step = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

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
        end = (i*batch_size) % 128 + batch_size
        #end = min(start+batch_size, dataset_size)
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            print(sess.run(w1))
    print("Final:\n", sess.run(w1))

'''
C:\Users\yuanx\AppData\Local\Programs\Python\Python36\python.exe C:/Users/yuanx/Documents/Code/Python/TensorFlow/chapter4/book4.2.2-2.py
2018-06-23 01:37:29.915381: I T:\src\github\tensorflow\tensorflow\core\platform\cpu_feature_guard.cc:140] Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2
[[-0.81131774]
 [ 1.4845992 ]]
[[-0.8106414]
 [ 1.485216 ]]
[[-0.80985266]
 [ 1.4859272 ]]
[[-0.80899984]
 [ 1.4866718 ]]
[[-0.80812407]
 [ 1.4874339 ]]
Final:
 [[-0.8072287]
 [ 1.4882191]]

Process finished with exit code 0

输出与书上预期不太一致。为什么？
'''
