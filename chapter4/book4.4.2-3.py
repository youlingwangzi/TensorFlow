# Page89 使用了集合来保存实体的程序示例

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

data = []
label1 = []
np.random.seed(0)

# 以原点为圆心，半径为1的圆把散点划分成红蓝两部分，并加入随机噪音。
for i in range(150):
    x1 = np.random.uniform(-1, 1)
    x2 = np.random.uniform(0, 2)
    if x1 ** 2 + x2 ** 2 <= 1:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label1.append(0)
    else:
        data.append([np.random.normal(x1, 0.1), np.random.normal(x2, 0.1)])
        label1.append(1)

data = np.hstack(data).reshape(-1, 2)
label = np.hstack(label1).reshape(-1, 1)
# 画出数据图
plt.scatter(data[:, 0], data[:, 1], c=label1,
            cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.show()


# 获取一层神经网络边上的权重，并将这个权重的L2正则化损失加入名为“losses”的集合中
def get_weight(shape, lambda1):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    # add_to_collection 函数将这个新生成的变量的L2正则化损失加入集合
    # 这个函数的第一个参数“losses”是集合的名字，第二个参数是要加入这个集合的内容
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lambda1)(var))
    # 返回生成的变量
    return var


# 两个输入节点
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
# 回归问题一般置有一个输出节点
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")
sample_size = len(data)

batch_size = 8

# 定义每一层网络中节点的个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 这个变量维护前向传播时最深层的节点，开始时就是输入层
cur_layer = x
# 当前层的节点个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接神经网络结构
for i in range(1, n_layers):
    # layyer_dimension[i] 为下一层节点个数
    out_dimension = layer_dimension[i]
    # 生成当前层中权重的变量，并将这个变量的L2正则化损失函数加入计算图上的集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))
    # 使用ReLU激活函数
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层节点个数更新为当前节点个数
    in_dimension = layer_dimension[i]

y = cur_layer

# 损失函数的定义。
mse_loss = tf.reduce_sum(tf.pow(y_ - y, 2)) / sample_size
tf.add_to_collection('losses', mse_loss)
loss = tf.add_n(tf.get_collection('losses'))

# 定义训练的目标函数mse_loss，训练次数及训练模型
train_op = tf.train.AdamOptimizer(0.001).minimize(mse_loss)
TRAINING_STEPS = 40000

# 训练不加入正则的损失函数数据，使用mes_loss
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(TRAINING_STEPS):
        sess.run(train_op, feed_dict={x: data, y_: label})
        if i % 2000 == 0:
            print("After %d steps, mse_loss: %f" % (i, sess.run(mse_loss, feed_dict={x: data, y_: label})))

    # 画出训练后的分割曲线
    xx, yy = np.mgrid[-1.2:1.2:.01, -0.2:2.2:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x: grid})
    probs = probs.reshape(xx.shape)

plt.scatter(data[:, 0], data[:, 1], c=label1,
            cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
plt.show()


# 定义训练的目标函数loss，训练次数及训练模型
train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
TRAINING_STEPS = 40000

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(TRAINING_STEPS):
        sess.run(train_op, feed_dict={x: data, y_: label})
        if i % 2000 == 0:
            print("After %d steps, loss: %f" % (i, sess.run(loss, feed_dict={x: data, y_: label})))

    # 画出训练后的分割曲线
    xx, yy = np.mgrid[-1:1:.01, 0:2:.01]
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = sess.run(y, feed_dict={x:grid})
    probs = probs.reshape(xx.shape)

plt.scatter(data[:,0], data[:,1], c=label1,
           cmap="RdBu", vmin=-.2, vmax=1.2, edgecolor="white")
plt.contour(xx, yy, probs, levels=[.5], cmap="Greys", vmin=0, vmax=.1)
plt.show()


'''
output:
=================================================================
After 0 steps, mse_loss: 660.295410
After 2000 steps, mse_loss: 0.367207
After 4000 steps, mse_loss: 0.347266
After 6000 steps, mse_loss: 0.271647
After 8000 steps, mse_loss: 0.253126
After 10000 steps, mse_loss: 0.061860
After 12000 steps, mse_loss: 0.060554
After 14000 steps, mse_loss: 0.060352
After 16000 steps, mse_loss: 0.059600
After 18000 steps, mse_loss: 0.059438
After 20000 steps, mse_loss: 0.059300
After 22000 steps, mse_loss: 0.059244
After 24000 steps, mse_loss: 0.059229
After 26000 steps, mse_loss: 0.059233
After 28000 steps, mse_loss: 0.059226
After 30000 steps, mse_loss: 0.059217
After 32000 steps, mse_loss: 0.059218
After 34000 steps, mse_loss: 0.059220
After 36000 steps, mse_loss: 0.059221
After 38000 steps, mse_loss: 0.059234
After 0 steps, loss: 7.240983
After 2000 steps, loss: 0.632736
After 4000 steps, loss: 0.610672
After 6000 steps, loss: 0.586658
After 8000 steps, loss: 0.049882
After 10000 steps, loss: 0.039175
After 12000 steps, loss: 0.030192
After 14000 steps, loss: 0.024813
After 16000 steps, loss: 0.022639
After 18000 steps, loss: 0.021775
After 20000 steps, loss: 0.021616
After 22000 steps, loss: 0.021559
After 24000 steps, loss: 0.021556
After 26000 steps, loss: 0.021566
After 28000 steps, loss: 0.021502
After 30000 steps, loss: 0.021525
After 32000 steps, loss: 0.021545
After 34000 steps, loss: 0.021482
After 36000 steps, loss: 0.021486
After 38000 steps, loss: 0.021499
PyDev console: starting.
Python 3.6.5 (v3.6.5:f59c0932b4, Mar 28 2018, 17:00:18) [MSC v.1900 64 bit (AMD64)] on win32
Process finished with exit code -1

'''