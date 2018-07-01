import tensorflow as tf

# 定义变量及滑动平均类，变量的初始值为0，指定了变量类型为float32.
# 所需的计算滑动平均的变量必须为实数型
v1 = tf.Variable(0, dtype=tf.float32)
# step变量模拟神经网络的迭代步数，可以用来东爱控制衰减率
step = tf.Variable(0, trainable=False)
# 定义一个华东平均类，初始化时给定了衰减率和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)
# 定义一个更新变量滑动平均的操作。这里给定了一个列表，
# 每次执行操作时，这个列表的变量都会被更新
maintain_averages_op = ema.apply([v1])

# 查看不同迭代中变量取值的变化
with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 使用ema.average(v1)函数获取滑动平均之后的变量值。
    # 初始化之后的滑动平均会被更新为0
    print(sess.run([v1, ema.average(v1)]))

    # 更新变量v1的取值到5
    sess.run(tf.assign(v1, 5))
    # 更新v1的滑动平均值，衰减率为4.4.3节min函数
    # 所以v1的华东平均会被更新为0.1*0+0.9*5=4.5
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新step和v1的取值为10000
    sess.run(tf.assign(step, 10000))
    # 更新v1的值为10
    sess.run(tf.assign(v1, 10))
    # 更新v1的滑动平均值，衰减率为0.99
    # 所以v1的滑动平均会被更新为4.555
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新一次v1的滑动平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

'''
output:
[0.0, 0.0]
[5.0, 4.5]
[10.0, 4.555]
[10.0, 4.60945]
'''