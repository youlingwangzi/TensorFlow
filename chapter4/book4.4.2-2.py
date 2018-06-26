# page87 学习率的设置2 正则化函数样例

import tensorflow as tf

weight = tf.constant([[1.0, -2.0], [-3.0, 4.0]])

with tf.Session() as sess:
    # 输出为：(|1|+|-2|+|-3|+|4|)*0.5 = 5
    print(sess.run(tf.contrib.layers.l1_regularizer(.5)(weight)))
    # 输出为：(1^2+(-2)^2+(-3)^2+4^2)/2*0.5 = 7.5
    # 其中tensorflow会将L2的正则化损失值除以2使求导得到的结果更加简洁
    print(sess.run(tf.contrib.layers.l2_regularizer(.5)(weight)))

'''
output:
====================================
5.0
7.5
====================================
'''