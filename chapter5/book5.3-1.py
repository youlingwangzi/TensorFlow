# 变量的管理

import tensorflow as tf

# 两种定义常量的方式，两种常量定义方式最大的区别在于，前者的变量名称是一个必填的参数，后者是一个选填的参数
# 前者会在创建前检查是否已有同名的变量
v = tf.get_variable("V", shape=[1],
                    initializer=tf.constant_initializer(1.0))
v2 = tf.Variable(tf.constant(1.0, shape=[1], name="v2"))


# 在命名空间中创建v变量。
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1], initializer=tf.constant_initializer(1.0))

# 因为已经在命名空间foo中创建了v变量，所以以下代码会报错
# with tf.variable_scope("foo"):
# v = tf.get_variable("v", [1])

# 如果在函数中加上reuse=True的参数，那么函数会直接获取已经定义的v变量
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])

# 结果可以看到两个变量时一样的，输出结果为 True
print(v == v1)

# 因为将reuse=True设置后，tf.variable_scope只会在已有的变量中去寻找，
# 此时bar空间中的v变量还没有定义，因此代码会报错
# with tf.variable_scope("bar", reuse=True):
#     v2 = tf.get_variable("v", [1])

# 在bar空间中定义变量v
with tf.variable_scope("bar"):
    v2 = tf.get_variable("v", [1])
# 将bar中的v和foo中的v变量做比较，可以发现，他们是两个不同的变量
print(v == v2)

'''
output
============================================================
True
False
'''
