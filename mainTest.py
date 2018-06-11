import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf

hello = tf.constant("hello world")
sess = tf.Session()

print(sess.run(hello))


def add(param, param1):
    return param + param1


print(add(16, 87))

names = ['yuanxiaokun', 'wangwenfu', 'wumengxinag', 'luojiawei', 123]

for name in names:
    print(name)


def fun(n):
    if n == 1:
        return n
    return n * fun(n - 1)


print(fun(100))


