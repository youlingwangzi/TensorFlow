# page80 自定义损失函数

import tensorflow as tf

v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

sess = tf.InteractiveSession()
print(tf.greater(v1, v2).eval())
# output [false false true true]

print(tf.where(tf.greater(v1, v2), v1, v2).eval())
# output [4. 3. 3. 4.]