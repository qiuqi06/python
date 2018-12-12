import tensor as tf
m1=tf.constant([[2,3]])
m2=tf.constant([[2],[3]])
m=tf.matmul(m1,m2)
sess=tf.Session()

r=sess.run(m)
print(r)
