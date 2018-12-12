import tensor as tf
state=tf.Variable(0,name="counter")
# 加法op
newValue=tf.add(state,1)
# 赋值op
update=tf.assign(state,newValue)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(state))
    for _ in range(5):
         sess.run(update)
         print(sess.run(state))