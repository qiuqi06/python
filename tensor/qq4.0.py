import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
batch_size=100
n_batch=mnist.train.num_examples//batch_size
# init weight
def weight_variable(shape):
    inital=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(inital)
#init bias
def bias_variable(shape):
    inital=tf.constant(0.1,shape=shape)
    return tf.Variable(inital)
# 卷基层
def conv2d(x,w):
    return tf.nn.conv2d(x,w,strides=[1,1,1,1],padding='SAME')
# 池化层
def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding="SAME")

x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])
# 改变x格式
x_image=tf.reshape(x,[-1,28,28,1])
# 初始化第一层权值和偏置 5*5采样窗口 32卷积核 1特征平面
w_conv1=weight_variable([5,5,1,32])
b_conv1=bias_variable([32])
# 把x_image权值向量就行卷积，再加偏向  应用relu激活函数
h_conv1=tf.nn.relu(conv2d(x_image,w_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)
# 第二层                            64个卷积核从32个平面提取特征
w_conv2=weight_variable([5,5,32,64])
b_conv2=bias_variable([64])
# h_pool1和权值卷积
h_conv2=tf.nn.relu(conv2d(h_pool1,w_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)

#第一次卷积28*28，第一次池化后14*14 第二次池化后7*7
w_fc1=weight_variable([7*7*64,1024])
b_fc1=bias_variable([1024])

#把池化层的输出转为扁平化1维
h_pool2_flat=tf.reshape(h_pool2,[-1,7*7*64])
# h_pool2_flat=h_pool2
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,w_fc1)+b_fc1)

keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)

w_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])

prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,w_fc2)+b_fc2)

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))

accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

init=tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(21):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch((batch_size))
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.0})

        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        print("ITer"+str(epoch)+",Test Accuracy"+str(test_acc)+"，train Accuracy")


