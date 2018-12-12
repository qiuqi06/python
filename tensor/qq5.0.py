import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# load data
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
n_input=28
max_time=28
lstm_size=100
n_classes=10
batch_size=100
n_batch=mnist.train.num_examples//batch_size
# define placeholder
x=tf.placeholder(tf.float32,[None,784])
y=tf.placeholder(tf.float32,[None,10])

weights=tf.Variable(tf.truncated_normal([lstm_size,n_classes],stddev=0))
biases=tf.Variable(tf.constant(0.1,shape=[n_classes]))

def RNN(X,weights,biases):
    # 转为50 28 maxtime 28n_inputs
    inputs=tf.reshape(X,[-1,max_time,n_input])
    # 定义lstm CELL
    lstm_cell=tf.contrib.rnn.BasicLSTMCell(lstm_size)
    outputs,final_state=tf.nn.dynamic_rnn(lstm_cell,inputs,dtype=tf.float32)
    results=tf.nn.softmax(tf.matmul(final_state[1],weights)+biases)
    return results

prediction=RNN(x,weights,biases)

cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))  #返回一维张量最大值的位置


# bool list
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))  #返回一维张量最大值的位置
# correct rate
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(6):
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch((batch_size))
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys})
        test_acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels})
        print("ITer"+str(epoch)+",Test Accuracy"+str(test_acc)+"，train Accuracy")


