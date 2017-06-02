import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

L1=784
L2=800
L3=10

x = tf.placeholder(tf.float32, [None, L1])
W1 = tf.Variable(tf.truncated_normal([L1, L2], stddev=0.1))
b1 = tf.Variable(tf.truncated_normal([L2], stddev=0.1))
z = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
W3 = tf.Variable(tf.truncated_normal([L2, L3], stddev=0.1))
b3 = tf.Variable(tf.truncated_normal([L3], stddev=0.1))
y = tf.nn.softmax(tf.matmul(z, W3) + b3)


y_ = tf.placeholder(tf.float32, [None, L3])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for i in range(1000):
    print(i)
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

saver = tf.train.Saver()
saver.save(sess, './model.cpkt')
print("done")