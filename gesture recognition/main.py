import numpy as np
from skimage import io
import tensorflow as tf

batch_size = 64

data_x = []
data_y = []
import os
from PIL import Image

for filename in os.listdir("D:\\handpicture\\0"):
    im = Image.open('D:\\handpicture\\0\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(0)

for filename in os.listdir("D:\\handpicture\\1"):
    im = Image.open('D:\\handpicture\\1\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(1)

for filename in os.listdir("D:\\handpicture\\2"):
    im = Image.open('D:\\handpicture\\2\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(2)

for filename in os.listdir("D:\\handpicture\\3"):
    im = Image.open('D:\\handpicture\\3\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(3)
for filename in os.listdir("D:\\handpicture\\4"):
    im = Image.open('D:\\handpicture\\4\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(4)
for filename in os.listdir("D:\\handpicture\\5"):
    im = Image.open('D:\\handpicture\\5\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(5)
for filename in os.listdir("D:\\handpicture\\6"):
    im = Image.open('D:\\handpicture\\6\\' + filename)
    im2 = np.array(im)

    data_x.append(im2)
    data_y.append(6)

for filename in os.listdir("D:\\handpicture\\7"):
    im = Image.open('D:\\handpicture\\7\\'+filename)
    im2 = np.array(im)
    data_x.append(im2)
    data_y.append(7)

for filename in os.listdir("D:\\handpicture\\8"):
    im = Image.open('D:\\handpicture\\8\\'+filename)
    im2 = np.array(im)
    data_x.append(im2)
    data_y.append(8)

for filename in os.listdir("D:\\handpicture\\9"):
    im = Image.open('D:\\handpicture\\'+filename)
    im2 = np.array(im)
    data_x.append(im2)
    data_y.append(9)

data_x = np.array(data_x)
data_y = np.array(data_y)
print(data_x)
print(data_y)
print(data_x.shape, data_y.shape)

from sklearn.preprocessing import OneHotEncoder

onehot_encoder = OneHotEncoder(sparse=False)
data_y = data_y.reshape(-1, 1)
data_y = onehot_encoder.fit_transform(data_y)
# print(data_y)

from sklearn.model_selection import train_test_split

train_x_disorder, test_x_disorder, train_y_disorder, test_y_disorder = train_test_split(data_x, data_y, train_size=0.8,
                                                                                        random_state=33)
print(train_x_disorder.shape)

input_queue = tf.train.slice_input_producer([train_x_disorder, train_y_disorder], shuffle=True)
x_batch_train, y_batch_train = tf.train.batch(input_queue, batch_size=batch_size, num_threads=1,
                                              allow_smaller_final_batch=True)


# x_batch=np.array(x_batch)
# y_batch=np.array(y_batch)
# print(x_batch.dtype,y_batch.shape)
# 矩阵

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 偏置
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 卷积处理 变厚过程
def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1] x_movement、y_movement就是步长
    # Must have strides[0] = strides[3] = 1 padding='SAME'表示卷积后长宽不变
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# pool 长宽缩小一倍
def max_pool_3x3(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')


xs = tf.placeholder(tf.float32, [None, 100, 100, 3])  # 原始数据的维度：16
print(xs.name)
# input_xs = tf.reshape(xs, [-1, 64, 64, 1])  # 原始数据16变成二维图片4*4
ys = tf.placeholder(tf.float64, [None, 7])  # 输出数据为维度：1
keep_prob = tf.placeholder(tf.float32)

## conv1 layer ##第一卷积层
W_conv1 = weight_variable([5, 5, 3, 8])  # patch 2x2, in size 1, out size 32,每个像素变成32个像素，就是变厚的过程
b_conv1 = bias_variable([8])
h_conv1 = tf.nn.relu(conv2d(xs, W_conv1) + b_conv1)  # output size 2x2x32，长宽不变，高度为32的三维图像
h_pool1 = max_pool_3x3(h_conv1)  # output size 2x2x32 长宽缩小一倍

## conv2 layer ##第二卷积层
W_conv2 = weight_variable([5, 5, 8, 8])  # patch 2x2, in size 32, out size 64
b_conv2 = bias_variable([8])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 输入第一层的处理结果 输出shape 4*4*64
h_pool2 = max_pool_3x3(h_conv2)

## conv2 layer ##第3卷积层
W_conv3 = weight_variable([3, 3, 8, 64])  # patch 2x2, in size 32, out size 64
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # 输入第一层的处理结果 输出shape 4*4*64
h_pool3 = max_pool_3x3(h_conv3)

## conv2 layer ##第4卷积层
W_conv4 = weight_variable([3, 3, 64, 64])  # patch 2x2, in size 32, out size 64
b_conv4 = bias_variable([64])
h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)  # 输入第一层的处理结果 输出shape 4*4*64
h_pool4 = max_pool_3x3(h_conv4)

W_conv5 = weight_variable([3, 3, 64, 48])  # patch 2x2, in size 32, out size 64
b_conv5 = bias_variable([48])
h_conv5 = tf.nn.relu(conv2d(h_pool4, W_conv5) + b_conv5)  # 输入第一层的处理结果 输出shape 4*4*64
h_pool5 = max_pool_3x3(h_conv5)

W_fc1 = weight_variable([4 * 4 * 48, 384])  # 4x4 ，高度为64的三维图片，然后把它拉成512长的一维数组
b_fc1 = bias_variable([384])
h_pool5_flat = tf.reshape(h_pool5, [-1, 4 * 4 * 48])  # 把4*4，高度为64的三维图片拉成一维数组 降维处理

h_fc1 = tf.nn.relu(tf.matmul(h_pool5_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)  # 把数组中扔掉比例为keep_prob的元素

W_fc2 = weight_variable([384, 192])  # 4x4 ，高度为64的三维图片，然后把它拉成512长的一维数组
b_fc2 = bias_variable([192])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)  # 把数组中扔掉比例为keep_prob的元素

W_fc3 = weight_variable([192, 7])  # 4x4 ，高度为64的三维图片，然后把它拉成512长的一维数组
b_fc3 = bias_variable([7])
prediction = tf.add(tf.matmul(h_fc2_drop, W_fc3), b_fc3)
print(prediction.name)
print(ys.name)
print(keep_prob.name)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=tf.argmax(ys, 1))
# 0.01学习效率,minimize(loss)减小loss误差
regularizer = tf.contrib.layers.l2_regularizer(0.4)
regularization = regularizer(W_fc1) + regularizer(W_fc2) + regularizer(W_fc3)
final_loss = tf.reduce_mean(cross_entropy) + regularization
train_step = tf.train.AdamOptimizer(0.001).minimize(final_loss)
result = tf.argmax(prediction, 1)
crorent_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
accuracy = tf.reduce_mean(tf.cast(crorent_prediction, tf.float32))

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # tf.train.start_queue_runners()

    coord = tf.train.Coordinator()
    # 使用start_queue_runners 启动队列填充
    threads = tf.train.start_queue_runners(sess, coord)
    # 训练500次
    for i in range(300):
        sess.run(tf.local_variables_initializer())
        x_batch, y_batch = sess.run([x_batch_train, y_batch_train])
        sess.run(train_step, feed_dict={xs: x_batch, ys: y_batch, keep_prob: 0.9})

        if i % 5 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={xs: x_batch, ys: y_batch, keep_prob: 1.0})
            print('after %d steps training accuracy is %g%%' % (i, 100 * train_accuracy))
            test_accuracy = sess.run(accuracy, feed_dict={xs: test_x_disorder, ys: test_y_disorder, keep_prob: 1.0})
            print('test accuracy is %g%%' % (test_accuracy * 100))