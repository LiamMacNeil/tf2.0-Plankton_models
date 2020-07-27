
from pylab import *
from scipy import io
import matplotlib.pyplot as plt

# for getting the dataset
import glob
import os
from scipy.ndimage import imread
import random
import tensorflow as tf

from sklearn import linear_model

import pickle

from sklearn import svm, cross_validation

IMAGE_SIZE = 128

IMAGE_DEPTH = 1
IMAGE_LENGTH = 4
CLASS_SIZE = 19
BATCH_SIZE = CLASS_SIZE * IMAGE_LENGTH
INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE * IMAGE_DEPTH


data_dir = "/home/fractal/workspace/02_hoi/4Deep/classes19/"
data_train = "_train"
data_valid = "_test"

data_test = "_test"


save_dir = "/home/fractal/workspace/02_hoi/4Deep/classes19/save_dir"


num_val = 50
num_tst = 100

iteration = 3100


validation_check = 100

learning_rate = 1e-4

reg_constant = 0.0001

REUGULARIZAER = 2

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

phase_train = tf.compat.v1.placeholder(tf.bool, name='phase_train')

######################### getting dataset #################################

def get_dataset(data_type):
    images = []
    targets = []

    data_type_post = data_train

    if data_type == 1:
        data_type_post = data_valid
    elif data_type == 2:
        data_type_post = data_test

    for i in range(CLASS_SIZE):

        files = glob.glob(data_dir + str(i) + data_type_post + '/' + '*.png')
        selected_files = random.sample(files, IMAGE_LENGTH)
        for j in range(IMAGE_LENGTH):
            pngfile = imread(selected_files[j])
            images.append(pngfile)
            target = i
            targets.append(target)

    images = np.asarray(images, dtype=np.float32)
    targets = np.asarray(targets)
    # not using this one now due to structure of y

    target = []
    for j in range(CLASS_SIZE):
        t_building = []

        for k in range(CLASS_SIZE):
            if k < j:
                t_building.append(0)
            elif k == j:
                t_building.append(1)
            else:
                t_building.append(0)

        for l in range(IMAGE_LENGTH):
            target.append(t_building)

    y_batch_v = np.asarray(target)
    return images.reshape(BATCH_SIZE, INPUT_SIZE), y_batch_v


######################### creating confusion matrix ########################
def confusion_matrix(sess, data_type=2):
    images = []
    targets = []

    cn_matrix = np.zeros((CLASS_SIZE, CLASS_SIZE))
    append_class_num = []

    if data_type == 1:
        data_type_post = data_valid
    elif data_type == 2:
        data_type_post = data_test

    for i in range(CLASS_SIZE):

        files =glob.glob(data_dir + '\\' + str(i) + data_type_post + '\\' + '*.png')

        num_test_class = np.shape(files)[0]

        append_class_num.append(num_test_class)

        target = []
        for j in range(CLASS_SIZE):
            t_building = []

            for k in range(CLASS_SIZE):
                if k < j:
                    t_building.append(0)
                elif k == j:
                    t_building.append(1)
                else:
                    t_building.append(0)

            for l in range(IMAGE_LENGTH):
                target.append(t_building)

        Y_target = np.asarray(target)

        y_pred = []
        y_true = []
        for j in range(num_test_class):
            pngfile = imread(files[j])
            images.append(pngfile)
            targets.append(i)

            if (np.shape(images)[0] == BATCH_SIZE) or ((i == (CLASS_SIZE - 1)) and (j == (num_test_class - 1))):
                images = np.asarray(images, dtype=np.float32)
                images = images.reshape(-1, INPUT_SIZE)

                pred_out_ = sess.run(pred_out, feed_dict={x: images, y_: Y_target,keep_prob: 1.0,phase_train: False})

                for k in range(np.shape(pred_out_)[0]):
                    index_pred = np.argmax(pred_out_[k])
                    y_pred.append(index_pred)
                    target = targets[k]
                    y_true.append(target)
                    cn_matrix[target][index_pred] = cn_matrix[target][index_pred] + 1

                targets = []
                images = []

    num_class = np.shape(append_class_num)[0]
    for i in range(num_class):
        cn_matrix[i, :] = cn_matrix[i, :] / append_class_num[i]
        print(i)
        print(cn_matrix[i, :])


    fig = plt.figure()
    plt.matshow(cn_matrix)
    plt.title('Problem: Hologram Confusion Matrix')
    plt.colorbar()
    plt.ylabel('True Label')
    plt.xlabel('Predicated Label')
    plt.savefig('confusion_matrix.jpg')



########### convolution model ############################################




sess = tf.compat.v1.InteractiveSession()
F, Q, T = [], [], []

x = tf.compat.v1.placeholder(tf.float32, shape=[None, IMAGE_SIZE*IMAGE_SIZE*IMAGE_DEPTH])
y_ = tf.compat.v1.placeholder(tf.float32, shape=[None, CLASS_SIZE])

W = tf.Variable(tf.zeros([IMAGE_SIZE*IMAGE_SIZE*IMAGE_DEPTH, CLASS_SIZE]))
b = tf.Variable(tf.zeros([CLASS_SIZE]))

sess.run(tf.compat.v1.initialize_all_variables())

y = tf.matmul(x, W) + b

cross_entropy = tf.reduce_mean(
    input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_), logits=y))


train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(input=y, axis=1), tf.argmax(input=y_, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

if REUGULARIZAER == 1:
    regularizer_weight_decay = tf.keras.regularizers.l1(reg_constant)
elif REUGULARIZAER == 2:
    regularizer_weight_decay = tf.keras.regularizers.l2(0.5 * (reg_constant))

#print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))#




for i in range(400):

    if i % 100 == 0 :
        print('train #',i)
    x_batch,y_batch = get_dataset(0)

    train_step.run(feed_dict={x: x_batch, y_: y_batch})

# train before doing convolution




def weight_variable(shape,layer_index):
    '''
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    '''

    return tf.compat.v1.get_variable(name = 'filter'+str(layer_index), shape=shape, initializer=tf.compat.v1.keras.initializers.VarianceScaling(scale=1.0, mode="fan_avg", distribution="uniform"), regularizer=regularizer_weight_decay)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(input=x, filters=W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool2d(input=x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def error_rate(predictions, labels):
    """Return the error rate based on dense shapespredictions and sparse labels."""
    return 100.0 - (
        100.0 *
        numpy.sum(numpy.argmax(predictions, 1) == labels) /
        predictions.shape[0])


def batch_norm(x, n_out, phase_train, scope='bn'):
    """
    Batch normalization on convolutional maps.
    Args:
        x:           Tensor, 4D BHWD input maps
        n_out:       integer, depth of input maps
        phase_train: boolean tf.Varialbe, true indicates training phase
        scope:       string, variable scope
    Return:
        normed:      batch-normalized maps
    """
    with tf.compat.v1.variable_scope(scope):
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]),
                                     name='beta', trainable=True)
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]),
                                      name='gamma', trainable=True)
        batch_mean, batch_var = tf.nn.moments(x=x, axes=[0,1,2], name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=0.5)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(pred=phase_train,
                            true_fn=mean_var_with_update,
                            false_fn=lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-4)
    return normed



W_conv1 = weight_variable([5, 5, 1, 64],1)
b_conv1 = bias_variable([64])
x_image = tf.reshape(x, [-1,IMAGE_SIZE,IMAGE_SIZE,1])

h_conv1 = (conv2d(x_image, W_conv1) + b_conv1)
h_conv1 = batch_norm(h_conv1,64,phase_train,'conv_1')
h_conv1 = tf.nn.relu(h_conv1)
h_pool1 = max_pool_2x2(h_conv1)


W_conv2 = weight_variable([3, 3, 64, 128],2)
b_conv2 = bias_variable([128])

h_conv2 = (conv2d(h_pool1, W_conv2) + b_conv2)
h_conv2 = batch_norm(h_conv2,128,phase_train,'conv_2')
h_conv2 = tf.nn.relu(h_conv2)
h_pool2 = max_pool_2x2(h_conv2)

'''
W_conv3 = weight_variable([3, 3, 128, 128],3)
b_conv3 = bias_variable([128])

h_conv3 = (conv2d(h_pool2, W_conv3) + b_conv3)
h_conv3 = batch_norm(h_conv3,128,phase_train,'conv_3')
h_conv3 = tf.nn.relu(h_conv3)
h_pool3 = max_pool_2x2(h_conv3)

W_conv4 = weight_variable([3, 3, 128, 256],4)
b_conv4 = bias_variable([256])

h_conv4 = (conv2d(h_pool3, W_conv4) + b_conv4)
h_conv4 = batch_norm(h_conv4,256,phase_train,'conv_4')
h_conv4 = tf.nn.relu(h_conv4)
h_pool4 = max_pool_2x2(h_conv4)



W_conv5 = weight_variable([2, 2, 256, 256],5)
b_conv5 = bias_variable([256])

h_conv5 = (conv2d(h_conv4, W_conv5) + b_conv5)
h_conv5 = batch_norm(h_conv5,256,phase_train,'conv_5')
h_conv5 = tf.nn.relu(h_conv5)
h_pool5 = max_pool_2x2(h_conv5)


W_conv6 = weight_variable([2, 2, 256, 512],6)
b_conv6 = bias_variable([512])

h_conv6 = (conv2d(h_pool5, W_conv6) + b_conv6)
h_conv6 = batch_norm(h_conv6,512,phase_train,'conv_6')
h_conv6 = tf.nn.relu(h_conv6)


W_conv7 = weight_variable([2, 2, 512, 512],7)
b_conv7 = bias_variable([512])

h_conv7 = (conv2d(h_conv6, W_conv7) + b_conv7)
h_conv7 = batch_norm(h_conv7,512,phase_train,'conv_7')
h_conv7 = tf.nn.relu(h_conv7)
h_pool7 = max_pool_2x2(h_conv7)


W_conv8 = weight_variable([2, 2, 512, 1024],8)
b_conv8 = bias_variable([1024])

h_conv8 = (conv2d(h_pool7, W_conv8) + b_conv8)
h_conv8 = batch_norm(h_conv8,1024,phase_train,'conv_8')
h_conv8 = tf.nn.relu(h_conv8)
'''



W_fc1 = weight_variable([32 * 32 * 128, 1024],'fully_connected_1')
b_fc1 = bias_variable([1024])

h_pool4_flat = tf.reshape(h_pool2, [-1, 32 * 32 * 128])



keep_prob = tf.compat.v1.placeholder(tf.float32)

h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, 1 - (keep_prob))

W_fc2 = weight_variable([1024, 19],'fully_connected_2')
b_fc2 = bias_variable([19])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    input_tensor=tf.nn.softmax_cross_entropy_with_logits(labels=tf.stop_gradient(y_), logits=y_conv))

reg_losses = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.REGULARIZATION_LOSSES)

loss = cross_entropy + reg_constant*sum(reg_losses)

train_step = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(loss)

correct_prediction = tf.equal(tf.argmax(input=y_conv, axis=1), tf.argmax(input=y_, axis=1))
accuracy = tf.reduce_mean(input_tensor=tf.cast(correct_prediction, tf.float32))

pred_out = tf.nn.softmax(y_conv)


print(accuracy)
F, Q, K = [], [], []

sess.run(tf.compat.v1.initialize_all_variables())


for i in range(iteration):

    x_batch_v, y_batch_v = get_dataset(0)
    result_loss, output = sess.run([loss, y_conv], feed_dict={x: x_batch_v, keep_prob: 0.5, y_: y_batch_v, phase_train: True})

    if i % 100 == 0:

        train_accuracy = accuracy.eval(feed_dict={
            x: x_batch_v, y_: y_batch_v, keep_prob: 1,phase_train: True})
        print("step %d, training accuracy %g" % (i, train_accuracy))

        validation = []
        for j in range(num_val):
            x_valid, y_valid = get_dataset(1)
            result_valid = accuracy.eval(feed_dict={x: x_valid, y_: y_batch_v, keep_prob: 1.0,phase_train: False})
            validation.append(result_valid)

        valid_accuracy_new = np.mean(validation)

        print('Validation accuracy: ' + str(valid_accuracy_new))
        print('Variance:' + str(np.var(validation)))
        print('loss:' + str(result_loss))

        print()

        F.append(i)
        Q.append(train_accuracy)
        K.append(valid_accuracy_new)

        plt.axis([0, 3000, 0, 1])

        plt.scatter(i, train_accuracy, color='red')
        plt.scatter(i, valid_accuracy_new, color='blue')

        plt.plot(F, Q, color='red', label="training")
        plt.plot(F, K, color='blue', label="validation")

        plt.xlabel('iteration')
        plt.ylabel('training & validation accuracy')
        plt.title('cell red:training blue:validation')

        if (i) == 0:
            plt.legend(loc='lower right')

        plt.pause(0.05)

    #train_step.run(feed_dict={x: x_batch_v, y_: y_batch_v, keep_prob: 0.8})
    sess.run(train_step, feed_dict={x: x_batch_v, y_: y_batch_v, keep_prob: 0.5,phase_train: True})

plt.savefig('ac.png')



confusion_matrix(sess)

saver = tf.compat.v1.train.Saver()
save_path = saver.save(sess, save_dir)
print("Model saved in file: %s" % save_path)
















