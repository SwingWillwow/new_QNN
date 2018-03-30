import tensorflow as tf
# import numpy as np
import numpy as np
from numba import jit
import numba as nb
import time
import datetime
import math
import random
from pathlib import Path
from tensorflow.examples.tutorials.mnist import input_data

model_num = str(input('model number?'))
compressed_model_num = str(input('compressed model number?'))
CHECK_POINT_DIR = 'model/' + model_num + '/'
LOG_FREQUENCY = 1  # how often log training information
BATCH_SIZE = 128  # batch size
EPOCH_SIZE = int(60000 // BATCH_SIZE)
EPOCH_NUM = 100
MAX_STEPS = EPOCH_SIZE * EPOCH_NUM
TRAIN_LOG_DIR = 'train/' + model_num + '/'
DATA_DIR = str(Path.home()) + '/training_data/MNIST-data/'
EVAL_LOG_DIR = 'eval/' + model_num + '/'
# COMPRESS_MODEL_DIR = str(Path.home()) + '/QNN/compressed_model/' + compressed_model_num + '/'
COMPRESS_MODEL_DIR = 'compressed_model/' + compressed_model_num + '/'


def activations_summaries(activations):
    tf.summary.histogram(activations.op.name + '/activations', activations)
    tf.summary.scalar(activations.op.name + '/sparsity', tf.nn.zero_fraction(activations))


def weight_variables(shape, name):
    initial = tf.truncated_normal(shape=shape, stddev=0.1)
    return tf.Variable(initial, name=name + '_weight')


def bias_variables(shape, name):
    initial = tf.constant(value=0.1, dtype=tf.float32, shape=shape)
    return tf.Variable(initial, name=name + '_bias')


def conv2d(x, W):
    return tf.nn.conv2d(input=x, filter=W, strides=[1, 1, 1, 1],
                        padding='SAME')


def max_pooling_2x2(x):
    return tf.nn.max_pool(value=x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


def load_data():
    mnist = input_data.read_data_sets(DATA_DIR, validation_size=0)
    # get training data
    train_data = mnist.train.images
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    # get evaluating data
    eval_data = mnist.test.images
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
    # init data set
    train_data_set = tf.data.Dataset().from_tensor_slices((train_data, train_labels))
    train_data_set = train_data_set.repeat(EPOCH_NUM).batch(BATCH_SIZE).shuffle(10000)
    eval_data_set = tf.data.Dataset().from_tensors((eval_data, eval_labels))

    return train_data_set, eval_data_set


def inference(images, keep_rate):
    # first conv layer
    reshaped_image = tf.reshape(images, shape=[-1, 28, 28, 1])
    W_conv1 = weight_variables([5, 5, 1, 32], name='conv1')
    b_conv1 = bias_variables([32], name='conv1')
    h_conv1 = tf.nn.relu(conv2d(reshaped_image, W_conv1) + b_conv1)
    activations_summaries(h_conv1)
    tf.add_to_collection('conv_weights', W_conv1)
    tf.add_to_collection('biases', b_conv1)
    # first pooling

    h_pool1 = max_pooling_2x2(h_conv1)

    # second conv layer

    W_conv2 = weight_variables([5, 5, 32, 64], name='conv2')
    b_conv2 = bias_variables([64], name='conv2')
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    activations_summaries(h_conv2)
    tf.add_to_collection('conv_weights', W_conv2)
    tf.add_to_collection('biases', b_conv2)
    # second pooling layer

    h_pool2 = max_pooling_2x2(h_conv2)

    # flatten the pooling result for the dense layer

    flatten_pooling = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

    # fully-connected layer 1
    W_fc1 = weight_variables([7 * 7 * 64, 1024], name='fc1')
    b_fc1 = bias_variables([1024], name='fc1')

    h_fc1 = tf.nn.relu(tf.matmul(flatten_pooling, W_fc1) + b_fc1)
    activations_summaries(h_fc1)
    tf.add_to_collection('fc_weights', W_fc1)
    tf.add_to_collection('biases', b_fc1)
    # dropout

    h_fc1_after_drop = tf.nn.dropout(h_fc1, keep_rate)

    # fully-connected layer 2 (output layer)

    W_fc2 = weight_variables([1024, 10], name='fc2')
    b_fc2 = bias_variables([10], name='fc2')
    h_fc2 = tf.matmul(h_fc1_after_drop, W_fc2) + b_fc2
    activations_summaries(h_fc2)
    tf.add_to_collection('fc_weights', W_fc2)
    tf.add_to_collection('biases,', b_fc2)
    return h_fc2


def loss(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels,
                                                                   logits=logits,
                                                                   name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    return cross_entropy_mean


def loss_summary(total_loss):
    loss_averages = tf.train.ExponentialMovingAverage(decay=0.9)
    loss_averages_op = loss_averages.apply([total_loss])
    tf.summary.scalar(total_loss.op.name + ' (raw)', total_loss)
    tf.summary.scalar(total_loss.op.name, loss_averages.average(total_loss))
    return loss_averages_op


def train(total_loss, global_step):
    loss_averages_op = loss_summary(total_loss)
    with tf.control_dependencies([loss_averages_op]):
        opt = tf.train.AdamOptimizer(1e-4)
        grads = opt.compute_gradients(total_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step)

    for v in tf.trainable_variables():
        tf.summary.histogram(v.op.name, v)

    for grad, v in grads:
        if grad is not None:
            tf.summary.histogram(v.op.name + '/gradient', grad)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op('train_op')
    return train_op


def accuracy(logits, labels):
    predict = tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels)
    right_num = tf.reduce_sum(tf.cast(predict, tf.int32))
    total_num = labels.shape[0]
    return tf.div(tf.cast(right_num, tf.float32), total_num)


@jit(nopython=True)
def get_fc_code_book_and_indicator(subspace, subspace_length, centroid_num, n):
    code_book = np.zeros((subspace_length, centroid_num), dtype=np.float32)
    indicator = np.zeros((centroid_num, n), dtype=np.float32)
    classes = np.zeros(n, dtype=np.int32)
    indices = np.random.choice(np.array([s for s in range(n)]), centroid_num, False)
    for s in range(centroid_num):
        code_book[:, s] = subspace[:, indices[s]]
    min_distance = np.full(n, 1e38)
    while True:
        flag = False
        for j in range(n):
            for c in range(centroid_num):
                tmp_distance = np.linalg.norm(np.subtract(subspace[:, j],
                                                          code_book[:, c]))
                if tmp_distance < min_distance[j]:
                    min_distance[j] = tmp_distance
                    classes[j] = c
        sum_vector = np.zeros((subspace_length, centroid_num), dtype=np.float32)
        count = np.zeros(centroid_num, dtype=np.float32)
        for j in range(n):
            sum_vector[:, classes[j]] += subspace[:, j]
            count[classes[j]] += 1
        for c in range(centroid_num):
            tmp_centroid = sum_vector[:, c] / count[c]
            if np.linalg.norm(np.subtract(tmp_centroid, code_book[:, c])) > 1e-5:
                flag = True
                code_book[:, c] = tmp_centroid
        if not flag:
            break
    for j in range(n):
        indicator[classes[j], j] = 1
    return code_book, indicator


def get_fc_code_book(weights, centroid_num, subspace_length):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        name = str(weights.op.name)
        weights = sess.run(weights)
        m, n = weights.shape
        subspace_num = int(m // subspace_length)
        subspaces = np.split(weights, subspace_num, axis=0)
        code_books = []
        indicators = []
        for i in range(subspace_num):
            print('clustering ', name, 'subspace', i + 1)

            code_book, indicator = get_fc_code_book_and_indicator(subspaces[i],
                                                                  subspace_length,
                                                                  centroid_num,
                                                                  n)
            code_books.append(code_book)
            indicators.append(indicator)
            code_book_variable = tf.Variable(code_book, dtype=tf.float32, name=('fc_code_book_%d' % i))
            indicator_variable = tf.Variable(indicator, dtype=tf.float32, name=('fc_indicator_%d' % i))
            tf.add_to_collection('fc_code_books', code_book_variable)
            tf.add_to_collection('fc_indicators', indicator_variable)
        return code_books, indicators


def get_fc_search_list(images, weight_name):
    code_book_name = weight_name
    batch_num, image_length = images.shape
    code_book_list = []
    search_list = []
    print('getting', weight_name, ' search_list')
    for v in tf.get_collection('fc_code_books'):
        if str(v.name).find(code_book_name) != -1:
            code_book_list.append(v)
    # get code books in ascending order
    code_book_list.sort(key=lambda code_book: code_book.name, reverse=False)
    subspace_num = len(code_book_list)
    # change tf.Variable to numpy array. because tf objects don't support
    # item assignment
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if isinstance(images, tf.Tensor):
            images = np.array(sess.run(images))
        for i in range(subspace_num):
            code_book_list[i] = np.array(sess.run(code_book_list[i]))

    centroid_num = code_book_list[0].shape[1]
    sub_images = np.split(images, subspace_num, axis=1)
    # print(sub_images)
    for i in range(subspace_num):
        search_list.append(np.zeros([batch_num, centroid_num], dtype=np.float32))
        for j in range(batch_num):
            for c in range(centroid_num):
                search_list[i][j, c] = np.inner(code_book_list[i][:, c], sub_images[i][j, :])
    return search_list


def get_fc_response(search_list, weight_name):
    subspace_num = len(search_list)
    batch_num, centroid_num = search_list[0].shape
    indicators = []
    print('getting ', weight_name, ' response')
    for v in tf.get_collection('fc_indicators'):
        if str(v.name).find(weight_name) != -1:
            indicators.append(v)
    n = indicators[0].shape[1]
    response_map = np.zeros([batch_num, n])
    for i in range(batch_num):
        for j in range(n):
            for k in range(subspace_num):
                used_centroid = np.argmax(indicators[k][:, j])
                response_map[i, j] += search_list[k][i, used_centroid]
    return response_map


@jit(nopython=True)
def get_conv_code_book_and_indicator(subspace, subspace_length, centroid_num, height, width, out_channel):
    code_book = np.zeros((subspace_length, centroid_num), dtype=np.float32)
    indicator = np.zeros((height, width, centroid_num, out_channel))
    classes = np.zeros((height, width, out_channel), dtype=np.int32)
    indices = np.random.choice(np.array([s for s in range(height * width * out_channel)]), centroid_num, False)
    for s in range(centroid_num):
        t = indices[s]
        if height == 1:
            ti = 0
        else:
            ti = math.floor(t / (width * out_channel))
        if width == 1:
            tj = 0
        else:
            tj = math.floor((t - ti * width * out_channel) / out_channel)
        if out_channel == 1:
            tk = 0
        else:
            tk = t - ti * width * out_channel - tj * out_channel
        # print(ti, tj, tk)
        code_book[:, s] = subspace[ti, tj, :, tk]
    min_distance = np.full((height, width, out_channel), 1e38)
    while True:
        flag = False
        for j in range(height):
            for k in range(width):
                for l in range(out_channel):
                    for c in range(centroid_num):
                        tmp_distance = np.linalg.norm(np.subtract(subspace[j, k, :, l], code_book[:, c]))
                        if tmp_distance < min_distance[j, k, l]:
                            min_distance[j, k, l] = tmp_distance
                            classes[j, k, l] = c
        sum_vector = np.zeros((subspace_length, centroid_num), dtype=np.float32)
        count = np.zeros(centroid_num, dtype=np.float32)
        for j in range(height):
            for k in range(width):
                for l in range(out_channel):
                    sum_vector[:, classes[j, k, l]] += subspace[j, k, :, l]
                    count[classes[j, k, l]] += 1
        for c in range(centroid_num):
            if count[c] != 0:
                tmp_centroid = sum_vector[:, c] / count[c]
            else:
                continue
            if np.linalg.norm(np.subtract(tmp_centroid, code_book[:, c])) > 1e-5:
                flag = True
                code_book[:, c] = tmp_centroid
        if not flag:
            break
        for j in range(height):
            for k in range(width):
                for l in range(out_channel):
                    indicator[j, k, classes[j, k, l], l] = 1
    return code_book, indicator


def get_conv_code_book(weights, centroid_num, subspace_length):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        name = str(weights.op.name)
        weights = np.array(sess.run(weights))
        height, width, in_channel, out_channel = weights.shape
        subspace_num = int(in_channel / subspace_length)
        subspaces = np.split(weights, subspace_num, axis=2)
        code_books = []
        indicators = []
        for i in range(subspace_num):
            print('clustering ', name, ' subspace ', i + 1)
            code_book, indicator = get_conv_code_book_and_indicator(subspaces[i], subspace_length, centroid_num,
                                                                    height, width, out_channel)
            code_books.append(code_book)
            indicators.append(indicator)
            code_book_variable = tf.Variable(code_book, dtype=tf.float32, name=('conv_code_book_%d' % i))
            indicator_variable = tf.Variable(indicator, dtype=tf.float32, name=('conv_indicator_%d' % i))
            tf.add_to_collection('conv_code_books', code_book_variable)
            tf.add_to_collection('conv_indicators', indicator_variable)
        return code_books, indicators


def get_conv_search_list(images, weight_name):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        if isinstance(images, tf.Tensor):
            images = sess.run(images)
        # images = sess.run(images)
        code_book_name = weight_name
        print('getting ',weight_name,' search_list')
        batch_num, image_height, image_width, image_channel = images.shape
        code_book_list = []
        search_list = []
        for v in tf.get_collection('conv_code_books'):
            if str(v.name).find(code_book_name) != -1:
                code_book_list.append(v)
        code_book_list.sort(key=lambda code_book: code_book.name, reverse=False)
        subspace_num = len(code_book_list)
        for i in range(subspace_num):
            code_book_list[i] = sess.run(code_book_list[i])
        centroid_num = code_book_list[0].shape[1]
        sub_images = np.split(images, subspace_num, axis=3)
        for i in range(subspace_num):
            search_list.append(np.zeros([batch_num, image_height, image_width, centroid_num]
                                        , dtype=np.float32))
            for j in range(batch_num):
                for k in range(image_height):
                    for l in range(image_width):
                        for c in range(centroid_num):
                            search_list[i][j, k, l, c] = np.inner(code_book_list[i][:, c],
                                                                  sub_images[i][j, k, l, :])
    return search_list


def get_conv_response(search_list, weight_name, padding):
    subspace_num = len(search_list)
    batch_num, image_height, image_width, centroid_num = search_list[0].shape
    indicators = []
    print('getting ', weight_name, ' response')
    for v in tf.get_collection('conv_indicators'):
        if str(v.name).find(weight_name) != -1:
            indicators.append(v)
    indicators.sort(key=lambda indicator: indicator.name, reverse=False)
    height, width, centroid_num, out_channel = indicators[0].shape
    if padding == 'SAME':
        response_map = np.zeros([batch_num, image_height, image_width, out_channel],
                                dtype=np.float32)
        padding_height = height - 1
        padding_width = width - 1
        if padding_height % 2 == 1:
            padding_left = padding_height // 2
            padding_right = padding_left + 1
        else:
            padding_left = padding_height / 2
            padding_right = padding_left
        if padding_width % 2 == 1:
            padding_top = padding_width // 2
            padding_bottom = padding_top + 1
        else:
            padding_top = padding_width / 2
            padding_bottom = padding_top
        for n in range(batch_num):
            for h in range(image_height):
                for w in range(image_width):
                    for c in range(out_channel):
                        for i in range(-padding_left, padding_right + 1):
                            for j in range(-padding_top, padding_bottom + 1):
                                for k in range(subspace_num):
                                    if h + i < 0 or w + j < 0:
                                        continue
                                    used_centroid = np.argmax(indicators[k][i + padding_left, j + padding_top, :, c])
                                    response_map[n, h, w, c] += search_list[k][n, h + i, w + j, used_centroid]
    elif padding == 'VALID':
        response_map = np.zeros([batch_num, image_height - height + 1,
                                 image_width - width + 1, out_channel],
                                dtype=np.float32)
        out_height = image_height - height + 1
        out_width = image_width - width + 1
        for n in range(batch_num):
            for h in range(out_height):
                for w in range(out_width):
                    for c in range(out_channel):
                        for i in range(height):
                            for j in range(width):
                                for k in range(subspace_num):
                                    used_centroid = np.argmax(indicators[k][i, j, :, c])
                                    response_map[n, h, w, c] += search_list[k][n, h + i, w + j, used_centroid]
    return response_map


def train_model():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input_images')
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='input_labels')
    keep_rate = tf.placeholder(dtype=tf.float32, shape=None, name='keep_rate')
    logits = inference(x, keep_rate)
    losses = loss(logits, y)
    global_step = tf.train.get_or_create_global_step()
    train_op = train(losses, global_step=global_step)
    train_data_set, _ = load_data()
    train_iterator = train_data_set.make_initializable_iterator()
    next_data = train_iterator.get_next()
    with tf.Session() as sess:
        # initialization
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(train_iterator.initializer)
        saver = tf.train.Saver()
        summary_writer = tf.summary.FileWriter(logdir=TRAIN_LOG_DIR)
        merged_summaries = tf.summary.merge_all()
        for i in range(1, EPOCH_NUM * EPOCH_SIZE + 1):
            images, labels = sess.run(next_data)
            sess.run(train_op, feed_dict={x: images, y: labels, keep_rate: 0.5})
            if i % 100 == 0:
                ac_rate = accuracy(logits, labels)
                ac, current_loss = sess.run([ac_rate, losses], feed_dict={x: images, y: labels, keep_rate: 0.5})
                print(('%d' % i) + ' step: current accuracy ' + str(ac), ', current loss ' + str(current_loss))
            if i % 1000 == 0 or i == EPOCH_NUM * EPOCH_SIZE:
                saver.save(sess, CHECK_POINT_DIR, global_step)
                global_step_value = tf.train.global_step(sess, global_step)
                summaries = sess.run(merged_summaries, feed_dict={x: images,
                                                                  y: labels,
                                                                  keep_rate: 0.5})
                summary_writer.add_summary(summaries, global_step_value)


def eval_model():
    _, eval_data_set = load_data()
    eval_iterator = eval_data_set.make_initializable_iterator()
    next_eval_data = eval_iterator.get_next()

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='images')
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')
    keep_rate = tf.placeholder(dtype=tf.float32, shape=None, name='keep_rate')
    logits = inference(x, keep_rate)
    losses = loss(logits, y)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(eval_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('Can not find checkpoint files.')
            return
        images, labels = sess.run(next_eval_data)
        total_loss, ac_rate = sess.run([losses, accuracy(logits, labels)],
                                       feed_dict={x: images, y: labels, keep_rate: 1})
        print('testing finished!')
        print('final loss:' + str(total_loss))
        print('final accuracy' + str(ac_rate))
        merged_summaries = tf.summary.merge_all()
        summaries = sess.run(merged_summaries, feed_dict={x: images,
                                                          y: labels,
                                                          keep_rate: 1})
        summaries_writer = tf.summary.FileWriter(EVAL_LOG_DIR)
        summaries_writer.add_summary(summaries, global_step)


def compress_model_with_quantization():
    x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='images')
    y = tf.placeholder(dtype=tf.int32, shape=[None], name='labels')
    logits = inference(x)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(CHECK_POINT_DIR)
        # load model in CHECK_POINT_DIR
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('Can not find checkpoint files.')
            return
        # get trainable variables group in different type
        biases = tf.get_collection('biases')
        uncompressed_fc_weights = tf.get_collection('fc_weights')
        uncompressed_fc_weights.sort(key=lambda w: w.op.name, reverse=False)
        uncompressed_conv_weights = tf.get_collection('conv_weights')
        uncompressed_conv_weights.sort(key=lambda w: w.op.name, reverse=False)
        fc_weight_num = len(uncompressed_fc_weights)
        conv_weight_num = len(uncompressed_conv_weights)
        centroid_nums = []
        subspace_lengths = []
        conv_centroid_nums = []
        conv_subspace_lengths = []
        for i in range(1, fc_weight_num + 1):
            name = str(uncompressed_fc_weights[i - 1].op.name)
            print(name + ', and his shape is:', uncompressed_fc_weights[i - 1].shape)
            centroid_num = int(input('centroid number of ' + name + '?'))
            subspace_length = int(input('subspace length of ' + name + '?'))
            centroid_nums.append(centroid_num)
            subspace_lengths.append(subspace_length)
        for i in range(1, conv_weight_num + 1):
            name = str(uncompressed_conv_weights[i - 1].op.name)
            print(name + ', and his shape is:', uncompressed_conv_weights[i - 1].shape)
            centroid_num = int(input('centroid number of ' + name + '?'))
            subspace_length = int(input('subspace length of ' + name + '?'))
            conv_centroid_nums.append(centroid_num)
            conv_subspace_lengths.append(subspace_length)
        for weight in uncompressed_fc_weights:
            with tf.variable_scope(str(weight.op.name)):
                i = uncompressed_fc_weights.index(weight)
                get_fc_code_book(weight, centroid_nums[i], subspace_lengths[i])
        for weight in uncompressed_conv_weights:
            with tf.variable_scope(str(weight.op.name)):
                i = uncompressed_conv_weights.index(weight)
                get_conv_code_book(weight, conv_centroid_nums[i], conv_subspace_lengths[i])
        save_var_list = tf.get_collection('fc_code_books') + tf.get_collection('conv_code_books')
        save_var_list += tf.get_collection('fc_indicators') + tf.get_collection('conv_indicators')
        new_saver = tf.train.Saver(var_list=save_var_list, name='compressed')
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        new_saver.save(sess, COMPRESS_MODEL_DIR + 'compressed_model.ckpt')


def search_bias(biases, weight_name):
    for b in biases:
        if str(b.name).find(weight_name) != -1:
            bias = b
            return bias


def compressed_inference(images, keep_rate):
    with tf.Session() as sess:
        biases = tf.get_collection('biases')
        images = np.reshape(images, newshape=(-1, 28, 28, 1))
        # conv1
        conv1_bias = search_bias(biases, 'conv1')
        conv1_search_list = get_conv_search_list(images, 'conv1')
        conv1_response_map = get_conv_response(conv1_search_list, 'conv1', padding='SAME')
        h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1_response_map, conv1_bias))

        # pool1
        h_pool1 = max_pooling_2x2(h_conv1)

        # conv2
        conv2_bias = search_bias(biases, 'conv2')
        conv2_search_list = get_conv_search_list(h_pool1, 'conv2')
        conv2_response_map = get_conv_response(conv2_search_list, 'conv2', padding='SAME')
        h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2_response_map, conv2_bias))

        # pool2
        h_pool2 = max_pooling_2x2(h_conv2)

        # flatten pooling result
        flatten_pooling = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

        # fc1
        fc1_bias = search_bias(biases, 'fc1')
        fc1_search_list = get_fc_search_list(flatten_pooling, 'fc1')
        fc1_response_map = get_fc_response(fc1_search_list, 'fc1')
        h_fc1 = tf.nn.relu(tf.add(fc1_response_map, fc1_bias))

        # drop out
        drop_out1 = tf.nn.dropout(h_fc1, keep_rate)

        # fc2
        fc2_bias = search_bias(biases, 'fc2')
        fc2_search_list = get_fc_search_list(drop_out1, 'fc2')
        fc2_response_map = get_fc_response(fc2_search_list, 'fc2')
        h_fc2 = tf.add(fc2_response_map, fc2_bias)

        return h_fc2


def eval_compressed_model():
    _, eval_data_set = load_data()
    eval_iterator = eval_data_set.make_initializable_iterator()
    next_eval_data = eval_iterator.get_next()
    print('loading graph.')
    saver = tf.train.import_meta_graph(COMPRESS_MODEL_DIR + 'compressed_model.ckpt.meta')
    print('graph load finished.')
    with tf.Session() as sess:
        sess.run(eval_iterator.initializer)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(COMPRESS_MODEL_DIR))
        images, labels = sess.run(next_eval_data)
        logits = compressed_inference(images, 1)
        losses = loss(logits, labels)
        total_loss, ac_rate = sess.run([losses, accuracy(logits, labels)])
        print('testing finished!')
        print('final loss:' + str(total_loss))
        print('final accuracy' + str(ac_rate))
        # merged_summaries = tf.summary.merge_all()
        # summaries = sess.run(merged_summaries, feed_dict={x: images,
        #                                                   y: labels,
        #                                                   keep_rate: 1})
        # summaries_writer = tf.summary.FileWriter(EVAL_LOG_DIR)
        # summaries_writer.add_summary(summaries, global_step)


def main(args):
    compress_or_normal = str(input('compress or normal mode?'))
    is_compress = compress_or_normal == 'compress'
    if is_compress:
        get_or_eval = str(input('get code books or eval model?'))
        is_get = get_or_eval == 'get'
        if is_get:
            compress_model_with_quantization()
        else:
            eval_compressed_model()
    else:
        train_or_eval = str(input('train or eval model?'))
        is_train = train_or_eval == 'train'

        if is_train:
            if tf.gfile.Exists(CHECK_POINT_DIR):
                is_new = str(input('new model?'))
                is_new = is_new == 'True'
                if is_new:
                    tf.gfile.DeleteRecursively(CHECK_POINT_DIR)
                    tf.gfile.MakeDirs(CHECK_POINT_DIR)
                    train_model()
                    return
                else:
                    train_model()
                    return
            tf.gfile.MakeDirs(CHECK_POINT_DIR)
            train_model()
            return
        else:
            if tf.gfile.Exists(EVAL_LOG_DIR):
                tf.gfile.DeleteRecursively(EVAL_LOG_DIR)
            tf.gfile.MakeDirs(EVAL_LOG_DIR)
            eval_model()


if __name__ == '__main__':
    tf.app.run()
