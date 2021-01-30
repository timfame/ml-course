import itertools
import tensorflow._api.v2.compat.v1 as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plot


CLASSES = 10
CLASS_NAMES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bad", "Ankle boot"]


class Dataset:

    def __init__(self, train_x, train_y, test_x, test_y, one_hotted=False):
        self.train_x = train_x
        self.test_x = test_x

        if not one_hotted:
            self.train_y, self.test_y = [], []
            for y in train_y:
                one_hot = [0.0] * CLASSES
                one_hot[y] = 1.0
                self.train_y.append(one_hot)
            for y in test_y:
                one_hot = [0.0] * CLASSES
                one_hot[y] = 1.0
                self.test_y.append(one_hot)
        else:
            self.train_y = train_y
            self.test_y = test_y

        self.first_not_batched = 0

    def get_next_train_batch(self, batch_size):
        start = self.first_not_batched
        end = min(start + batch_size, len(self.train_x))
        self.first_not_batched += batch_size
        return self.train_x[start:end], self.train_y[start:end]

    def reset_batches(self):
        self.first_not_batched = 0


SAVE_PATH = "./saves/mnist"
LEARNING_RATE = 0.001
EPOCHS = 20
BATCH_SIZE = 200
DROPOUT = 0.3
STDDEV = 0.02


def get_dataset(load_data_func):
    (train_x, train_y), (test_x, test_y) = load_data_func()
    train_x = np.array(train_x, np.float)
    train_x /= 255.0
    test_x = np.array(test_x, np.float)
    test_x /= 255.0
    return Dataset(train_x, train_y, test_x, test_y)


def get_mnist_dataset():
    return get_dataset(lambda : mnist.load_data())


def get_fashion_dataset():
    return get_dataset(lambda : fashion_mnist.load_data())


def draw_confusion_matrix(cm):
    plot.figure(figsize=(8, 8))
    plot.imshow(cm, interpolation="nearest", cmap=plot.cm.Blues)
    plot.title("Confusion matrix")
    plot.colorbar()
    tick_marks = np.arange(CLASSES)
    plot.xticks(tick_marks, CLASS_NAMES, rotation=45)
    plot.yticks(tick_marks, CLASS_NAMES)

    cm = np.around(cm.astype("float") / cm.sum(axis=1)[:, np.newaxis], decimals=4)
    threshold = cm.max() / 2.

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plot.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plot.tight_layout()
    plot.ylabel("Real")
    plot.xlabel("Predicted")

    plot.show()


def draw_max_failure_pictures(images, indexes):
    figure, ax = plot.subplots(nrows=CLASSES, ncols=CLASSES)
    for i in range(CLASSES * CLASSES):
        img_ind = indexes[i // CLASSES][i % CLASSES]
        ax.ravel()[i].imshow(images[img_ind[0]], cmap=plot.cm.binary)
        ax.ravel()[i].set_yticklabels([])
        ax.ravel()[i].set_xticklabels([])
        ax.ravel()[i].get_xaxis().set_visible(False)
        ax.ravel()[i].get_yaxis().set_visible(True)
        y_label_text = "{:.3f}".format(img_ind[1])
        if i % CLASSES == 0:
            y_label_text = "Real\n" + CLASS_NAMES[i // CLASSES] + "\n\n" + y_label_text
        ax.ravel()[i].set_ylabel(y_label_text)
        if i < CLASSES:
            ax.ravel()[i].set_title(CLASS_NAMES[i])
    figure.tight_layout(pad=3.0)
    plot.show()


def create_conv_layer(input_data, input_channels_count, filters_count, filter_shape, pool_shape, name):

    conv_shape = [filter_shape[0], filter_shape[1], input_channels_count, filters_count]

    weights = tf.Variable(tf.truncated_normal(conv_shape, stddev=0.03), name=name + '_weights')
    bias = tf.Variable(tf.truncated_normal([filters_count]), name=name + '_bias')

    out_layer = tf.nn.conv2d(input_data, weights, [1, 1, 1, 1], padding='SAME')

    out_layer += bias

    out_layer = tf.nn.relu(out_layer)

    if pool_shape is None:
        return out_layer

    ksize = [1, pool_shape[0], pool_shape[1], 1]
    strides = [1, 2, 2, 1]
    out_layer = tf.nn.max_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

    return out_layer


def train(input_images, real, session, optimiser, loss, accuracy, saver, dataset):

    total_batch = len(dataset.train_x) // BATCH_SIZE

    for epoch in range(EPOCHS):

        dataset.reset_batches()
        avg_loss = 0

        for i in range(total_batch):
            batch_x, batch_y = dataset.get_next_train_batch(BATCH_SIZE)
            _, c = session.run([optimiser, loss], feed_dict={input_images: batch_x, real: batch_y})
            avg_loss += c

        avg_loss /= total_batch
        test_accuracy = session.run(accuracy, feed_dict={input_images: dataset.test_x, real: dataset.test_y})

        print("Epoch:", (epoch + 1),
              "Loss =", "{:.3f}".format(avg_loss),
              "Test accuracy:", " {:.3f}".format(test_accuracy))

    print("Total accuracy:",
          session.run(accuracy, feed_dict={input_images: dataset.test_x, real: dataset.test_y}) * 100, "%")
    saver.save(session, SAVE_PATH)


def main(trainModel=True, buildConfusionMatrix=True, restore=False, buildClassifiedMatrix=True):

    tf.disable_v2_behavior()

    input_images = tf.placeholder(tf.float32, [None, 28, 28], name="Input")
    real = tf.placeholder(tf.float32, [None, CLASSES], name="real_classes")

    layer1 = create_conv_layer(tf.reshape(input_images, [-1, 28, 28, 1]), 1, 28, [5, 5], [2, 2], name="conv_no_pool")
    layer2 = create_conv_layer(layer1, 28, 56, [5, 5], [2, 2], name='conv_with_pool')
    conv_result = tf.reshape(layer2, [-1, 7 * 7 * 56])

    relu_layer_weight = tf.Variable(tf.truncated_normal([7 * 7 * 56, 1000], stddev=STDDEV * 2), name='relu_layer_weight')
    rely_layer_bias = tf.Variable(tf.truncated_normal([1000], stddev=STDDEV / 2), name='rely_layer_bias')
    relu_layer = tf.matmul(conv_result, relu_layer_weight) + rely_layer_bias
    relu_layer = tf.nn.relu(relu_layer)
    relu_layer = tf.nn.dropout(relu_layer, DROPOUT)

    final_layer_weight = tf.Variable(tf.truncated_normal([1000, CLASSES], stddev=STDDEV * 2), name='final_layer_weight')
    final_layer_bias = tf.Variable(tf.truncated_normal([CLASSES], stddev=STDDEV / 2), name='final_layer_bias')
    final_layer = tf.matmul(relu_layer, final_layer_weight) + final_layer_bias

    predicts = tf.nn.softmax(final_layer)
    predicts_for_log = tf.clip_by_value(predicts, 1e-9, 0.999999999)

    #crossEntropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

    loss = -tf.reduce_mean(tf.reduce_sum(real * tf.log(predicts_for_log) +
                                         (1 - real) * tf.log(1 - predicts_for_log),
                                         axis=1),
                           axis=0)
    #test = tf.reduce_sum(real * tf.log(predicts_for_log) + (1 - real) * tf.log(1 - predicts_for_log), axis=1)
    #loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=final_layer, labels=real))
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(real, axis=1), tf.argmax(predicts, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    confusion_matrix = tf.confusion_matrix(labels=tf.argmax(real, axis=1), predictions=tf.argmax(predicts, axis=1),
                                           num_classes=CLASSES)

    saver = tf.train.Saver()

    # dataset = get_mnist_dataset()
    dataset = get_fashion_dataset()

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        if restore:
            saver.restore(session, SAVE_PATH)

        if trainModel:
            train(input_images, real, session, optimiser, loss, accuracy, saver, dataset)

        if buildConfusionMatrix:
            test_cm = session.run(confusion_matrix, feed_dict={input_images: dataset.test_x, real: dataset.test_y})
            draw_confusion_matrix(test_cm)

        if buildClassifiedMatrix:
            all_probs = session.run(predicts, feed_dict={input_images: dataset.test_x, real: dataset.test_y})
            max_failure_picture_index = [[(-1, -1.0)] * CLASSES for _ in range(CLASSES)]
            for i in range(len(all_probs)):
                real = np.argmax(dataset.test_y[i])
                for j in range(CLASSES):
                    if max_failure_picture_index[real][j][1] < all_probs[i][j]:
                        max_failure_picture_index[real][j] = (i, all_probs[i][j])
            draw_max_failure_pictures(dataset.test_x, max_failure_picture_index)


if __name__ == '__main__':
    main(trainModel=True, restore=False, buildConfusionMatrix=True, buildClassifiedMatrix=True)
