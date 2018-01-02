'''
Runs on Calc and Mass Data
'''
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, dropout, fully_connected, batch_norm
from sklearn.metrics import confusion_matrix, roc_auc_score
import time
from tensorflow.contrib.eager.python import tfe

tfe.enable_eager_execution()

model_version = 4
run_number = 1
batch_size = 400
val_batch_size = 400
learning_rate = 0.001
num_epochs = 50

image_dimensions = (100, 100)
train_dataset_file = "../Data/train_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
val_dataset_file = "../Data/val_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
test_dataset_file = "../Data/test_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"

def print_progress(batch, auc, loss, val_auc, val_loss, images_per_second):
    print("Batch: {}, Training AUC: {:.2f} %, Training Loss: {:.4f}, Validation AUC: {:.2f} %, Validation "
          "Loss: {:.4f}, Images per second: {:.2f} img/sec".format(batch, auc*100, loss, val_auc*100, val_loss,
                                                           images_per_second))


with tf.device("/GPU:0"):
    def convnet(x, s, d, keep_prob, reuse):
        with tf.variable_scope('ConvNet', reuse=reuse):
            x = (x / tf.reduce_max(tf.reduce_max(x)))
            conv = conv2d(x, 16, (3, 3), stride=1, activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), stride=1, activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), stride=1, activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), stride=1, activation_fn=tf.nn.leaky_relu)
            conv1 = max_pool2d(conv, (3, 3), (2, 2))
            conv = conv2d(x, 16, (5, 5), stride=1, activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (5, 5), stride=1, activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (5, 5), stride=1, activation_fn=tf.nn.leaky_relu)
            conv2 = max_pool2d(conv, (3, 3), (2, 2))
            concat = tf.concat([conv1, conv2], axis=3)

            conv = conv2d(concat, 16, (3, 3), stride=1, activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), stride=1, activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), stride=1, activation_fn=tf.nn.leaky_relu)
            conv1 = max_pool2d(conv, (3, 3), (2, 2))
            conv = conv2d(concat, 16, (5, 5), stride=1, activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (5, 5), stride=1, activation_fn=tf.nn.leaky_relu)
            conv2 = max_pool2d(conv, (3, 3), (2, 2))
            concat = tf.concat([conv1, conv2], axis=3)

            conv = conv2d(concat, 16, (3, 3), stride=1, activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), stride=1, activation_fn=tf.nn.leaky_relu)
            conv1 = max_pool2d(conv, (3, 3), (2, 2))
            conv = conv2d(concat, 16, (5, 5), stride=1, activation_fn=tf.nn.leaky_relu)
            conv2 = max_pool2d(conv, (3, 3), (2, 2))
            concat = tf.concat([conv1, conv2], axis=3)
            print(concat)
            conv = flatten(concat)
            conv = fully_connected(conv, 256, activation_fn=None)
            conv = dropout(conv, keep_prob)
            conv = fully_connected(conv, 1, activation_fn=None)
        return conv

with tf.device("/cpu:0"):
    def parser_function(example_proto):
        features = {
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64),
            "subtlety": tf.FixedLenFeature((), tf.int64),
            "density": tf.FixedLenFeature((), tf.int64)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.reshape(tf.decode_raw(parsed_features["image"], tf.float32),
                           [image_dimensions[0], image_dimensions[1], 1])
        label = tf.reshape(tf.cast(parsed_features["label"], tf.int32), [1])
        subtlety = tf.reshape(tf.cast(tf.one_hot(parsed_features["subtlety"], depth=5), tf.float32), [5])
        density = tf.reshape(tf.cast(tf.one_hot(parsed_features["density"], depth=4), tf.float32), [4])
        return image, label, subtlety, density
    dataset = tf.data.TFRecordDataset(train_dataset_file)
    dataset = dataset.map(parser_function)
    dataset = dataset.repeat(20)
    dataset = dataset.shuffle(batch_size*9)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()

    val_dataset = tf.data.TFRecordDataset(val_dataset_file)
    val_dataset = val_dataset.map(parser_function)
    val_dataset = val_dataset.repeat(1)
    val_dataset = val_dataset.shuffle(val_batch_size)
    val_dataset = val_dataset.batch(batch_size)
    val_iterator = val_dataset.make_initializable_iterator()

for elements in tfe.Iterator(iterator):
    print(elements)

