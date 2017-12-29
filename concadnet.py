import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, dropout, fully_connected

batch_size = 100

image_dimensions = (100, 100)
train_dataset_file = "../Data/train_"+str(image_dimensions[0]) + str(image_dimensions[1]) + ".tfrecords"
val_dataset_file = "../Data/val_"+str(image_dimensions[0]) + str(image_dimensions[1]) + ".tfrecords"
test_dataset_file = "../Data/test_"+str(image_dimensions[0]) + str(image_dimensions[1]) + ".tfrecords"


with tf.device("/GPU:0"):
    def convnet(x, keep_prob):
        x = 2*(x/tf.reduce_max(x))
        conv = conv2d(x, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
        conv = flatten(conv)
        conv = fully_connected(conv, 256, activation_fn=None)
        conv = dropout(conv, keep_prob)
        conv = fully_connected(conv, 1, activation_fn=None)
        return conv


with tf.device("/cpu:0"):
    def parser_function(example_proto):
        features = {
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64, default_value=0),
            "image_view": tf.FixedLenFeature((), tf.int64, default_value=1),
            "breast_density": tf.FixedLenFeature((), tf.int64, default_value=1)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        return parsed_features["image"], parsed_features["label"], \
               parsed_features["image_view"], parsed_features["breast_density"]
    dataset = tf.data.TFRecordDataset(train_dataset_file)
    dataset = dataset.map(parser_function)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
    x = tf.placeholder(tf.uint8, shape=[None, image_dimensions[0], image_dimensions[1], 1], name="input")
    y = tf.placeholder(tf.int8, shape=[None, 1], name="label")
    keep_prob = tf.placeholder(tf.float16, name='keep_prob')

    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))




