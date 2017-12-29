import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, dropout, fully_connected

batch_size = 100
learning_rate = 0.00001
num_epochs = 50

image_dimensions = (100, 100)
train_dataset_file = "../Data/train_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
val_dataset_file = "../Data/val_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
test_dataset_file = "../Data/test_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"


def print_progress(epoch, batch, auc, loss):
    print("Epoch: {}, Batch: {}, Training AUC: {}, Training Loss: {}".format(epoch, batch, auc, loss))

with tf.device("/GPU:0"):
    def convnet(x, keep_prob, reuse, is_training):
        with tf.variable_scope('ConvNet', reuse=reuse):
            x = 5*(x/tf.reduce_max(x))
            conv = conv2d(x, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv1 = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)

            conv = conv2d(x, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv2 = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)

            concat = tf.concat([conv1, conv2], axis=1)
            concat = max_pool2d(concat, (2, 2), (2, 2))

            conv = conv2d(concat, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv1 = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)

            conv = conv2d(concat, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv2 = conv2d(conv, 8, (3, 3), activation_fn=tf.nn.leaky_relu)

            conv = tf.concat([conv1, conv2], axis=1)
            conv = max_pool2d(conv, (2, 2), (2, 2))

            conv = flatten(conv)
            conv = fully_connected(conv, 512, activation_fn=None)
            conv = dropout(conv, keep_prob)
            conv = fully_connected(conv, 2, activation_fn=None)
            conv = tf.sigmoid(conv) if not is_training else conv
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
        image = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint16),
                           [image_dimensions[0], image_dimensions[1], 1])
        label = tf.one_hot(tf.cast(parsed_features["label"], tf.int32), 2, on_value=1, off_value=0)
        image_view = tf.reshape(tf.cast(parsed_features["image_view"], tf.int8), [1])
        breast_density = tf.reshape(tf.cast(parsed_features["breast_density"], tf.int8), [1])
        return {"image": image, "image_view": image_view, "breast_density": breast_density}, label
    dataset = tf.data.TFRecordDataset(train_dataset_file)
    dataset = dataset.map(parser_function)
    dataset = dataset.repeat(3)
    dataset = dataset.shuffle(batch_size)
    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    iterator = dataset.make_initializable_iterator()

with tf.Session() as sess:
    x = tf.placeholder(tf.uint16, shape=[None, image_dimensions[0], image_dimensions[1], 1], name="input")
    y = tf.placeholder(tf.int8, shape=[None, 2], name="label")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    logits = convnet(x, keep_prob, reuse=False, is_training=True)
    test_logits = convnet(x, keep_prob, reuse=True, is_training=False)

    loss_op = tf.losses.sigmoid_cross_entropy(y, logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    auc = tf.metrics.auc(y, test_logits, num_thresholds=20)
    # confusion_matrix = tf.confusion_matrix(y, test_logits, num_classes=1)
    saver = tf.train.Saver()
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    for epoch in range(num_epochs):
        sess.run(iterator.initializer)
        batch = 0
        epoch_loss = 0
        epoch_auc = 0
        while True:
            batch += 1
            try:
                features, labels = iterator.get_next()
                images = sess.run(features["image"])
                labels = sess.run(labels)
                _, loss, a = sess.run([train_op, loss_op, auc], feed_dict={x: images,
                                                                           y: labels,
                                                                           keep_prob: 0.8})
                epoch_auc += a[0]
                epoch_loss += loss
                if batch%10==0:print_progress(epoch, batch, epoch_auc/batch, epoch_loss/batch)
            except tf.errors.OutOfRangeError or tf.errors.InvalidArgumentError:
                break
