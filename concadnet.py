'''
Runs on Calc and Mass Data
'''
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, dropout, fully_connected, batch_norm
from sklearn.metrics import confusion_matrix, roc_auc_score
import time

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

with tf.Session() as sess:
    x = tf.placeholder(tf.float32, shape=[None, image_dimensions[0], image_dimensions[1], 1], name="input")
    y = tf.placeholder(tf.int32, shape=[None, 1], name="label")
    s = tf.placeholder(tf.float32, shape=[None, 5], name="subtlety")
    d = tf.placeholder(tf.float32, shape=[None, 4], name="density")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    logits = convnet(x, s, d, keep_prob, reuse=False)
    val_logits = convnet(x, s, d, keep_prob, reuse=True)
    make_sense_logits = tf.sigmoid(val_logits)

    loss_op = tf.losses.sigmoid_cross_entropy(y, logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    auc = tf.metrics.auc(y, make_sense_logits, num_thresholds=20)
    saver = tf.train.Saver(max_to_keep=100)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    sess.run(iterator.initializer)
    batch = 0
    batch_loss = 0
    batch_auc = 0
    batch_number_progress = 0
    val_loss = 0
    val_batch = -1
    last_time = time.time()
    num_images = 0
    while True:
        batch += 1
        batch_number_progress += 1
        try:
            images, labels, su, de = sess.run(iterator.get_next())
            _, preds, loss = sess.run([train_op, make_sense_logits, loss_op], feed_dict={x: images,
                                                                                         y: labels,
                                                                                         s: su,
                                                                                         d: de,
                                                                                         keep_prob: 0.8})
            batch_loss += loss
            batch_auc += roc_auc_score(labels, preds)
            num_images += len(labels)

            if batch%10==0 and batch!=0:
                sess.run(val_iterator.initializer)
                val_batch = -1
                val_loss = 0
                val_auc = 0
                tn, fp, fn, tp = 0,0,0,0
                while True:
                    val_batch += 1
                    try:
                        val_images, val_labels, vsu, vde = sess.run(val_iterator.get_next())
                        preds, loss= sess.run([make_sense_logits, loss_op], feed_dict={x:val_images,
                                                                                       y: val_labels,
                                                                                       s: vsu,
                                                                                       d: vde,
                                                                                       keep_prob: 1.0})
                        #tn, fp, fn, tp = confusion_matrix(val_labels, preds)
                        val_loss += loss
                        val_auc += roc_auc_score(val_labels, preds)
                        num_images += len(val_labels)
                    except tf.errors.OutOfRangeError:
                        print_progress(batch, batch_auc/batch_number_progress, batch_loss/batch_number_progress, val_auc/val_batch,
                                       val_loss/val_batch, num_images/(time.time()-last_time))
                        num_images = 0
                        last_time = time.time()
                        break
                _ = saver.save(sess, save_path=str("../Models/Breast_Cancer/model-" + str(model_version)),
                               global_step=batch)
                if (val_loss / val_batch) > (2 * batch_loss / batch_number_progress):
                    break
                batch_auc = 0
                batch_loss = 0
                batch_number_progress = 0

        except tf.errors.OutOfRangeError:
            break
