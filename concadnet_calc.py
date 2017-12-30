import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, dropout, fully_connected
from sklearn.metrics import confusion_matrix, roc_auc_score

model_version = 2
batch_size = 100
val_batch_size = 100
learning_rate = 0.001
num_epochs = 200

image_dimensions = (100, 100)
train_dataset_file = "../Data/calc_train_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
val_dataset_file = "../Data/calc_val_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
test_dataset_file = "../Data/calc_test_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"


def print_progress(epoch, batch, auc, loss, val_auc, val_loss):
    print("Epoch: {}, Batch: {}, Training AUC: {:.2f} %, Training Loss: {:.4f}, Validation AUC: {:.2f} %, Validation "
          "Loss: {:.4f}".format(epoch, batch, auc*100, loss, val_auc*100, val_loss))

with tf.device("/GPU:0"):
    def convnet(x, keep_prob, reuse):
        with tf.variable_scope('ConvNet', reuse=reuse):
            x = 5 * (x / tf.reduce_max(x))
            conv = conv2d(x, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv1 = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)

            conv = conv2d(x, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv2 = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)

            concat = tf.concat([conv1, conv2], axis=1)
            concat = max_pool2d(concat, (3, 3), (2, 2))

            conv = conv2d(concat, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv1 = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)

            conv = conv2d(concat, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)
            conv2 = conv2d(conv, 16, (3, 3), activation_fn=tf.nn.leaky_relu)

            concat = tf.concat([conv1, conv2], axis=1)
            concat = max_pool2d(concat, (3, 3), (2, 2))

            conv = flatten(concat)
            conv = fully_connected(conv, 256, activation_fn=None)
            conv = dropout(conv, keep_prob)
            conv = fully_connected(conv, 1, activation_fn=None)
        return conv

with tf.device("/cpu:0"):
    def parser_function(example_proto):
        features = {
            "image": tf.FixedLenFeature((), tf.string, default_value=""),
            "label": tf.FixedLenFeature((), tf.int64, default_value=0)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.reshape(tf.decode_raw(parsed_features["image"], tf.uint16),
                           [image_dimensions[0], image_dimensions[1], 1])
        label = tf.reshape(tf.cast(parsed_features["label"], tf.int32), [1])
        return image, label
    dataset = tf.data.TFRecordDataset(train_dataset_file)
    dataset = dataset.map(parser_function)
    dataset = dataset.repeat(1)
    dataset = dataset.shuffle(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()

    val_dataset = tf.data.TFRecordDataset(test_dataset_file)
    val_dataset = val_dataset.map(parser_function)
    val_dataset = val_dataset.repeat(1)
    val_dataset = val_dataset.shuffle(val_batch_size)
    val_dataset = val_dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
    val_iterator = val_dataset.make_initializable_iterator()

with tf.Session() as sess:
    x = tf.placeholder(tf.uint16, shape=[None, image_dimensions[0], image_dimensions[1], 1], name="input")
    y = tf.placeholder(tf.int32, shape=[None, 1], name="label")
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    logits = convnet(x, keep_prob, reuse=False)
    val_logits = convnet(x, keep_prob, reuse=True)
    make_sense_logits = tf.round(tf.sigmoid(val_logits))

    loss_op = tf.losses.sigmoid_cross_entropy(y, logits)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    auc = tf.metrics.auc(y, make_sense_logits, num_thresholds=20)
    vauc = tf.metrics.auc(y, make_sense_logits, num_thresholds=20)
    saver = tf.train.Saver(max_to_keep=40)
    sess.run(tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()))
    for epoch in range(num_epochs):
        sess.run(iterator.initializer)
        batch = -1
        epoch_loss = 0
        epoch_auc = 0
        val_loss = 0
        val_batch = -1
        while True:
            batch += 1
            try:
                images, labels = sess.run(iterator.get_next())
                _, preds, loss = sess.run([train_op, make_sense_logits, loss_op], feed_dict={x: images,
                                                                   y: labels,
                                                                   keep_prob: 0.8})
                epoch_loss += loss
                epoch_auc += roc_auc_score(labels, preds)
                #epoch_auc += preds[0]
                if batch%10==0 and batch!=0:
                    sess.run(val_iterator.initializer)
                    val_batch = -1
                    val_loss = 0
                    val_auc = 0
                    tn, fp, fn, tp = 0,0,0,0
                    while True:
                        val_batch += 1
                        try:
                            val_images, val_labels = sess.run(val_iterator.get_next())
                            preds, loss= sess.run([make_sense_logits, loss_op], feed_dict={x:val_images,
                                                                                           y: val_labels,
                                                                                           keep_prob: 1.0})
                            #tn, fp, fn, tp = confusion_matrix(val_labels, preds)
                            val_loss += loss
                            val_auc += roc_auc_score(val_labels, preds)
                            #val_auc += preds[0]
                        except tf.errors.OutOfRangeError:
                            print_progress(epoch, batch, epoch_auc/batch, epoch_loss/batch, val_auc/val_batch,
                                           val_loss/val_batch)
                            break
            except tf.errors.OutOfRangeError:
                if epoch%5==0:
                    _ = saver.save(sess, save_path=str("../Models/Breast_Cancer/model-"+str(model_version)),
                                   global_step=epoch)
                break
        if val_loss / val_batch > 4 * epoch_loss / batch:
            break
