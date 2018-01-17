'''
Runs on Calc and Mass Data
0.71126 AUC, Model Num 6
'''
import tensorflow as tf
from tensorflow.contrib.layers import conv2d, max_pool2d, flatten, dropout, fully_connected
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve, average_precision_score, precision_recall_curve
import os
import matplotlib.pyplot as plt
import numpy as np

model_version = 4
batch_size = 800

num_epochs = 2000
image_dimensions = (100, 100)
test_dataset_file = "../Data/test_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"


with tf.device("/GPU:0"):
    def convnet(x, keep_prob, reuse):
        with tf.variable_scope('ConvNet', reuse=reuse):
            x = 5 * (x / tf.reduce_max(tf.reduce_max(x)))
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
            "label": tf.FixedLenFeature((), tf.int64, default_value=0)
        }
        parsed_features = tf.parse_single_example(example_proto, features)
        image = tf.reshape(tf.decode_raw(parsed_features["image"], tf.float32),
                           [image_dimensions[0], image_dimensions[1], 1])
        label = tf.reshape(tf.cast(parsed_features["label"], tf.int32), [1])
        return image, label
    dataset = tf.data.TFRecordDataset(test_dataset_file)
    dataset = dataset.map(parser_function)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_initializable_iterator()

def test_model(epoch_number):
    with tf.Session() as sess:
        x = tf.placeholder(tf.float32, shape=[None, image_dimensions[0], image_dimensions[1], 1], name="input")
        y = tf.placeholder(tf.int32, shape=[None, 1], name="label")
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        test_logits = convnet(x, keep_prob, reuse=tf.AUTO_REUSE)
        make_sense_logits = tf.nn.sigmoid(test_logits)
        #auc = tf.metrics.auc(y, make_sense_logits)

        saver = tf.train.Saver()
        saver.restore(sess, save_path=str("../Models/Breast_Cancer/Model 4/model-"+str(model_version)+"-run-1-"+str(
            epoch_number)))
        sess.run(tf.local_variables_initializer())
        sess.run(iterator.initializer)
        test_auc = 0
        test_batch = -1
        fpr, tpr = [], []
        while True:
            test_batch += 1
            try:
                images, labels = sess.run(iterator.get_next())
                preds= sess.run(make_sense_logits, feed_dict={x: images,
                                                                         y: labels,
                                                                         keep_prob: 1})
                test_auc += roc_auc_score(labels, preds)
                print(average_precision_score(labels, preds))
                print(precision_recall_curve(labels, preds, pos_label=1))
                #print(roc_curve(labels, preds))
                '''fpr, tpr, _ = roc_curve(labels, preds)
                plt.figure()
                lw = 2
                plt.plot(fpr, tpr, color='darkorange',
                         lw=lw, label='ROC curve')
                plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve of ConCaDNet')
                plt.legend(loc="lower right")
                plt.show()'''
            except tf.errors.OutOfRangeError:
                return test_auc/test_batch

if __name__ == "__main__":
    max_accuracy_auc = 0
    best_model_num = 0
    for epoch in range(num_epochs):
        model_num = epoch
        if os.path.isfile(str("../Models/Breast_Cancer/Model 4/model-"+str(model_version)+"-run-1-"+str(
                model_num)+".meta")):
            auc = test_model(model_num)
            print(auc)
            if auc > max_accuracy_auc:
                max_accuracy_auc = auc
                best_model_num = model_num
    print("Best Model AUC: ")
    print(max_accuracy_auc)
    print("Best Model: ")
    print(best_model_num)
