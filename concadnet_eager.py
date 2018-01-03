'''
Runs on Calc and Mass Data
'''
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score
import time
import matplotlib.pyplot as plt
import cv2
import numpy as np
from tensorflow.contrib.eager.python import tfe

tfe.enable_eager_execution()

model_version = 4
run_number = 1
batch_size = 400
val_batch_size = 800
learning_rate = 0.001
num_epochs = 50

image_dimensions = (100, 100)
train_dataset_file = "../Data/train_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
val_dataset_file = "../Data/val_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"
test_dataset_file = "../Data/test_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords"


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
        label = tf.reshape(tf.cast(parsed_features["label"], tf.float32), [1])
        subtlety = tf.reshape(tf.cast(tf.one_hot(parsed_features["subtlety"], depth=5), tf.float32), [5])
        density = tf.reshape(tf.cast(tf.one_hot(parsed_features["density"], depth=4), tf.float32), [4])
        return image, label, subtlety, density
    dataset = tf.data.TFRecordDataset(train_dataset_file)
    dataset = dataset.map(parser_function)
    dataset = dataset.repeat(20)
    dataset = dataset.shuffle(batch_size*9)
    dataset = dataset.batch(batch_size)

    val_dataset = tf.data.TFRecordDataset(val_dataset_file)
    val_dataset = val_dataset.map(parser_function)
    val_dataset = val_dataset.repeat(1)
    val_dataset = val_dataset.shuffle(val_batch_size)
    val_dataset = val_dataset.batch(batch_size)


def print_progress(batch, auc, loss, val_auc, val_loss, images_per_second):
    print("Batch: {}, Training AUC: {:.2f} %, Training Loss: {:.4f}, Validation AUC: {:.2f} %, Validation "
          "Loss: {:.4f}, Images per second: {:.2f} img/sec".format(batch, auc*100, loss, val_auc*100, val_loss,
                                                           images_per_second))


def display_layer(input, layer):
    layer_size = layer.numpy().shape[0], layer.numpy().shape[1]
    input = cv2.resize(np.asarray(input), layer_size)
    input = np.expand_dims(input, axis=-1)
    avg_activation = np.mean(layer.numpy())
    print(avg_activation)
    images = ((avg_activation+layer.numpy())/avg_activation).astype(np.uint16)
    num_kernels = layer.numpy().shape[2]
    concat = np.squeeze(input, axis=2).astype(np.uint16)
    for i in range(num_kernels):
        concat = np.concatenate((concat, images[:, :, i]), axis=1)
    cv2.imshow('image', concat)
    cv2.waitKey(2)


class ConCaDNet(tfe.Network):
    """
    The ConCaDNet Model Class.
    """
    def __init__(self):
        super(ConCaDNet, self).__init__(name='')
        self.l1_1 = self.track_layer(tf.layers.Conv2D(16, 3, padding="SAME", name="Conv_1_1", activation=tf.nn.leaky_relu))
        self.l1_2 = self.track_layer(tf.layers.Conv2D(16, 3, padding="SAME", name="Conv_1_2", activation=tf.nn.leaky_relu))
        self.l1_3 = self.track_layer(tf.layers.Conv2D(16, 3, padding="SAME", name="Conv_1_3", activation=tf.nn.leaky_relu))
        self.l1_4 = self.track_layer(tf.layers.Conv2D(16, 3, padding="SAME", name="Conv_1_4", activation=tf.nn.leaky_relu))
        self.lmp = self.track_layer(tf.layers.MaxPooling2D(3, 2, padding="SAME", name="MaxPool_1"))
        self.l2_1 = self.track_layer(tf.layers.Conv2D(16, 5, padding="SAME", name="Conv_2_1", activation=tf.nn.leaky_relu))
        self.fc1 = self.track_layer(tf.layers.Dense(units=2048))
        self.fc_out = self.track_layer(tf.layers.Dense(units=1))

    def call(self, inputs, training=True):
        x = inputs/tf.reduce_max(inputs)
        l1_1 = self.l1_1(x)
        l1_2 = self.l1_2(l1_1)
        if not training:
            display_layer(inputs[0], l1_2[0])
        l1_3 = self.l1_3(l1_2)
        l1_4 = self.l1_4(l1_3)
        l1_mp = self.lmp(l1_4)
        conv = self.l2_1(l1_mp)
        conv = tf.layers.flatten(conv)
        conv = self.fc1(conv)
        conv = self.fc_out(conv)
        return conv


def loss(preds, labels):
    return tf.losses.sigmoid_cross_entropy(labels, preds)


def train_one_epoch(model, optimizer, ds, log_interval=None):
    tf.train.get_or_create_global_step()

    def model_loss(x, y):
        preds = model(x, training=True)
        loss_value = loss(preds, y)
        return loss_value
    with tf.device("/GPU:0"):
        for (batch, (x, y, s, d)) in enumerate(tfe.Iterator(ds)):
            grads = tfe.implicit_gradients(model_loss)(x, y)
            optimizer.apply_gradients(grads)
            if batch%log_interval == 0:
                evaluate(model)


def evaluate(model):
    def model_loss_auc(x, y):
        preds = model(x, training=False)
        loss_value = loss(preds, y)
        auc = roc_auc_score(y, preds)
        return loss_value, auc
    with tf.device("/GPU:0"):
        for (x, y, s, d) in tfe.Iterator(val_dataset):
            loss_value, auc = model_loss_auc(x, y)
            print("Model Loss: {}, Model AUC: {}".format(loss_value.numpy(), auc))


def train_model():
    model = ConCaDNet()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    summary_writer = tf.contrib.summary.create_file_writer("../Models/", flush_millis=10000)
    for epoch in range(num_epochs):
        print("New Epoch")
        with tfe.restore_variables_on_create(tf.train.latest_checkpoint("../Models/")):
            global_step = tf.train.get_or_create_global_step()
            with summary_writer.as_default():
                train_one_epoch(model, optimizer, dataset, log_interval=10)
        all_variables = (model.variables + optimizer.variables() + [global_step])
        tfe.Saver(all_variables).save("../Models/", global_step=global_step)
        # print(labels)


if __name__ == "__main__":
    train_model()
