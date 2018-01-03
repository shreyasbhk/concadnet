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
    # images = (255*layer.numpy()/tf.reduce_max(layer).numpy()).astype(np.uint8)
    images = (255*layer.numpy()).astype(np.uint16)
    # print(input.shape)
    # print(images.shape)
    # print(np.mean(images[0]))
    '''num_kernels = layer.numpy().shape[3]
    print(num_kernels)
    plt.figure(1, figsize=(20, 20))
    n_cols = 4
    n_rows = np.ceil(num_kernels/n_cols)+1
    col_concat = []
    concat = []
    for i in range(num_kernels):
        plt.subplot(n_rows, n_cols, i+1)
        plt.title('Filter'+str(i))
        plt.imshow(images[0, :, :, i], interpolation="nearest", cmap="gray")
        print(images[0])
    # print(images[0, :, :, 1].shape)
    for i in range(n_rows):
        col_concat.append(np.concatenate((images[0, :, :, i], images[0, :, :, i+1],
                                          images[0, :, :, i+2], images[0, :, :, i+3]), axis=1))
    for g in col_concat:
        np.concatenate(())'''
    # layer_shape = layer.numpy().shape[0], layer.numpy().shape[1]
    num_kernels = layer.numpy().shape[2]
    concat = np.squeeze(input, axis=2).astype(np.uint16)
    # cv2.imshow("input", concat)
    for i in range(num_kernels):
        concat = np.concatenate((concat, images[:, :, i]), axis=1)
    cv2.imshow('image', concat)
    cv2.waitKey(2)
    # cv2.destroyAllWindows()


class ConCaDNet(tfe.Network):
    """
    The ConCaDNet Model Class.
    """
    def __init__(self):
        super(ConCaDNet, self).__init__(name='')
        self.l1_1 = self.track_layer(tf.layers.Conv2D(16, 3, padding="SAME", activation=tf.nn.leaky_relu))
        self.fc1 = self.track_layer(tf.layers.Dense(units=2048))
        self.fc_out = self.track_layer(tf.layers.Dense(units=1))

    def call(self, inputs):
        x = inputs/tf.reduce_max(inputs)
        conv = self.l1_1(x)
        display_layer(inputs[0], conv[0])
        conv = tf.layers.flatten(conv)
        conv = self.fc1(conv)
        conv = self.fc_out(conv)
        return conv


def loss_function(model, x, y):
    preds = model(x)
    return tf.losses.sigmoid_cross_entropy(y, preds)


def evaluate(model):
    for (x, y, s, d) in tfe.Iterator(val_dataset):
        loss = loss_function(model, x, y).numpy()
        auc = roc_auc_score(y, model(x).numpy())
        return loss, auc


def train_one_epoch(model, optimizer, ds, log_interval=None):
    tf.train.get_or_create_global_step()
    batch_num = 0
    for (x, y, s, d) in tfe.Iterator(ds):
        batch_num += 1
        grads = tfe.implicit_gradients(loss_function)(model, x, y)
        optimizer.apply_gradients(grads)
        if batch_num%log_interval==0 and batch_num != 0:
                print(evaluate(model))


def train_model():
    model = ConCaDNet()
    optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
    for epoch in range(num_epochs):
        print("New Epoch")
        with tfe.restore_variables_on_create(tf.train.latest_checkpoint("../Models/")):
            global_step = tf.train.get_or_create_global_step()
            with tf.device("/GPU:0"):
                train_one_epoch(model, optimizer, dataset, log_interval=1)
        all_variables = (model.variables + optimizer.variables() + [global_step])
        tfe.Saver(all_variables).save("../Models/", global_step=global_step)
        # print(labels)


if __name__ == "__main__":
    train_model()

