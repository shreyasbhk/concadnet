'''
ConCaDNet V2.0 - Eager
'''
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score
import time
import cv2
import numpy as np
from tensorflow.contrib.eager.python import tfe

tfe.enable_eager_execution()

model_version = 7
run_number = 1

batch_size = 450
val_batch_size = 800
learning_rate = 0.0003
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
    dataset = dataset.repeat(1)
    dataset = dataset.shuffle(batch_size*3)
    dataset = dataset.batch(batch_size)

    val_dataset = tf.data.TFRecordDataset(val_dataset_file)
    val_dataset = val_dataset.map(parser_function)
    val_dataset = val_dataset.repeat(1)
    #val_dataset = val_dataset.shuffle(val_batch_size)
    val_dataset = val_dataset.batch(val_batch_size)

    test_dataset = tf.data.TFRecordDataset(test_dataset_file)
    test_dataset = test_dataset.map(parser_function)
    test_dataset = test_dataset.repeat(1)
    #test_dataset = test_dataset.shuffle(val_batch_size)
    test_dataset = test_dataset.batch(val_batch_size)


def display_layer(input, layer, window_name):
    layer_size = layer.numpy().shape[0], layer.numpy().shape[1]
    input = cv2.resize(np.asarray(input), layer_size)
    input = np.expand_dims(input, axis=-1)
    layer = layer - np.min(layer)
    max_activation = np.max(layer)
    images = (65535*(layer/max_activation)).numpy().astype(np.uint16)
    num_kernels = layer.numpy().shape[2]
    concat = np.squeeze(input, axis=2).astype(np.uint16)
    for i in range(num_kernels):
        concat = np.concatenate((concat, images[:, :, i]), axis=1)
    cv2.imshow(window_name, concat)
    cv2.waitKey(2)


class ConCaDNet(tfe.Network):
    """
    The ConCaDNet Model Class.
    """
    def __init__(self):
        super(ConCaDNet, self).__init__(name='')
        self.l1_1 = self.track_layer(tf.layers.Conv2D(8, 3, padding="SAME", name="Conv_1_1", activation=tf.nn.leaky_relu))
        self.l1_2 = self.track_layer(tf.layers.Conv2D(8, 3, padding="SAME", name="Conv_1_2", activation=tf.nn.leaky_relu))
        self.l1_3 = self.track_layer(tf.layers.Conv2D(8, 3, padding="SAME", name="Conv_1_3", activation=tf.nn.leaky_relu))
        self.l1_4 = self.track_layer(tf.layers.Conv2D(8, 3, padding="SAME", name="Conv_1_4", activation=tf.nn.leaky_relu))
        self.l1_mp = self.track_layer(tf.layers.MaxPooling2D(3, 2, padding="SAME", name="MaxPool_1"))
        self.l2_1 = self.track_layer(tf.layers.Conv2D(8, 5, padding="SAME", name="Conv_2_1", activation=tf.nn.leaky_relu))
        self.l2_2 = self.track_layer(tf.layers.Conv2D(8, 5, padding="SAME", name="Conv_2_2", activation=tf.nn.leaky_relu))
        self.l2_3 = self.track_layer(tf.layers.Conv2D(8, 5, padding="SAME", name="Conv_2_3", activation=tf.nn.leaky_relu))
        self.l2_4 = self.track_layer(tf.layers.Conv2D(8, 5, padding="SAME", name="Conv_2_4", activation=tf.nn.leaky_relu))
        self.l2_mp = self.track_layer(tf.layers.MaxPooling2D(3, 2, padding="SAME", name="MaxPool_2"))
        self.l3_1 = self.track_layer(tf.layers.Conv2D(5, 3, padding="SAME", name="Conv_3_1", activation=tf.nn.leaky_relu))
        self.l3_2 = self.track_layer(tf.layers.Conv2D(5, 3, padding="SAME", name="Conv_3_2", activation=tf.nn.leaky_relu))
        self.l3_3 = self.track_layer(tf.layers.Conv2D(5, 3, padding="SAME", name="Conv_3_3", activation=tf.nn.leaky_relu))
        self.l3_4 = self.track_layer(tf.layers.Conv2D(5, 3, padding="SAME", name="Conv_3_4", activation=tf.nn.leaky_relu))
        self.l3_mp = self.track_layer(tf.layers.MaxPooling2D(3, 2, padding="SAME", name="MaxPool_3"))
        self.l4_1 = self.track_layer(tf.layers.Conv2D(16, 5, padding="SAME", name="Conv_4_1", activation=tf.nn.leaky_relu))
        self.l4_2 = self.track_layer(tf.layers.Conv2D(16, 5, padding="SAME", name="Conv_4_2", activation=tf.nn.leaky_relu))
        self.l4_3 = self.track_layer(tf.layers.Conv2D(16, 5, padding="SAME", name="Conv_4_3", activation=tf.nn.leaky_relu))
        self.l4_4 = self.track_layer(tf.layers.Conv2D(16, 5, padding="SAME", name="Conv_4_4", activation=tf.nn.leaky_relu))
        self.l4_mp = self.track_layer(tf.layers.MaxPooling2D(3, 2, padding="SAME", name="MaxPool_4"))
        self.fc1 = self.track_layer(tf.layers.Dense(units=2048))
        self.fc_out = self.track_layer(tf.layers.Dense(units=1))

    def display_layers(self, inputs, layers):
        for (i, layer) in enumerate(layers):
            display_layer(inputs[1], layer[1], window_name="Layer "+str(i))

    def call(self, inputs, training=True):
        x = inputs/tf.reduce_max(inputs)
        l1_1 = self.l1_1(x)
        l1_2 = self.l1_2(l1_1)
        l1_3 = self.l1_3(l1_2)
        l1_4 = self.l1_4(l1_3)
        l1_mp = self.l1_mp(l1_4)
        l2_1 = self.l2_1(x)
        l2_2 = self.l2_2(l2_1)
        l2_3 = self.l2_3(l2_2)
        l2_4 = self.l2_4(l2_3)
        l2_mp = self.l2_mp(l2_4)
        conv = tf.concat([l1_mp, l2_mp], axis=3)
        # l3_1 = self.l3_1(l2_mp)
        # l3_2 = self.l3_2(l3_1)
        # l3_3 = self.l3_3(l3_2)
        # l3_4 = self.l3_4(l3_3)
        # l3_mp = self.l3_mp(l3_4)
        l4_1 = self.l4_1(conv)
        l4_2 = self.l4_2(l4_1)
        l4_3 = self.l4_3(l4_2)
        l4_4 = self.l4_4(l4_3)
        conv = self.l4_mp(l4_4)
        # conv = tf.concat([l3_mp, l4_mp], axis=3)
        conv = tf.layers.flatten(conv)
        conv = self.fc1(conv)
        conv = self.fc_out(conv)

        if not training:
            self.display_layers(inputs, [l1_1, l1_2, l1_3, l1_4, l2_1, l2_2, l2_3, l2_4,
                                         # l3_1, l3_2, l3_3, l3_4,
                                         l4_1, l4_2, l4_3, l4_4])
        return conv


def loss(preds, labels):
    return tf.losses.sigmoid_cross_entropy(labels, preds)


def train_one_epoch(model, optimizer, ds, epoch, log_interval=None):
    tf.train.get_or_create_global_step()

    def model_loss_auc(x, y):
        preds = model(x, training=True)
        loss_value = loss(preds, y)
        auc = roc_auc_score(y, preds)
        return loss_value.numpy(), auc

    def model_loss(x, y):
        preds = model(x, training=True)
        loss_value = loss(preds, y)
        return loss_value

    with tf.device("/GPU:0"):
        for (batch, (x, y, s, d)) in enumerate(tfe.Iterator(ds)):
            grads = tfe.implicit_gradients(model_loss)(x, y)
            optimizer.apply_gradients(grads)
            if batch%log_interval == 0:
                evaluate(model, model_loss_auc(x, y))
                all_variables = (model.variables + optimizer.variables())
                saver = tfe.Saver(all_variables)
                _ = saver.save("../Models/"+str(model_version)+"/"+str(run_number)+"/"+str(epoch)+"-"+str(batch))


def evaluate(model, train_values):
    def model_loss_auc(x, y):
        preds = model(x, training=False)
        loss_value = loss(preds, y)
        auc = roc_auc_score(y, preds)
        return loss_value.numpy(), auc
    with tf.device("/GPU:0"):
        trl, tra = train_values
        vl, va, tl, ta = 0, 0, 0, 0
        for (x, y, s, d) in tfe.Iterator(val_dataset):
            vl, va = model_loss_auc(x, y)
        for (x, y, s, d) in tfe.Iterator(test_dataset):
            tl, ta = model_loss_auc(x, y)
        print("Training Loss: {:.5f}, Training AUC: {:.2f}, Validation Loss: {:.5f}, Validation AUC: {:.2f}, "
              "Testing Loss: {:.5f}, Testing AUC: {:.2f}".format(trl, tra*100, vl, va*100, tl, ta*100))


def train_model():
    model = ConCaDNet()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    summary_writer = tf.contrib.summary.create_file_writer("../Models/"+str(model_version)+"/"+str(run_number)+"/", flush_millis=10000)
    for epoch in range(num_epochs):
        print("Epoch #"+str(epoch))
        with tfe.restore_variables_on_create(tf.train.latest_checkpoint("../Models/"+str(model_version)+
                                                                        "/"+str(run_number)+"/")):
            with summary_writer.as_default():
                train_one_epoch(model, optimizer, dataset, epoch, log_interval=10)
        # print(labels)


def test_model():
    def model_loss_auc(model, x, y):
        preds = model(x, training=False)
        #loss_value = loss(preds, y)
        auc = roc_auc_score(y, preds)
        return auc
    model = ConCaDNet()
    with tfe.restore_variables_on_create("../Models/" + str(model_version) + "/" + str(run_number) + "/9000"):
        with tf.device("/GPU:0"):
            for (x, y, s, d) in tfe.Iterator(test_dataset):
                #print(len(x.numpy()))
                print(model_loss_auc(model, x, y))


if __name__ == "__main__":
    train_model()
    #test_model()

