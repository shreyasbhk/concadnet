'''
Created By Shreyas Hukkeri
Printed Saturday, January 21, 2018
'''
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_auc_score
import os
import cv2
import numpy as np
from tensorflow.contrib.eager.python import tfe

tfe.enable_eager_execution()

model_version = 15
run_number = "1_0"

batch_size = 350
val_batch_size = 500
learning_rate = 0.0003
num_epochs = 30

image_dimensions = (75, 75)
train_dataset_file = "../Data/train_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + "-m-only.tfrecords"
val_dataset_file = "../Data/val_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + "-m-only.tfrecords"
test_dataset_file = "../Data/test_"+str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + "-m-only.tfrecords"


def initialize_datasets():
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
        return dataset, val_dataset, test_dataset


def display_layer(input, layer, window_name):
    layer_size = layer.numpy().shape[0], layer.numpy().shape[1]
    input = cv2.resize(np.asarray(input), layer_size)
    input = np.expand_dims(input, axis=-1)
    layer = layer - np.min(layer)
    max_activation = np.max(layer)
    rescaled = layer/max_activation
    images = (65535*rescaled).numpy().astype(np.uint16)
    num_kernels = layer.numpy().shape[2]
    rows = 16
    cols = int(num_kernels/16)
    row_concats = []
    for i in range(cols):
        temp_concat = np.squeeze(input, axis=2).astype(np.uint16)
        for j in range(rows):
            temp_concat = np.concatenate((temp_concat, images[:, :, (i*16+j)]), axis=1)
        row_concats.append(temp_concat)
    final_concat = row_concats[0]
    for (i, rc) in enumerate(row_concats):
        if i !=0:
            final_concat = np.concatenate((final_concat, rc), axis=0)
    cv2.imshow(window_name, final_concat)
    cv2.waitKey(2)


class ConCaDNet(tfe.Network):
    """
    The ConCaDNet Model Class.
    """
    def __init__(self):
        super(ConCaDNet, self).__init__(name='')
        self.l1_1 = self.track_layer(tf.layers.Conv2D(96, 11, strides=1, padding="SAME", name="Conv_1_1",
                                                      activation=tf.nn.leaky_relu))
        self.l1_mp = self.track_layer(tf.layers.MaxPooling2D(3, strides=2, padding="SAME"))

        self.l2_1 = self.track_layer(tf.layers.Conv2D(256, 5, strides=1, padding="SAME", name="Conv_2_1",
                                                      activation=tf.nn.leaky_relu))
        self.l2_mp = self.track_layer(tf.layers.MaxPooling2D(3, strides=2, padding="SAME"))

        self.l3_1 = self.track_layer(tf.layers.Conv2D(384, 3, strides=1, padding="SAME", name="Conv_3_1",
                                                      activation=tf.nn.leaky_relu))
        self.l3_2 = self.track_layer(tf.layers.Conv2D(384, 3, strides=1, padding="SAME", name="Conv_3_2",
                                                      activation=tf.nn.leaky_relu))
        self.l3_3 = self.track_layer(tf.layers.Conv2D(256, 3, strides=1, padding="SAME", name="Conv_3_3",
                                                      activation=tf.nn.leaky_relu))
        self.l3_mp = self.track_layer(tf.layers.MaxPooling2D(3, strides=2, padding="SAME"))

        self.fc_out = self.track_layer(tf.layers.Dense(units=1))

    def display_layers(self, inputs, layers):
        # image_num = 1+int(np.round(np.random.random()*150))
        image_num = 1
        for (i, layer) in enumerate(layers):
            display_layer(inputs[image_num], layer[image_num], window_name="Layer "+str(i))

    def calculate_ranges(self, layers):
        temp_dict = {}
        for (i, layer) in enumerate(layers):
            layer = layer - np.min(layer)
            max_activation = np.max(layer)
            rescaled = layer / max_activation
            temp_ranges = {"0.1": 0, "0.2": 0, "0.3": 0, "0.4": 0, "0.5": 0,
                           "0.6": 0, "0.7": 0, "0.8": 0, "0.9": 0, "1": 0}
            for j in range(layer.numpy().shape[3]):
                r = np.max(rescaled[:, :, :, j]) - np.min(rescaled[:, :, :, j])
                if r < 0.1:
                    temp_ranges["0.1"] += 1
                elif r < 0.2:
                    temp_ranges["0.2"] += 1
                elif r < 0.3:
                    temp_ranges["0.3"] += 1
                elif r < 0.4:
                    temp_ranges["0.4"] += 1
                elif r < 0.5:
                    temp_ranges["0.5"] += 1
                elif r < 0.6:
                    temp_ranges["0.6"] += 1
                elif r < 0.7:
                    temp_ranges["0.7"] += 1
                elif r < 0.8:
                    temp_ranges["0.8"] += 1
                elif r < 0.9:
                    temp_ranges["0.9"] += 1
                else:
                    temp_ranges["1"] += 1
            temp_dict["Layer "+str(i)] = temp_ranges
        return temp_dict

    def call(self, inputs, display_image=False, training=False, return_ranges=False):
        x = ((inputs-tf.reduce_min(inputs))/tf.reduce_max(inputs))-0.5
        conv1 = self.l1_1(x)
        conv2 = self.l1_mp(conv1)
        conv3 = self.l2_1(conv2)
        conv4 = self.l2_mp(conv3)
        conv5 = self.l3_1(conv4)
        conv6 = self.l3_2(conv5)
        conv7 = self.l3_3(conv5)
        conv = self.l3_mp(conv7)
        self.display_layers(inputs, [conv1, conv3, conv5, conv6, conv7]) if display_image else None
        conv = tf.layers.flatten(conv)
        conv = tf.nn.dropout(conv, keep_prob=0.5) if training else conv
        conv = self.fc_out(conv)
        if return_ranges:
            return self.calculate_ranges([conv1, conv3, conv5, conv6, conv7])
        return conv


def loss(preds, labels):
    return tf.losses.sigmoid_cross_entropy(labels, preds)


def train_one_epoch(model, optimizer, epoch, log_interval=None):
    tf.train.get_or_create_global_step()
    def model_loss_auc(x, y):
        preds = model(x, display_image=False)
        loss_value = loss(preds, y)
        auc = roc_auc_score(y, preds)
        return loss_value.numpy(), auc

    def model_loss(x, y):
        preds = model(x, display_image=False, training=True)
        loss_value = loss(preds, y)
        return loss_value

    with tf.device("/GPU:0"):
        tr_ds, v_ds, t_ds = initialize_datasets()
        for (batch, (x, y, s, d)) in enumerate(tfe.Iterator(tr_ds)):
            grads = tfe.implicit_gradients(model_loss)(x, y)
            optimizer.apply_gradients(grads)
            if batch%log_interval == 0:
                evaluate(model, model_loss_auc(x, y), [v_ds, t_ds], epoch, batch)
            if batch%(log_interval*2) == 0:
                global_step = tf.train.get_or_create_global_step()
                all_variables = (model.variables + optimizer.variables() + [global_step])
                saver = tfe.Saver(all_variables)
                _ = saver.save("../Models/"+str(model_version)+"/"+str(run_number)+"/"+str(epoch)+"-"+str(batch))


def save_training_progress(vars):
    with open("../Models/"+str(model_version)+ "/"+str(run_number)+"/Training_Progress.txt", "a+") as f:
        f.write(vars+"\n")


def evaluate(model, train_values, datasets, epoch, batch):
    def get_ranges(x):
        return model(x, return_ranges=True)

    def model_loss_auc(x, y, display_image):
        preds = model(x, display_image=display_image)
        loss_value = loss(preds, y)
        auc = roc_auc_score(y, preds)
        return loss_value.numpy(), auc
    with tf.device("/GPU:0"):
        trl, tra = train_values
        val_dataset = datasets[0]
        test_dataset = datasets[1]
        val = tfe.Iterator(val_dataset)
        x, y, s, d = val.next()
        vl, va = model_loss_auc(x, y, display_image=False)
        test = tfe.Iterator(test_dataset)
        x, y, s, d = test.next()
        tl, ta = model_loss_auc(x, y, display_image=False)
        test = tfe.Iterator(test_dataset)
        x, y, s, d = test.next()
        ranges = get_ranges(x)
        for r in range(len(ranges)):
            print("Layer {} Ranges: {}".format(r, ranges["Layer "+str(r)]))
        print("Epoch: {}, Batch {}, Training Loss: {:.5f}, Training AUC: {:.2f}, " 
              "Validation Loss: {:.5f}, Validation AUC: {:.2f}, Testing Loss: {:.5f}, " 
              "Testing AUC: {:.2f}".format(epoch, batch, trl, tra*100, vl, va*100, tl, ta*100))
        save_training_progress("Epoch: {}, Batch {}, Training Loss: {:.5f}, Training AUC: {:.2f}, "
                               "Validation Loss: {:.5f}, Validation AUC: {:.2f}, Testing Loss: {:.5f}, "
                               "Testing AUC: {:.2f}, Layer Ranges: {}"
                               .format(epoch, batch, trl, tra*100, vl, va*100, tl, ta*100, ranges))


def train_model():
    model = ConCaDNet()
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    directory = "../Models/"+str(model_version)+"/"+str(run_number)+"/"
    if not os.path.exists(directory):
        os.makedirs(directory)
    summary_writer = tf.contrib.summary.create_file_writer(directory, flush_millis=10000)
    for epoch in range(1, num_epochs):
        print("Epoch #"+str(epoch))
        with tfe.restore_variables_on_create(tf.train.latest_checkpoint(directory)):
            with summary_writer.as_default():
                train_one_epoch(model, optimizer, epoch, log_interval=5)


def test_model():
    def model_loss_auc(model, x, y):
        preds = model(x, training=False)
        #loss_value = loss(preds, y)
        auc = roc_auc_score(y, preds)
        return auc
    model = ConCaDNet()
    tr_ds, v_ds, t_ds = initialize_datasets()
    with tfe.restore_variables_on_create("../Models/" + str(model_version) + "/" + str(run_number) + "/20-100"):
        with tf.device("/GPU:0"):
            for (x, y, s, d) in tfe.Iterator(t_ds):
                #print(len(x.numpy()))
                print(model_loss_auc(model, x, y))


if __name__ == "__main__":
    train_model()
    #test_model()

