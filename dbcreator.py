import tensorflow as tf
import dicom
import numpy as np
import csv
import cv2
import os

image_dimensions = [100, 100]

training_csv_file = "../Data/mass_case_description_train_set.csv"
testing_csv_file = "../Data/mass_case_description_test_set.csv"
images_location = ""


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_image_file(filename):
    image = dicom.read_file(filename)
    image = np.resize(image, [100, 100])
    print(image)
    return image


def read_csv_file(filename):
    def sort_row(row):
        if row[9] == "MALIGNANT":
            label = 1
        else:
            label = 0
        if row[3] == "MLO":
            image_view = 1
        else:
            image_view = 0
        return row[12], label, image_view, row[1]
    patients = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)
        for row in reader:
            image_path, label, image_view, breast_density = sort_row(row)
            patients.append([image_path, label, image_view, breast_density])
    return patients


def create_train_val_database(patients_data, val_patients_data):
    writer = tf.python_io.TFRecordWriter(
        "Data/train_"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".tfrecords")
    for i in range(len(patients_data)):
        image = read_image_file("../Data/DOI/"+str(patients_data[i][0])).toString()
        label = patients_data[i][1]
        image_view = patients_data[i][2]
        density = patients_data[i][3]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image),
            'label': _int64_feature(label),
            'image_view': _int64_feature(image_view),
            'breast_density': _int64_feature(density)
        }))
        writer.write(example.SerializeToString())
    writer.close()
    writer = tf.python_io.TFRecordWriter(
        "Data/val_" + str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords")
    for i in range(len(val_patients_data)):
        image = read_image_file("../Data/DOI/"+str(val_patients_data[i][0])).toString()
        label = val_patients_data[i][1]
        image_view = val_patients_data[i][2]
        density = val_patients_data[i][3]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image),
            'label': _int64_feature(label),
            'image_view': _int64_feature(image_view),
            'breast_density': _int64_feature(density)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def create_test_database(patients_data):
    writer = tf.python_io.TFRecordWriter(
        "Data/test_" + str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords")
    for i in range(len(patients_data)):
        image = read_image_file("../Data/DOI/"+str(patients_data[i][0])).toString()
        label = patients_data[i][1]
        image_view = patients_data[i][2]
        density = patients_data[i][3]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image),
            'label': _int64_feature(label),
            'image_view': _int64_feature(image_view),
            'breast_density': _int64_feature(density)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def patients_sequencer(filename, type="training"):
    if type == "training":
        train_patients, val_patients = [], []
        patients = read_csv_file(filename)
        num_examples = len(patients)
        random_array = np.arange(num_examples)
        np.random.shuffle(random_array)
        for i in range(num_examples):
            if i < num_examples*0.8:
                image_path = patients[random_array[i]][0]
                label = patients[random_array[i]][1]
                image_view = patients[random_array[i]][2]
                density = patients[random_array[i]][3]
                train_patients.append([image_path, label, image_view, density])
            else:
                image_path = patients[random_array[i]][0]
                label = patients[random_array[i]][1]
                image_view = patients[random_array[i]][2]
                density = patients[random_array[i]][3]
                val_patients.append([image_path, label, image_view, density])
        return train_patients, val_patients
    else:
        test_patients = []
        patients = read_csv_file(filename)
        num_examples = len(patients)
        for i in range(num_examples):
            image_path = patients[i][0]
            label = patients[i][1]
            image_view = patients[i][2]
            density = patients[i][3]
            test_patients.append([image_path, label, image_view, density])
        return test_patients


if __name__ == '__main__':
    train_patients, val_patients = patients_sequencer(training_csv_file, type="training")
    test_patients = patients_sequencer(testing_csv_file, type="testing")
    create_train_val_database(train_patients, val_patients)
    create_test_database(test_patients)
