import tensorflow as tf
import dicom
import numpy as np
import csv
import cv2
import os

image_dimensions = (100, 100)

training_csv_file_m = "../Data/mass_case_description_train_set.csv"
training_csv_file_c = "../Data/calc_case_description_train_set.csv"
testing_csv_file_m = "../Data/mass_case_description_test_set.csv"
testing_csv_file_c = "../Data/calc_case_description_test_set.csv"
images_location = ""


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def display_image(image):
    image = image.astype(np.uint16)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def add_random_augmentation(image):
    row, col = image_dimensions[0], image_dimensions[1]
    gaussian = np.random.random((row, col, 1)).astype(np.float32)
    image = cv2.addWeighted(image, 0.75, 0.25 * gaussian, 0.25, 0)
    rotation_angle = np.random.choice(4, 1)[0]*90
    m = cv2.getRotationMatrix2D((col / 2, row / 2), rotation_angle, 1)
    image = cv2.warpAffine(image, m, (col, row))
    zoom_factor_x = 1+np.random.random()
    zoom_factor_y = 1+np.random.random()
    image = cv2.resize(image,image_dimensions, fx=zoom_factor_x, fy=zoom_factor_y, interpolation=0)
    #display_image(image)
    return image

def add_augmentation(image, rotation_angle=0, flip_horizontal=False, flip_vertical=False, gaussian=True, zoom=False):
    row, col = image_dimensions[0], image_dimensions[1]
    if gaussian:
        gauss = np.random.random((row, col, 1)).astype(np.float32)
        image = cv2.addWeighted(image, 0.75, 0.25 * gauss, 0.25, 0)
    m = cv2.getRotationMatrix2D((col / 2, row / 2), rotation_angle, 1)
    image = cv2.warpAffine(image, m, (col, row))
    if zoom:
        zoom_factor_x = 1+(np.random.random()/2)
        zoom_factor_y = 1+(np.random.random()/2)
        image = cv2.resize(image,image_dimensions, fx=zoom_factor_x, fy=zoom_factor_y, interpolation=0)
    if flip_horizontal:
        image = cv2.flip(image, 0)
    if flip_vertical:
        image = cv2.flip(image, 1)
    #display_image(image)
    return image

def read_image_file(filename):
    dicom_file = dicom.read_file(filename)
    image = dicom_file.pixel_array
    image = image.astype(np.float32)
    image = cv2.resize(image, image_dimensions)
    image = np.expand_dims(image, axis=-1)
    return image

def read_csv_file(filenames):
    patients = []
    for filename in filenames:
        def sort_row(row):
            if row[9] == "MALIGNANT":
                label = 1
            else:
                label = 0
            if ((os.path.getsize(str("../Data/DOI/"+str(row[12].replace("\n", '')))))/(1024.*1024)) > 2:
                path = row[13].replace("\n", '')
            else:
                path = row[12].replace("\n", '')
            subtlety = int(row[10])
            b_density = int(row[1])
            return path, label, subtlety, b_density
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            next(reader, None)
            for row in reader:
                image_path, label, subtlety, density = sort_row(row)
                patients.append([image_path, label, subtlety, density])
    print(len(patients))
    return patients


def write_image(writer, image, patient):
    image = image.tobytes()
    label = patient[1]
    subtlety = patient[2]
    density = patient[3]
    example = tf.train.Example(features=tf.train.Features(feature={
        'image': _bytes_feature(image),
        'label': _int64_feature(label),
        'subtlety': _int64_feature(subtlety),
        'density': _int64_feature(density)
    }))
    writer.write(example.SerializeToString())


def create_train_val_database(patients_data, val_patients_data):
    writer = tf.python_io.TFRecordWriter(
        "../Data/train_"+str(image_dimensions[0])+"x"+str(image_dimensions[1])+".tfrecords")
    for i in range(len(patients_data)):
        patient = patients_data[i]
        image = read_image_file("../Data/DOI/"+str(patients_data[i][0]))
        write_image(writer, image, patient)
        image1 = add_augmentation(image, rotation_angle=0, flip_horizontal=False, flip_vertical=False,
                                  gaussian=True, zoom=True)
        write_image(writer, image1, patient)
        image1 = add_augmentation(image, rotation_angle=0, flip_horizontal=True, flip_vertical=False,
                                  gaussian=True, zoom=True)
        write_image(writer, image1, patient)
        image1 = add_augmentation(image, rotation_angle=0, flip_horizontal=False, flip_vertical=True,
                                  gaussian=True, zoom=True)
        write_image(writer, image1, patient)
        image1 = add_augmentation(image, rotation_angle=0, flip_horizontal=True, flip_vertical=True,
                                  gaussian=True, zoom=True)
        write_image(writer, image1, patient)
        image1 = add_augmentation(image, rotation_angle=90, flip_horizontal=True, flip_vertical=False,
                                  gaussian=True, zoom=True)
        write_image(writer, image1, patient)
        image1 = add_augmentation(image, rotation_angle=90, flip_horizontal=False, flip_vertical=True,
                                  gaussian=True, zoom=True)
        write_image(writer, image1, patient)
        image1 = add_augmentation(image, rotation_angle=270, flip_horizontal=False, flip_vertical=False,
                                  gaussian=True, zoom=True)
        write_image(writer, image1, patient)
        image1 = add_augmentation(image, rotation_angle=270, flip_horizontal=True, flip_vertical=False,
                                  gaussian=True, zoom=True)
        write_image(writer, image1, patient)
        image1 = add_augmentation(image, rotation_angle=270, flip_horizontal=False, flip_vertical=True,
                                  gaussian=True, zoom=True)
        write_image(writer, image1, patient)
    writer.close()
    writer = tf.python_io.TFRecordWriter(
        "../Data/val_" + str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords")
    for i in range(len(val_patients_data)):
        image = read_image_file("../Data/DOI/"+str(val_patients_data[i][0])).tobytes()
        label = val_patients_data[i][1]
        subtlety = patients_data[i][2]
        density = patients_data[i][3]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image),
            'label': _int64_feature(label),
            'subtlety': _int64_feature(subtlety),
            'density': _int64_feature(density)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def create_test_database(patients_data):
    writer = tf.python_io.TFRecordWriter(
        "../Data/test_" + str(image_dimensions[0]) + "x" + str(image_dimensions[1]) + ".tfrecords")
    for i in range(len(patients_data)):
        image = read_image_file("../Data/DOI/"+str(patients_data[i][0])).tobytes()
        label = patients_data[i][1]
        subtlety = patients_data[i][2]
        density = patients_data[i][3]
        example = tf.train.Example(features=tf.train.Features(feature={
            'image': _bytes_feature(image),
            'label': _int64_feature(label),
            'subtlety': _int64_feature(subtlety),
            'density': _int64_feature(density)
        }))
        writer.write(example.SerializeToString())
    writer.close()


def patients_sequencer(filename, training):
    if training:
        train_patients, val_patients = [], []
        patients = read_csv_file(filename)
        num_examples = len(patients)
        random_array = np.arange(num_examples)
        np.random.shuffle(random_array)
        for i in range(num_examples):
            if i < num_examples*0.8:
                patient = patients[random_array[i]]
                train_patients.append(patient)
            else:
                patient = patients[random_array[i]]
                val_patients.append(patient)
        print(len(train_patients))
        print(len(val_patients))
        return train_patients, val_patients
    else:
        test_patients = []
        patients = read_csv_file(filename)
        num_examples = len(patients)
        for i in range(num_examples):
            patient = patients[i]
            test_patients.append(patient)
        return test_patients


if __name__ == '__main__':
    train_patients, val_patients = patients_sequencer([training_csv_file_m, training_csv_file_c], training=True)
    test_patients = patients_sequencer([testing_csv_file_m, testing_csv_file_c], training=False)
    create_train_val_database(train_patients, val_patients)
    create_test_database(test_patients)
