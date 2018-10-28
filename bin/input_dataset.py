# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import glob

IMAGE_SIZE = 224
BATCH_SIZE = 25

def read_and_decode(tfrecords_file):
    '''read and decode tfrecord file, generate (image, label) batches
    Args:
        tfrecords_file: the directory of tfrecord file
        batch_size: number of images in each batch
    Returns:
        image: 4D tensor - [batch_size, width, height, channel]
        label: 2D tensor - [batch_size, class_num]
    '''
    # make an input queue from the tfrecord file
    tfrecords_files = glob.glob(tfrecords_file+'*.tfrecord')
    # print tfrecords_files
    filename_queue = tf.train.string_input_producer(tfrecords_files)
    
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    img_features = tf.parse_single_example(serialized=serialized_example,
                              features={
                                 'data':tf.FixedLenFeature([256, 256], tf.float32),
                                 'label':tf.FixedLenFeature([], tf.int64),
                                 'id':tf.FixedLenFeature([], tf.int64)
                                 })
    # image = tf.decode_raw(img_features['data'], tf.uint8), but I don't need to use it here.
    image = img_features['data']
    ##########################################################
    # you can put data augmentation here
    ##########################################################
    label = tf.cast(img_features['label'], tf.int32) 
    # get batch
    image_batch, label_batch = tf.train.batch([image, label],
                                batch_size= BATCH_SIZE,
                                num_threads= 64, 
                                capacity = 2000)
    
    return image_batch, tf.one_hot(tf.subtract(label_batch, 1), 5) # generate the label one-hot vectors, 5 is class_num

def preprocess_input_image(img_batch, train=False):
    img_batch = tf.image.resize_images(images=img_batch, size=(IMAGE_SIZE, IMAGE_SIZE))
    if train: # to testset, just resize
        img_batch = tf.image.random_flip_left_right(img_batch)
        img_batch = tf.image.random_flip_up_down(img_batch)
    return img_batch
    