# -*- coding: utf-8 -*-

# filter warnings
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import tensorflow as tf
import numpy as np
from tensorflow.python.framework import graph_util
import keras
import matplotlib.pyplot as plt

# py scripts imports
from input_dataset import read_and_decode, preprocess_input_image
# read_and_decode(tfrecords_files)， preprocess_input_image(img_batch, train=False)

def train(filenames):
    with tf.Graph().as_default() as g:  # 指定当前图为默认graph
        images, labels = read_and_decode(filenames)
        images = tf.expand_dims(images, 3)
        images = preprocess_input_image(images, train=True)
        with tf.Session() as sess:
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            model = keras.models.load_model('./models/flowers_5.h5')
            adam = keras.optimizers.Adam(lr=.0001)
            model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
            
            MAX_EPOCH = 10
            i = 0
            
            try:
                while i < MAX_EPOCH:
                    j = 0
                    while j < 2:
                        image, label = sess.run([images, labels])
                        image = (image+1)*127
                        uint_image = image.astype(np.uint8)
                        gray3_image = np.repeat(uint_image, 3, axis=-1)
                        model.fit(x=gray3_image, y=label, batch_size=64, epochs=1)
                        j += 1
                    i += 1
                
                model.save('./models/mymodel.h5', include_optimizer=False)
                
            except tf.errors.OutOfRangeError:
                print('done!')
            finally:
                coord.request_stop()
            coord.join(threads)
            
if __name__ == '__main__':
    filenames = './Ele_5_datasets/train/'
    train(filenames)