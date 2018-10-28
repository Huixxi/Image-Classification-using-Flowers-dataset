# -*- coding: utf-8 -*-

import keras
import tensorflow as tf
# model = keras.models.load_model('./models/mymodel.h5')

from input_dataset import *

def model_test(test_file):
    test_data, test_label = read_and_decode(tfrecords_file=test_file)
    test_data = tf.expand_dims(test_data, 3)
    test_data = preprocess_input_image(test_data)
    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        model_h = keras.models.load_model('./mymodel.h5')
        
        i = 0
        try:
            while not coord.should_stop() and i<1:
                image, label = sess.run([test_data, test_label])
                image = (image+1)*127
                uint_image = image.astype(np.uint8)
                gray3_image = np.repeat(uint_image, 3, axis=-1)
                predicts = model_h.predict(x=gray3_image)
                i += 1
            return np.argmax(predicts, axis=1) + 1

        except tf.errors.OutOfRangeError:
            print('done!')
        finally:
            coord.request_stop()
        coord.join(threads)
        
def main():
    label = model_test('./')
    return label
    
    
if __name__ == '__main__':
    main()