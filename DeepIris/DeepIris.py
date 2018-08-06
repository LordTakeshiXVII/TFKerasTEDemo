import tensorflow as tf

from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard


import os
import tempfile


_TRAIN_COUNT = 120
_VAL_COUNT = 30
_BATCH_SIZE = 30


def load_data(train_test):
    url = "http://download.tensorflow.org/data/iris_{}.csv".format(train_test)
    fname = "{}\\{}".format(tempfile.gettempdir(), os.path.basename(url))
    if not os.path.isfile(fname):
        tf.keras.utils.get_file(fname=fname, origin=url)

    print("Local copy of the dataset file: {}".format(fname))

    def parse_csv(line):
        example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
        parsed_line = tf.decode_csv(line, example_defaults)
        # First 4 fields are features, combine into single tensor
        features = tf.reshape(parsed_line[:-1], shape=(4,))
        # Last field is the label
        label = tf.one_hot(tf.reshape(parsed_line[-1], shape=()), 3)
        return features, label

    dataset = tf.data.TextLineDataset(fname) \
        .skip(1) \
        .map(parse_csv) \
        .shuffle(buffer_size=500) \
        .batch(_BATCH_SIZE) \
        .repeat()
    return dataset.make_one_shot_iterator().get_next()
    # iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
    #                                            train_dataset.output_shapes)
    # next_element = iterator.get_next()
    # training_init_op = iterator.make_initializer(train_dataset)
    #
    # return next_element, training_init_op


def build_model(features_learn, labels_learn, features_val, labels_val):
    input = Input(tensor=features_learn)
    x = BatchNormalization(trainable=True)(input)
    x = Dense(10, activation='relu')(x)
    x = Dense(8, activation='relu')(x)
    x = Dense(6, activation='relu')(x)
    output = Dense(3, activation='softmax')(x)
    model = Model(inputs=input, outputs=output)
    opt = Adam(lr=0.04, decay=1e-6)
    model.compile(opt, 'categorical_crossentropy', metrics=['acc'], target_tensors=[labels_learn])
    print(model.summary())

    tensorboard_dir = "{}\\tensorboard1".format(tempfile.gettempdir())
    print(tensorboard_dir)
    tensorboard = TensorBoard(log_dir=tensorboard_dir, histogram_freq=0,
                              write_graph=True, write_images=False, write_grads=True)
    model.fit(steps_per_epoch=_TRAIN_COUNT // _BATCH_SIZE,
              epochs=50,
              verbose=2,
              callbacks=[tensorboard])
    #model.fit(steps_per_epoch=_TRAIN_COUNT // _BATCH_SIZE,
    #          epochs=50,
    #          verbose=2,
    #          callbacks=[tensorboard],
    #          validation_data=(features_val, labels_val),
    #          validation_steps=_VAL_COUNT // _BATCH_SIZE)

    model.predict()
    return model


def run_model():
    train = load_data("training")
    test = load_data("test")
    m = build_model(train[0], train[1], test[0], test[1])
    e1 = m.evaluate(train[0], train[1], steps=_TRAIN_COUNT // _BATCH_SIZE)
    print(e1)
    e = m.evaluate(test[0], test[1], steps=_VAL_COUNT // _BATCH_SIZE)
    print(e)
    m.save("model.h5")


run_model()
print("the end")

# tensorboard -logdir C:\Users\chris\AppData\Local\Temp\tensorboard1

# def simple_dataset_with_error():
#     x = np.arange(0, 10)
#     # create dataset object from the numpy array
#     dx = tf.data.Dataset.from_tensor_slices(x).repeat().batch(2)
#     # create a one-shot iterator
#     iterator = tf.data.Iterator.from_structure(dx.output_types,
#                                                dx.output_shapes)
#     # extract an element
#     next_element = iterator.get_next()
#     training_init_op = iterator.make_initializer(dx)
#     with tf.Session() as sess:
#         sess.run(training_init_op)
#         for i in range(10):
#             val = sess.run(next_element)
#             print(val)


# def iterate_blumen():
#     next, init = load_blumen()
#     nx = next[1]
#     with tf.Session() as sess:
#         sess.run(init)
#         for i in range(10):
#             val = sess.run(nx)
#             print(val)

