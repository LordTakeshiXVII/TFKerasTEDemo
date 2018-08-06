
TRAINING_SAMPLES = 120
VALIDATION_SAMPLES = 30
BATCH_SIZE = 30
TRAINING_STEPS = TRAINING_SAMPLES // BATCH_SIZE
VALIDATION_STEPS = VALIDATION_SAMPLES // BATCH_SIZE


def load_data(training_or_validation):
    import tensorflow as tf

    # function to parse a single line into features and labels
    def parse_csv(line):
        parsed_line = tf.decode_csv(line, [[0.], [0.], [0.], [0.], [0]])
        # extract features
        features = tf.reshape(parsed_line[:-1], shape=(4,))
        # extract label and make one-hot encoding
        label = tf.one_hot(tf.reshape(parsed_line[-1], shape=()), 3)
        return features, label

    file_name = ".\data\iris_{}.csv".format(training_or_validation)

    dataset = tf.data.TextLineDataset(file_name) \
        .skip(1) \
        .map(parse_csv) \
        .shuffle(buffer_size=500) \
        .batch(BATCH_SIZE) \
        .repeat()

    return dataset


def build_model():
    from tensorflow.keras.layers import Dense, Input
    from tensorflow.keras.models import Model

    input = Input(shape=(4,), name="input_layer")
    x = Dense(10, activation='relu', name="hidden_layer_1")(input)
    x = Dense(8, activation='relu', name="hidden_layer_2")(x)
    x = Dense(6, activation='relu', name="hidden_layer_3")(x)
    output = Dense(3, activation='softmax', name="output_layer")(x)
    model = Model(inputs=input, outputs=output)
    return model

def compile_model(model):
    from tensorflow.keras.optimizers import Adam
    optimizer = Adam(lr=0.04, decay=1e-6)
    lossfunction = 'categorical_crossentropy'
    model.compile(optimizer, 
                  lossfunction, 
                  metrics=['accuracy'])

def create_tensorboard_callback():
    from tensorflow.keras.callbacks import TensorBoard
    return TensorBoard(log_dir=".\\tensorboard", histogram_freq=0,
                       write_graph=True, write_images=False, write_grads=True)

def run_model():
    model = build_model()
    print(model.summary())

    compile_model(model)

    training_dataset = load_data("training")

    model.fit(training_dataset,
              steps_per_epoch=TRAINING_STEPS,
              epochs=50,
              verbose=1,
              callbacks=[create_tensorboard_callback()])

    # evaluate model on training data
    training_eval = model.evaluate(training_dataset, steps = TRAINING_STEPS)

    # evaluate model on validation data
    validation_dataset = load_data("validation") 
    validation_eval = model.evaluate(validation_dataset, steps = VALIDATION_STEPS)
    print("Training accuracy:   {}".format(training_eval[1]))
    print("Validation accuracy: {}".format(validation_eval[1]))

    model.save(".\model\iris.h5")

run_model()


# tensorboard -logdir C:\Users\chris\Source\Repos\TFKerasTEDemo\DeepIris\tensorboard
