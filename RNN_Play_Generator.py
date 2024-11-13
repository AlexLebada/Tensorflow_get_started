from keras_preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np
from keras import losses


### LOAD DATA
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
#print ('Length of text: {} characters'.format(len(text)))
#print(text[:500])

### PREPROCESS & SPLIT
# sort and retains unique characters in a list
vocab = sorted(set(text))
# make a dictionary with character u as an integer i; enumerate provide the counter/integer i
char2idx = {u:i for i, u in enumerate(vocab)}
# integer list to array for character selection based on array index
idx2char = np.array(vocab)

def text_to_int(text):
    return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text) # integer form array

def int_to_text(ints):
    try:
        ints = ints.numpy() # numpy() is specific to Tensorflow and used if ints is array/tensor
    except:
        pass
    return ''.join(idx2char[ints]) # returns a string(not separate elements

#print(int_to_text(text_as_int[:13]))


seq_length = 100
examples_per_epoch = len(text)//(seq_length+1) # // is floor division and it returns an integer instead of possible float
# text characters in integer form is splitted one by one as input
char_dataset = tf.data.Dataset.from_tensor_slices((text_as_int))
# it takes these splitted chars and batch them as 101 chars group; (low level data type BatchDataset)
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

# map() apply function to every batched element, also splitting into input/target text per each batch
# map() creates a low level data type mapDataset
dataset = sequences.map(split_input_target)


if 1==0:
    for x, y in dataset.take(2):
      print("\n\nEXAMPLE\n")
      print("INPUT")
      print(int_to_text(x))
      print("\nOUTPUT")
      print(int_to_text(y))

### BUILD MODEL
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 1024 # no. of nodes for LSTM
BUFFER_SIZE = 10000

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        # Embedding layer it handles input at char level
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                  input_length=None), # input_length keeps
        tf.keras.layers.LSTM(rnn_units,
                      return_sequences=True, # returning contextual prediction for every char
                      stateful=True, # keeps memory of previous batches, not reseting after a new batch; useful for long sequences
                      recurrent_initializer='glorot_uniform'), # sets method of adjusting weights
        tf.keras.layers.Dense(vocab_size) # return prob. of unique chars list
    ])
    return model

if (not os.path.exists('Recurrent_NN_2.keras')) or 1==0:
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    #model.build(input_shape=(BATCH_SIZE, None))
    #model.summary()
    # model prediction before training
    for input_example_batch, target_example_batch in data.take(1):
        predicted_example_batch = model(input_example_batch) # [64,100,65] - 64 senquences
        pred = predicted_example_batch[0]  # [100,65] - entire sequence
        time_pred = pred[0] # one char prediction

    # Uses sampling distribution for picking next char highest probability,but sometimes picking a lower probab.
    sampled_indices = tf.random.categorical(pred, num_samples=1)
    sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
    predicted_chars = int_to_text(sampled_indices)
    #print(len(predicted_chars))

    # Build a loss function to compare 3D nested arrays outputs[64,100,65] from model
    # Loss func. needs to be registered so can be saved/loaded
    @keras.saving.register_keras_serializable()
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels,logits, from_logits=True)

    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))

    # Create checkpoints
    checkpoint_filepath = "./training_checkpoints/ckpt_{epoch}.weights.h5"
    #checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        verbose=1,
        save_weights_only=True, # disabled for saving weights in other format than .h5
        save_best_only = False,
        save_freq='epoch'
    )

    ### TRAIN THE MODEL; in case of interrupt/pause resume training
    EPOCH_NO = 1
    last_ckp_name = './training_checkpoints/ckpt_'+str(EPOCH_NO)+'.weights.h5'
    if (not os.path.exists(last_ckp_name)):
        model.fit(data, epochs=EPOCH_NO, callbacks=[checkpoint_callback])
    else:
        model.load_weights('./training_checkpoints/ckpt_'+str(EPOCH_NO)+'.weights.h5')
        #model.fit(data, epochs=8, callbacks=[checkpoint_callback])
        model.save('Recurrent_NN_2.keras')

else:
    # !! NOT WORKING YET PREDICT/EVALUATE - causes: loss function/weights save/ input batch size
    model = keras.models.load_model('Recurrent_NN_2.keras')
    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, 1)
    model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
    sample_batch = next(iter(data))
    input_example_batch, target_example_batch = sample_batch
    loss, accuracy = model.evaluate(input_example_batch, target_example_batch, verbose=2)

    def generate_text(model, start_string):
        # Evaluation step (generating text using the learned model)

        # Number of characters to generate
        num_generate = 800

        # Converting our start string to numbers (vectorizing)
        input_eval = [char2idx[s] for s in start_string]
        input_eval = tf.expand_dims(input_eval, 0)

        # Empty string to store our results
        text_generated = []

        # Low temperatures results in more predictable text.
        # Higher temperatures results in more surprising text.
        # Experiment to find the best setting.
        temperature = 1.0

        # Here batch size == 1
        #model.reset_states()
        for i in range(num_generate):
            predictions = model(input_eval)
            # remove the batch dimension

            predictions = tf.squeeze(predictions, 0)

            # using a categorical distribution to predict the character returned by the model
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return (start_string + ''.join(text_generated))


    inp = input("Type a starting string: ")
    #print(generate_text(model, inp))