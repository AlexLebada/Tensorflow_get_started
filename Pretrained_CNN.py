import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
keras = tf.keras
import tensorflow_datasets as tfds
tfds.disable_progress_bar()

### LOAD & SPLIT
# this load returns tf.data.Dataset object
# split: 80% training, 10% validation, 10% test
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True, # for supervised learning
)

get_label_name = metadata.features['label'].int2str
print(get_label_name)

# display 5 images from the dataset
if 1==0:
    for image, label in raw_train.take(5):
      plt.figure()
      plt.imshow(image)
      plt.title(get_label_name(label))
      plt.show()

### PREPROCESS
# convert diff images sizes to same size
IMG_SIZE = 160 # 160x160 size

def format_image(image, label):
    image = tf.cast(image, tf.float32) # from integer image tensor to floating (better practice)
    image =(image/127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

# Apply function on every element in the dataset
train = raw_train.map(format_image)
validation = raw_validation.map(format_image)
test = raw_test.map(format_image)

# Verify couple of samples
if 1==0:
    for image, label in train.take(5):
      plt.figure()
      plt.imshow(image)
      plt.title(get_label_name(label))
      plt.show()
      for img, label in raw_train.take(5):
          print("Original shape:", img.shape)

      for img, label in train.take(5):
          print("New shape:", img.shape)

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000
# restructure tf object with batching and shuffling
train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE) # it's the entire dataset
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

if not os.path.exists('Custom_MobileNetV2.keras'):
    ### BUILD MODEL - with pretrained


    # for this model it will output (32,5,5,1280) from our input (1, 160, 160, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet'
                                                   )
    #base_model.summary()

    if 1==0:
        for image, _ in train_batches.take(1):
            pass
        #output the feature_map of our image
        feature_batch = base_model(image)
        print(feature_batch.shape)

    base_model.trainable = False
    #base_model.summary()

    # Add our classifier
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D() # average 5x5 feature_map and flatten to 1280 array
    prediction_layer = tf.keras.layers.Dense(1) # 1 neuron=2 classes
    model = tf.keras.Sequential([
        base_model,
        global_average_layer,
        prediction_layer
    ])

    ### TRAIN THE MODEL
    base_learning_rate = 0.0001
    model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # Evaluate before training
    initial_epochs = 3
    validation_steps = 20
    loss_init, accuracy_init = model.evaluate(validation_batches, steps=validation_steps) # steps = batches
    print('\nScore before training ')
    print(loss_init, accuracy_init)

    model.fit(train_batches,
              epochs=initial_epochs,
              validation_data=validation_batches)
    model.save('Custom_MobileNetV2.keras')
    test_loss, test_acc = model.evaluate(test_batches, verbose=2)
    print('\nScore after training ')
    print('\nLoss: ', test_loss, '\nAccuracy: ', test_acc)

else:
    model = keras.models.load_model('Custom_MobileNetV2.keras')
    ### PREDICT
    prediction = model.predict(test_batches)
    for image, label in test_batches.take(1):  # Take one example from the raw dataset
        print(prediction[20])
        image=image[20].numpy()
        image = (image + 1) / 2
        plt.figure()
        plt.imshow(image)
        plt.title(get_label_name(label[20].numpy()))
        plt.axis('off')
        plt.show()
