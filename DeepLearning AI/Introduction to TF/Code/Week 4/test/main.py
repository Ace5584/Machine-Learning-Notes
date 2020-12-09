import tensorflow as tf
import os
from os import path, getcwd, chdir
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop


# GRADED FUNCTION: train_happy_sad_model
def train_happy_sad_model():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.
    
    DESIRED_ACCURACY = 0.999

    class myCallback(tf.keras.callbacks.Callback):
         def on_epoch_end(self, epoch, log={}):
             if log.get('accuracy') > DESIRED_ACCURACY:
                print('Reached 99.9% accuracy so cancelling training!')
                model.stop_training = True
    callbacks = myCallback()
    
    # This Code Block should Define and Compile the Model. Please assume the images are 150 X 150 in your implementation.
    model = tf.keras.models.Sequential([
       tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
       tf.keras.layers.MaxPool2D(2, 2),
       tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
       tf.keras.layers.MaxPool2D(2, 2),
       tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
       tf.keras.layers.MaxPool2D(2, 2),
       tf.keras.layers.Flatten(),
       tf.keras.layers.Dense(40, activation='relu'),
       tf.keras.layers.Dense(1, 'sigmoid')
    ])

    
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics=['accuracy'])


    # This code block should create an instance of an ImageDataGenerator called train_datagen 
    # And a train_generator by calling train_datagen.flow_from_directory

    

    train_datagen = ImageDataGenerator(rescale=1./255)

    # Please use a target_size of 150 X 150.
    train_generator = train_datagen.flow_from_directory(
        'C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Deep Learning AI course/Week 4/test/happy-or-sad',
        target_size=(150, 150), batch_size=10, class_mode='binary'
    )
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(train_generator, steps_per_epoch=8, epochs=15, callbacks=[callbacks], verbose=1)
    # model fitting
    return history.history['accuracy'][-1]

# The Expected output: "Reached 99.9% accuracy so cancelling training!""
train_happy_sad_model()





