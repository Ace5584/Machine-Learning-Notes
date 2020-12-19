import os
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

base_dir = "C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Machine Learning/Deep Learning AI course 4/Week 1/cats_and_dogs_filtered"
train_dir = base_dir + '/train'
validation_dir = base_dir + '/validation'

train_cats_dir = train_dir + '/cats'
train_dogs_dir = train_dir + '/dogs'

validation_cats_dir = validation_dir + '/cats'
validation_dogs_dir = validation_dir + '/dogs'

train_cats_frame = os.listdir(train_cats_dir)
train_dogs_frame = os.listdir(train_dogs_dir)

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
  tf.keras.layers.MaxPool2D(2, 2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPool2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPool2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, 'sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001), metrics='acc')
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(150, 150), batch_size=20, class_mode='binary'
)
validation_generator = test_datagen.flow_from_directory(
    validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary'
)

history = model.fit(train_generator, validation_data=validation_generator, steps_per_epoch=100,epochs=15, validation_steps=50,verbose=2)
