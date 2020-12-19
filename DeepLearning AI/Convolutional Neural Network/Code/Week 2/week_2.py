from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import tensorflow as tf
import matplotlib.pyplot as plt

train_horse_dir = "C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Machine Learning/Deep Learning AI course 4/Week 2/horse-or-human/horses"

train_human_dir = "C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Machine Learning/Deep Learning AI course 4/Week 2/horse-or-human/humans"

validation_horse_dir = "C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Machine Learning/Deep Learning AI course 4/Week 2/validation-horse-or-human/horses"

validation_human_dir = "C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Machine Learning/Deep Learning AI course 4/Week 2/validation-horse-or-human/humans"

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(300, 300, 3)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])

train_datagen = ImageDataGenerator(
      rescale=1./255, rotation_range=40,width_shift_range=0.2, height_shift_range=0.2,
      shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    'C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Machine Learning/Deep Learning AI course 4/Week 2/horse-or-human/',  
    target_size=(300, 300), batch_size=128, class_mode='binary')
validation_generator = validation_datagen.flow_from_directory(
    'C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Machine Learning/Deep Learning AI course 4/Week 2/validation-horse-or-human/', 
    target_size=(300, 300), batch_size=32, class_mode='binary')

history = model.fit(train_generator, steps_per_epoch=8, epochs=10, validation_data=validation_generator, validation_steps=8)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()