from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
import matplotlib.pyplot as plt
import os

local_weights_file = "C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Machine Learning/Deep Learning AI course 4/Week 3/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"

pre_trained_model = InceptionV3(input_shape=(150, 150, 3), include_top=False, weights=None)
pre_trained_model.load_weights(local_weights_file)
for layer in pre_trained_model.layers:
  layer.trainable = False
last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape', last_layer.output_shape)
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)

model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics='accuracy')

base_dir = "C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Machine Learning/Deep Learning AI course 4/Week 3/cats_and_dogs_filtered"
train_dir = base_dir + '/train'
validation_dir = base_dir + '/validation'

train_cats_dir = train_dir + '/cats'
train_dogs_dir = train_dir + '/dogs'

validation_cats_dir = validation_dir + '/cats'
validation_dogs_dir = validation_dir + '/dogs'

train_cats_frame = os.listdir(train_cats_dir)
train_dogs_frame = os.listdir(train_dogs_dir)

# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255.,
rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)


test_datagen = ImageDataGenerator( rescale = 1.0/255. )

# Flow training images in batches of 20 using train_datagen generator
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 20, class_mode = 'binary',  target_size = (150, 150))     

# Flow validation images in batches of 20 using test_datagen generator
validation_generator =  test_datagen.flow_from_directory(validation_dir, batch_size  = 20, class_mode  = 'binary', target_size = (150, 150))

history = model.fit(
            train_generator, validation_data = validation_generator, steps_per_epoch = 100, epochs = 20, validation_steps = 50)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()