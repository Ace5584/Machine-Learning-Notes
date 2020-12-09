import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

class MyCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, log):
        if log.get('accuracy') > 0.999:
            print("\nReached 99.9 accuracy so cancelling training!")
            self.model.stop_training = True

callback = MyCallBack()

train_hourse_dir = os.path.join('C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Deep Learning AI course/Week 4/horse or human/horse-or-human/horses')
train_human_dir = os.path.join('C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Deep Learning AI course/Week 4/horse or human/horse-or-human/humans')

train_hourse_labels = os.listdir(train_hourse_dir)
train_human_labels = os.listdir(train_human_dir)
# print(train_hourse_labels[0:10])
# print(train_human_labels[0:10])
# print('Total hourse Images', len(train_hourse_labels))
# print('Total human Images', len(train_human_labels))

# ------------------------------------------------------------- #
# Plotting plt images
# nrows = 4
# ncols = 4
# pic_index = 0
# fig = plt.gcf()
# fig.set_size_inches(ncols*4, nrows*4)
# pic_index += 28
# next_horse_pix = [os.path.join(train_hourse_dir, fname) 
#                 for fname in train_hourse_labels[pic_index-8:pic_index]]
# next_human_pix = [os.path.join(train_human_dir, fname) 
#                 for fname in train_human_labels[pic_index-8:pic_index]]

# for i, img_path in enumerate(next_horse_pix+next_human_pix):
#   # Set up subplot; subplot indices start at 1
#   sp = plt.subplot(nrows, ncols, i + 1)
#   sp.axis('Off') # Don't show axes (or gridlines)

#   img = mpimg.imread(img_path)
#   plt.imshow(img)

# plt.show()
# ------------------------------------------------------------- #

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer=RMSprop(lr=0.001),metrics=['accuracy'])
train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
        'C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/Deep Learning AI course/Week 4/horse or human/horse-or-human/', 
        target_size=(300, 300), batch_size=128, class_mode='binary')
history = model.fit(train_generator, steps_per_epoch=8, epochs=15, verbose=1, callbacks=[callback])
















