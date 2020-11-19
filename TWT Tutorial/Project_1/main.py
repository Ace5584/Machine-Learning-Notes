import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

data = keras.datasets.fashion_mnist
(train_data, train_labels), (test_data, test_labels) = data.load_data()
train_data = train_data/255
test_data = test_data/255

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(10)
    ]
)

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=10)

test_loss, test_acc = model.evaluate(test_data, test_labels, verbose=2)
print("Test Acc:", test_acc)

probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
prediction = probability_model.predict(test_data)

for i in range(5):
    plt.imshow(test_data[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[np.argmax(prediction[i])])
    plt.show()



