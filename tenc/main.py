import tensorflow as tf
import numpy as np
import datetime

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

train_filter = (y_train == 8) | (y_train == 6)
test_filter = (y_test == 8) | (y_test == 6)
x_train, y_train = x_train[train_filter], y_train[train_filter]
x_test, y_test = x_test[test_filter], y_test[test_filter]

y_train = np.where(y_train == 8, 0, 1)
y_test = np.where(y_test == 8, 0, 1)

x_train = x_train / 255.0 
x_test = x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer ='adam',
    loss = 'binary_crossentropy',
    metrics =['accuracy']
)

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard_callback])

loss, accuracy = model.evaluate(x_test, y_test)

model.save('binary_classifier.keras')
print("Model saved successfully")