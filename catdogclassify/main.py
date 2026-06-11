import tensorflow as tf

print("Starting the data pipeline...")

train_dataset = tf.keras.utils.image_dataset_from_directory(
    'PetImages',
    image_size = (150, 150),
    batch_size = 32
)

print("Building Convolutional Brain...")

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(150, 150, 3)),

    tf.keras.layers.Rescaling(1./255,),

    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()