import tensorflow as tf

print("Setting up data pipeline with validation split...")

# Variables for the dataset
folder_name = 'PetImages'
img_size = (150, 150)
batch_size = 32

# 1. TRAINING DATASET (80% of the photos)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    folder_name,
    validation_split=0.2, 
    subset="training",    
    seed=123,             
    image_size=img_size,
    batch_size=batch_size
)

# 2. VALIDATION DATASET (20% of the photos)
val_dataset = tf.keras.utils.image_dataset_from_directory(
    folder_name,
    validation_split=0.2,
    subset="validation",  
    seed=123,             
    image_size=img_size,
    batch_size=batch_size
)

print("Loading Google's Pre-trained MobileNetV2 brain...")

# 3. LOAD THE PRE-TRAINED FEATURE EXTRACTOR
# We download the model weights trained on the ImageNet dataset.
# We set include_top=False to discard the final 1000-class decision layer,
# so we can build our own custom 2-class (Cat vs Dog) decision layer on top.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(150, 150, 3),
    include_top=False,
    weights='imagenet'
)

# CRITICAL: We freeze the base model so we don't destroy its pre-trained knowledge.
base_model.trainable = False

print("Building model architecture with Transfer Learning...")

# 4. BUILDING THE NEW MODEL
model = tf.keras.models.Sequential([
    # The "Front Door"
    tf.keras.layers.Input(shape=(150, 150, 3)),

    # DATA AUGMENTATION (Protects our new decision layer from overfitting)
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),

    # MobileNetV2 expects input pixel values scaled between -1 and 1
    tf.keras.layers.Rescaling(scale=1./127.5, offset=-1),

    # The frozen pre-trained feature extractor
    base_model,

    # Instead of Flatten, we use GlobalAveragePooling2D which works beautifully with transfer learning
    tf.keras.layers.GlobalAveragePooling2D(),

    # Dropout for robust regularization
    tf.keras.layers.Dropout(0.2),

    # Output (0 = Cat, 1 = Dog)
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# 5. COMPILING
# We use a slightly lower learning rate (0.0001) which is standard for transfer learning
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("Starting Transfer Learning training with Early Stopping...")

# EARLY STOPPING (Stops when validation error is optimized)
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',         
    patience=3,                 
    restore_best_weights=True   
)

# 6. TRAINING AND MONITORING
# Since the massive feature extraction layers are already trained, 
# this will run incredibly fast and reach high accuracy in just a few epochs!
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20, 
    callbacks=[early_stopping]
)

# 7. SAVING THE NEW BRAIN
model.save('cat_dog_model.keras')
print("Training complete! The pre-trained transfer learning model has been saved!")