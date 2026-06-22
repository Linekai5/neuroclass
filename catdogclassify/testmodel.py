import os
import numpy as np
import tensorflow as tf

print("Activating classifier...")

# 1. LOAD THE SAVED MODEL
model_path = 'cat_dog_model.keras'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Error: Cannot find {model_path}")

model = tf.keras.models.load_model(model_path)

# 2. LOAD YOUR CUSTOM PHOTO
# Pas het pad aan naar de foto die je wilt testen (bijv. 'testphotos/cat.png')
image_path = 'testphotos/dog.png'

if not os.path.exists(image_path):
    raise FileNotFoundError(f"Fout: Kan de afbeelding op {image_path} niet vinden.")

# Resize naar 150x150 (moet exact overeenkomen met je training)
img = tf.keras.utils.load_img(image_path, target_size=(150, 150))

# 3. TRANSLATE THE PHOTO INTO MATH & NORMALIZE
img_array = tf.keras.utils.img_to_array(img)

# The model already has a Rescaling layer, so we don't need to divide by 255 here.
# img_array = img_array / 255.0 

# Voeg een extra dimensie toe zodat het een batch van 1 afbeelding wordt
img_array = tf.expand_dims(img_array, 0) 

print(f"Analyzing the photo: '{image_path}'...")

# 4. MAKE THE PREDICTION
prediction = model.predict(img_array, verbose=0)

# 5. TRANSLATE MATH BACK TO ENGLISH
probability = prediction[0][0]

# Uitleg: 0.5 is het omslagpunt. Dichter bij 1 = Hond, dichter bij 0 = Kat.
if probability >= 0.5:
    certainty = probability * 100
    print(f"I am {certainty:.2f}% sure this is a DOG!")
else:
    certainty = (1.0 - probability) * 100
    print(f"I am {certainty:.2f}% sure this is a CAT!")
