import os
import tensorflow as tf

print("Putting on the hazmat suit. Cleaning the dataset...")
deleted = 0

# Look inside both the Cat and Dog folders
for folder in ['Cat', 'Dog']:
    folder_path = os.path.join('PetImages', folder)
    
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Mac computers often create hidden files like '.DS_Store'. 
        # We need to delete those too because they aren't images!
        if filename.startswith('.'):
            os.remove(file_path)
            continue
            
        try:
            # We ask TensorFlow to try and read the image
            img_bytes = tf.io.read_file(file_path)
            # We ask TensorFlow to decode it. If the image is corrupt or 
            # has the wrong color channels, this line will crash!
            _ = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
            
        except Exception:
            # If the code above crashes, it jumps down here.
            # We delete the bad file so it can't ruin our training later.
            print(f"Deleting broken file: {file_path}")
            os.remove(file_path)
            deleted += 1

print(f"Done! Swept away {deleted} broken files.")