import os
import cv2 # type: ignore
import numpy as np
import tensorflow as tf # type: ignore

# =====================================================================
# 1. LOAD THE TRAINED BRAIN
# =====================================================================
model_path = 'cat_dog_model.keras'
if not os.path.exists(model_path):
    model_path = '../catdogclassify/cat_dog_model.keras'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Could not find '{model_path}'.")

print("Waking up the AI Brain for live video...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# =====================================================================
# 2. EXTRACT BRAIN HALVES FOR FAST GRAD-CAM
# =====================================================================
print("Compiling Sci-Fi Target Lock (Grad-CAM)...")
base_model = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)][0]

# Grab the preprocessing layers (like Rescaling) that happen BEFORE MobileNetV2
prep_layers = []
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        break
    prep_layers.append(layer)

# Split the internal brain
last_conv_layer = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]
last_conv_model = tf.keras.models.Model(base_model.inputs, last_conv_layer.output)

classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
last_conv_index = base_model.layers.index(last_conv_layer)
for layer in base_model.layers[last_conv_index + 1:]:
    x = layer(x)
base_model_index = model.layers.index(base_model)
for layer in model.layers[base_model_index + 1:]:
    x = layer(x)
classifier_model = tf.keras.models.Model(classifier_input, x)

# MAGIC TRICK: @tf.function compiles this math into lightning-fast C++ 
# so it can run at 30+ frames per second without lagging your webcam!
@tf.function
def get_prediction_and_heatmap(img_tensor):
    # 1. Preprocess the image (tell layers we are NOT training right now)
    x = img_tensor
    for layer in prep_layers:
        x = layer(x, training=False)
        
    # 2. Run the first half of the brain
    # (tape watches the math so we can rewind it)
    with tf.GradientTape() as tape:
        last_conv_output = last_conv_model(x, training=False)
        tape.watch(last_conv_output)
        preds = classifier_model(last_conv_output, training=False)
        class_channel = preds[:, 0]

    # 3. Rewind the math to create the Heatmap
    grads = tape.gradient(class_channel, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    heatmap = last_conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # Clean up the math and normalize from 0.0 to 1.0
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)
    
    return preds[0][0], heatmap

# =====================================================================
# 3. INITIALIZE WEBCAM AND SMOOTHING UTILITIES
# =====================================================================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemError("Could not access your MacBook webcam.")

prediction_history = []
HISTORY_LENGTH = 10 

print("\n--- Live Classifier Active ---")
print("Press the 'q' key in the video window to quit.")

# =====================================================================
# 4. THE VIDEO STREAM LOOP
# =====================================================================
while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera disconnected/failed. Exiting...")
        break

    # -----------------------------------------------------------------
    # MIRROR FIX: Flip the frame horizontally so it feels natural
    # -----------------------------------------------------------------
    frame = cv2.flip(frame, 1)

    # Get dimensions
    height, width, _ = frame.shape
    
    # -----------------------------------------------------------------
    # STEP A: Crop the center square for the AI
    # -----------------------------------------------------------------
    crop_size = min(height, width)
    start_y = (height - crop_size) // 2
    start_x = (width - crop_size) // 2
    cropped_frame = frame[start_y:start_y+crop_size, start_x:start_x+crop_size]

    # -----------------------------------------------------------------
    # STEP B: Prepare the image for the model
    # -----------------------------------------------------------------
    img_resized = cv2.resize(cropped_frame, (150, 150))
    # Convert BGR (OpenCV) to RGB (TensorFlow)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    img_array = tf.expand_dims(img_rgb, 0)
    
    # -----------------------------------------------------------------
    # STEP C: Run the FAST prediction function
    # -----------------------------------------------------------------
    # Call the @tf.function optimized prediction and gradcam
    raw_prediction, heatmap_tensor = get_prediction_and_heatmap(img_array)
    raw_prediction = float(raw_prediction)

    # Add to our smoothing queue
    prediction_history.append(raw_prediction)
    if len(prediction_history) > HISTORY_LENGTH:
        prediction_history.pop(0)
        
    # Calculate the average score of our queue
    smoothed_prediction = np.mean(prediction_history)

    # -----------------------------------------------------------------
    # STEP D: Dynamic HUD Drawing & Live Heatmap
    # -----------------------------------------------------------------
    # We widened the "Nothing" State to 10% - 90%. 
    # Only extreme confidence triggers the label and the heatmap!
    if 0.10 < smoothed_prediction < 0.90:
        hud_text = "Searching for pet..."
        box_color = (0, 165, 255)  # Orange BGR color code
        # No heatmap generated here, just normal video
    else:
        # We are highly confident! Let's generate the live heatmap.
        # Format heatmap for OpenCV display
        heatmap = heatmap_tensor.numpy()

        # 4. Resize heatmap to fit our live cropped video window
        heatmap_resized = cv2.resize(heatmap, (crop_size, crop_size))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        colored_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # 5. Filter out the cold areas! Only overlay where heatmap intensity is > 40%
        hot_mask = heatmap_uint8 > 100
        hot_mask_3d = np.repeat(hot_mask[:, :, np.newaxis], 3, axis=2)
        
        alpha = 0.6 # Transparency of the heatmap
        overlay = cropped_frame.copy()
        cv2.addWeighted(overlay, 1 - alpha, colored_heatmap, alpha, 0, overlay)
        
        # Combine the frames: use heatmap where hot, keep original video where cold
        cropped_frame = np.where(hot_mask_3d, overlay, cropped_frame)
        
        # Put the modified cropped square back into the main screen
        frame[start_y:start_y+crop_size, start_x:start_x+crop_size] = cropped_frame

        # 6. Set up the Text
        if smoothed_prediction >= 0.90:
            confidence = smoothed_prediction * 100
            hud_text = f"DOG ({confidence:.1f}%)"
            box_color = (0, 0, 255)  # Bright Red
        else:
            confidence = (1.0 - smoothed_prediction) * 100
            hud_text = f"CAT ({confidence:.1f}%)"
            box_color = (255, 0, 0)  # Bright Blue

    # Draw the targeting box in the center of the live screen
    cv2.rectangle(
        frame, (start_x, start_y), (start_x + crop_size, start_y + crop_size), 
        box_color, thickness=3
    )
    # Write the classification text safely INSIDE the box so it never gets cut off
    cv2.putText(
        frame, hud_text, (start_x + 10, start_y + 35), 
        cv2.FONT_HERSHEY_SIMPLEX, 1.0, box_color, 2, cv2.LINE_AA
    )

    # Render
    cv2.imshow("Live Pet AI Classifier", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Shutting down camera and releasing resources...")
cap.release()
cv2.destroyAllWindows()
print("Done! See you next session!")