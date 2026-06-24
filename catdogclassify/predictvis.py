import tensorflow as tf # type: ignore
import numpy as np
import matplotlib.pyplot as plt

print("Waking up the AI Dashboard...")

# ==========================================
# 1. SETUP AND LOADING
# ==========================================
model = tf.keras.models.load_model('cat_dog_model.keras')
image_path = 'testphotos/dog.png' # CHANGE THIS to your test image

# Load original image for drawing
original_img = tf.keras.utils.load_img(image_path, target_size=(150, 150))
img_array = tf.keras.utils.img_to_array(original_img)
img_array = tf.expand_dims(img_array, 0) 

print("Analyzing the photo...")

# ==========================================
# 2. THE PREDICTION & CONFIDENCE
# ==========================================
prediction = model.predict(img_array, verbose=0)
probability = prediction[0][0]

# Calculate exact probabilities for the graph
dog_prob = probability
cat_prob = 1.0 - probability

if probability >= 0.5:
    label = f"DOG ({dog_prob * 100:.1f}%)"
else:
    label = f"CAT ({cat_prob * 100:.1f}%)"
print(f"Prediction: {label}")

# ==========================================
# 3. EXTRACT INTERNAL MODELS & PREPROCESSING
# ==========================================
base_model = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)][0]

# Preprocess image (Data Augmentation & Rescaling from your main model)
preprocessed_img = img_array
for layer in model.layers:
    if layer.name == base_model.name:
        break
    preprocessed_img = layer(preprocessed_img)

# ==========================================
# 4. X-RAYS (Shallow and Deep)
# ==========================================
# SHALLOW X-RAY: First Conv2D layer (sees edges and outlines)
first_conv_layer = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)][0]
shallow_xray_model = tf.keras.models.Model(inputs=base_model.inputs, outputs=first_conv_layer.output)
shallow_features = shallow_xray_model.predict(preprocessed_img, verbose=0)[0] 

# DEEP X-RAY: A Conv2D layer from the middle of the brain (sees abstract textures)
mid_index = len(base_model.layers) // 2
mid_conv_layer = [layer for layer in base_model.layers[mid_index:] if isinstance(layer, tf.keras.layers.Conv2D)][0]
deep_xray_model = tf.keras.models.Model(inputs=base_model.inputs, outputs=mid_conv_layer.output)
deep_features = deep_xray_model.predict(preprocessed_img, verbose=0)[0]

# ==========================================
# 5. HEATMAP (Grad-CAM)
# ==========================================
last_conv_layer = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.Conv2D)][-1]

# Split brain for math recording
last_conv_layer_model = tf.keras.models.Model(base_model.inputs, last_conv_layer.output)

classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
x = classifier_input
last_conv_index = base_model.layers.index(last_conv_layer)
for layer in base_model.layers[last_conv_index + 1:]:
    x = layer(x)
base_model_index = model.layers.index(base_model)
for layer in model.layers[base_model_index + 1:]:
    x = layer(x)
classifier_model = tf.keras.models.Model(classifier_input, x)

with tf.GradientTape() as tape:
    last_conv_layer_output = last_conv_layer_model(preprocessed_img)
    tape.watch(last_conv_layer_output)
    preds = classifier_model(last_conv_layer_output)
    class_channel = preds[:, 0]

grads = tape.gradient(class_channel, last_conv_layer_output)
pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
last_conv_layer_output = last_conv_layer_output[0]
heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
heatmap = tf.squeeze(heatmap)
heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
heatmap = heatmap.numpy()

# Resize heatmap for drawing
heatmap_resized = tf.image.resize(heatmap[..., tf.newaxis], (150, 150))
heatmap_resized = tf.squeeze(heatmap_resized).numpy()

# ==========================================
# 6. DRAWING THE DASHBOARD
# ==========================================
print("Generating 6-Panel Dashboard...")

# Dark grey theme
DARK_GREY = '#2d2d2d'
plt.rcParams.update({
    'text.color': 'white',
    'axes.facecolor': DARK_GREY,
    'axes.edgecolor': '#555555',
    'axes.labelcolor': 'white',
    'xtick.color': 'white',
    'ytick.color': 'white',
    'grid.color': '#555555',
    'figure.facecolor': DARK_GREY,
})

# Create a large window for our grid
plt.figure(figsize=(18, 10), facecolor=DARK_GREY)
plt.suptitle("AI Diagnostic Dashboard", fontsize=20, fontweight='bold', color='white')

original_img_array = tf.keras.utils.img_to_array(original_img) / 255.0

# --- PANEL 1: Original Image ---
plt.subplot(2, 3, 1)
plt.title(f"Prediction: {label}", fontsize=14, color='white')
plt.imshow(original_img_array)
plt.axis('off')

# --- PANEL 2: Rainbow Heatmap ---
plt.subplot(2, 3, 2)
plt.title("Grad-CAM Heatmap", fontsize=14, color='white')
colormap = plt. get_cmap("jet")
heatmap_colors = colormap(heatmap_resized)[:, :, :3]
superimposed_img = np.clip(heatmap_colors * 0.4 + original_img_array, 0, 1)
plt.imshow(superimposed_img)
plt.axis('off')

# --- PANEL 3: Confidence Graph ---
ax3 = plt.subplot(2, 3, 3)
ax3.set_facecolor(DARK_GREY)
ax3.set_title("Confidence Distribution", fontsize=14, color='white')
bars = ax3.bar(['Cat', 'Dog'], [cat_prob * 100, dog_prob * 100], color=['#3498db', '#e74c3c'])
ax3.set_ylim(0, 100)
ax3.set_ylabel("Probability (%)", color='white')
ax3.tick_params(colors='white')
ax3.spines['bottom'].set_color('#555555')
ax3.spines['top'].set_color('#555555')
ax3.spines['left'].set_color('#555555')
ax3.spines['right'].set_color('#555555')
# Add the exact percentages on top of the bars
for bar in bars:
    yval = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold', color='white')

# --- PANEL 4: Shallow X-Ray ---
plt.subplot(2, 3, 4)
plt.title("Shallow X-Ray (Edges/Outlines)", fontsize=14, color='white')
plt.imshow(shallow_features[:, :, 0], cmap='viridis')
plt.axis('off')

# --- PANEL 5: Deep X-Ray ---
plt.subplot(2, 3, 5)
plt.title("Deep X-Ray (Abstract Textures)", fontsize=14, color='white')
# We show filter #5 from the deeper layer (it looks more blocky/abstract)
plt.imshow(deep_features[:, :, 5], cmap='plasma')
plt.axis('off')

# --- PANEL 6: AI Focus Spotlight ---
plt.subplot(2, 3, 6)
plt.title("AI Focus Spotlight", fontsize=14, color='white')
# Multiply the original image by the heatmap. Cold areas become black!
focus_spotlight = original_img_array * heatmap_resized[..., np.newaxis]
plt.imshow(np.clip(focus_spotlight, 0, 1))
plt.axis('off')

# Render the window
plt.tight_layout()
plt.subplots_adjust(top=0.90) # Give room for the big title
plt.show()