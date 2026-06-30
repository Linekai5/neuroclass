# NeuroClass — AI Pet Classifier

A full end-to-end deep learning project that trains, tests, visualizes, and deploys a real-time Cat vs. Dog image classifier powered by **MobileNetV2 Transfer Learning** and **Grad-CAM** (Gradient-weighted Class Activation Mapping). The project spans three phases: a simple digit classifier prototype, a full image recognition pipeline, and a live camera classifier with mobile support.

---

## Project Structure

```
neuroclass/
│
├── tenc/                          # Phase 1 — Digit Classifier Prototype
│   ├── main.py                    # Trains a binary digit classifier (6 vs 8) on MNIST
│   ├── requirements.txt           # Dependencies for this phase
│   └── binary_classifier.keras    # Saved trained model output
│
├── catdogclassify/                # Phase 2 — Image Classifier Training & Analysis
│   ├── clean.py                   # Dataset cleaner (removes corrupt/broken images)
│   ├── main.py                    # Trains the MobileNetV2 Cat vs. Dog model
│   ├── testmodel.py               # Single-image prediction tester (CLI)
│   ├── predictvis.py              # 6-panel AI diagnostic dashboard with Grad-CAM
│   └── cat_dog_model.keras        # The trained model (used by everything else)
│
└── cameraclassifier/              # Phase 3 — Live Real-Time Classifier
    ├── main.py                    # Webcam classifier with live Grad-CAM HUD
    └── mobilemain.py              # Flask server + mobile browser camera classifier
```

---

## How Everything Fits Together

```
                    ┌──────────────────────┐
                    │   tenc/main.py       │  ← Phase 1: Learning the basics
                    │  (MNIST 6 vs 8)      │     Simple binary NN, TensorBoard
                    └──────────────────────┘

                              │
                              ▼

         ┌────────────────────────────────────────┐
         │         catdogclassify/                │
         │                                        │
         │  clean.py ──► main.py ──► model.keras  │  ← Phase 2: Real training pipeline
         │                   │                    │
         │            testmodel.py                │  ← Quick CLI prediction test
         │            predictvis.py               │  ← Full visual analysis dashboard
         └────────────────────────────────────────┘

                              │
                              ▼

         ┌────────────────────────────────────────┐
         │         cameraclassifier/              │
         │                                        │
         │  main.py       ← Webcam (desktop)      │  ← Phase 3: Real-time deployment
         │  mobilemain.py ← Flask + mobile web    │
         └────────────────────────────────────────┘
```

---

## File-by-File Breakdown

---

### `tenc/main.py` — The Prototype (Phase 1)

**What it does:**  
This is where the learning journey began. It trains the very first neural network — a simple binary classifier that distinguishes between **handwritten 6s and 8s** from the MNIST dataset.

**How it works step by step:**
1. **Loads the MNIST dataset** — a famous collection of 70,000 handwritten digit images (28×28 pixels, grayscale) built directly into TensorFlow.
2. **Filters the data** — keeps only images labeled `6` or `8`, discarding all other digits.
3. **Re-labels** — `8 → 0` and `6 → 1` to make it a binary problem.
4. **Normalizes pixels** — divides all pixel values by 255.0 so they fall between 0 and 1 (neural networks train much better on small numbers).
5. **Builds a small neural network:**
   - `Flatten` — unrolls the 28×28 grid into a flat list of 784 numbers.
   - `Dense(32, relu)` — 32 neurons that detect patterns.
   - `Dense(1, sigmoid)` — one output neuron that outputs a probability (0 = "8", 1 = "6").
6. **Trains for 5 epochs** with the Adam optimizer and binary cross-entropy loss.
7. **Logs to TensorBoard** so you can view training graphs in the browser.
8. **Saves the trained model** as `binary_classifier.keras`.

**Key concepts introduced:** neural network basics, binary classification, data normalization, model saving, TensorBoard logging.

---

### `catdogclassify/clean.py` — Dataset Cleaner

**What it does:**  
Before training on thousands of real photos, the dataset must be cleaned. The Microsoft Kaggle `PetImages` dataset contains some corrupt, truncated, or non-image files that would crash the training script. This file eliminates them.

**How it works step by step:**
1. **Loops through** both the `PetImages/Cat` and `PetImages/Dog` folders.
2. **Deletes hidden Mac files** (like `.DS_Store`) which start with a dot — these are system files, not images.
3. **Attempts to decode each image** using TensorFlow's `tf.io.decode_image()`. If this throws any exception, the file is broken.
4. **Deletes broken files** silently and counts how many were removed.
5. **Reports** the final count of deleted files.

**Why this matters:** If even one corrupt image makes it into training, TensorFlow will throw a cryptic error and the entire training run will fail. Running `clean.py` first is essential.

---

### `catdogclassify/main.py` — The Training Engine

**What it does:**  
This is the core of the project. It trains a high-accuracy Cat vs. Dog image classifier using **Transfer Learning** from Google's pre-trained **MobileNetV2** model.

**How it works step by step:**

**1. Data Pipeline:**
- Reads the `PetImages/` directory and automatically splits it **80% training / 20% validation** using `image_dataset_from_directory`.
- All images are resized to **150×150 pixels** and loaded in **batches of 32**.

**2. Loading MobileNetV2:**
- Downloads **MobileNetV2** pre-trained on **ImageNet** (1.2 million images, 1000 categories).
- `include_top=False` removes the original 1000-class output — we only want its feature-extraction "brain", not its decision layer.
- `base_model.trainable = False` **freezes** all MobileNetV2 weights so training doesn't destroy the pre-learned knowledge.

**3. Building the New Model (Sequential):**
```
Input (150×150×3)
  → RandomFlip (horizontal mirror augmentation)
  → RandomRotation (±10°)
  → RandomZoom (±10%)
  → Rescaling (pixels from 0–255 to -1 to +1, required by MobileNetV2)
  → MobileNetV2 (frozen — extracts features)
  → GlobalAveragePooling2D (condenses feature maps into a single vector)
  → Dropout(0.2) (randomly turns off 20% of neurons to prevent overfitting)
  → Dense(1, sigmoid) (final output: 0.0 = Cat, 1.0 = Dog)
```

**4. Training:**
- **Adam optimizer** with a low learning rate of `0.0001` (standard for transfer learning — we're only training the new top layers, so large updates would break things).
- **Binary cross-entropy loss** (the standard loss function for yes/no decisions).
- **EarlyStopping** with `patience=3` — if validation loss doesn't improve for 3 epochs in a row, training stops automatically and the best weights are restored. This prevents wasted compute and overfitting.
- Maximum of 20 epochs, but early stopping usually kicks in earlier.

**5. Saving:**
- The trained model is saved as `cat_dog_model.keras` — this single file is used by every other script in the project.

---

### `catdogclassify/testmodel.py` — Quick CLI Tester

**What it does:**  
A simple command-line script to make a one-shot prediction on a single image file. Good for quickly checking if the model works.

**How it works:**
1. Loads `cat_dog_model.keras`.
2. Loads a single image from `testphotos/dog.png`, resized to 150×150.
3. Wraps it in a batch dimension (`tf.expand_dims`) since Keras always expects batches.
4. Calls `model.predict()` and reads the raw sigmoid output (0.0–1.0).
5. **Interprets:** `>= 0.5` → Dog, `< 0.5` → Cat, with percentage certainty printed.

> Note: The model already contains a `Rescaling` layer internally, so the raw pixel values (0–255) are passed in directly — no manual normalization needed here.

---

### `catdogclassify/predictvis.py` — The AI Diagnostic Dashboard

**What it does:**  
This is the most visually rich script. It runs a single image through the model and generates a **6-panel matplotlib dashboard** that shows exactly what the AI is "thinking" and "seeing" at every layer.

**The 6 Panels:**

| Panel | Name | What it shows |
|---|---|---|
| 1 | **Original Image** | The input photo with the prediction label |
| 2 | **Grad-CAM Heatmap** | Rainbow overlay showing which pixels influenced the decision |
| 3 | **Confidence Distribution** | Bar chart of Cat% vs Dog% probabilities |
| 4 | **Shallow X-Ray** | Output of the first Conv2D layer — edges and outlines |
| 5 | **Deep X-Ray** | Output of a mid-network Conv2D — abstract textures and shapes |
| 6 | **AI Focus Spotlight** | Original image multiplied by the heatmap — cold areas go black |

**How Grad-CAM works (the heatmap):**
1. The model is **split** into two sub-models: everything up to the last Conv2D layer, and everything after it.
2. A **`GradientTape`** records all the math as it flows forward through both halves.
3. The gradients of the final prediction score are computed with respect to the last conv layer's output — this tells us "how much did each feature map pixel contribute to this prediction?"
4. The gradients are **averaged** across all feature maps (`reduce_mean`).
5. The result is a **weighted sum** of the feature maps — areas with high weights "light up" on the heatmap.
6. Negative values are clipped to 0 (ReLU), and the result is normalized 0→1.

This technique lets you see **exactly where the AI was looking** when it made its decision.

---

### `cameraclassifier/main.py` — Live Webcam Classifier

**What it does:**  
This is the desktop live-camera version. It opens your MacBook webcam and runs the classifier in **real-time at 30+ FPS**, drawing a live Grad-CAM heatmap directly on top of the video feed.

**How it works step by step:**

**1. Model Loading & Brain Splitting:**
- Loads `cat_dog_model.keras`.
- Extracts all preprocessing layers (Rescaling, etc.) that come before MobileNetV2.
- Splits MobileNetV2 into two sub-models at the last Conv2D layer, exactly like `predictvis.py` — required for Grad-CAM.

**2. `@tf.function` Optimization:**
```python
@tf.function
def get_prediction_and_heatmap(img_tensor):
    ...
```
This decorator compiles the prediction + Grad-CAM math into **native C++ code** the first time it's called. Every subsequent call runs at maximum speed without Python overhead — this is what enables real-time performance.

**3. Webcam Loop:**
- Opens camera with `cv2.VideoCapture(0)`.
- **Flips the frame horizontally** so it acts like a mirror (more natural for the user).
- **Center-crops** the frame to a square (the model was trained on square images).
- Resizes the crop to **150×150** and converts **BGR → RGB** (OpenCV uses BGR, TensorFlow uses RGB).

**4. Temporal Smoothing:**
- Keeps a rolling history of the last **10 predictions**.
- Averages them (`np.mean`) — this prevents the label from flickering on every frame due to minor image noise.

**5. Dynamic HUD:**
- **Orange box / "Searching..."** — when `0.10 < score < 0.90` (not confident enough)
- **Red box / "DOG (X%)"** — when score ≥ 0.90
- **Blue box / "CAT (X%)"** — when score ≤ 0.10

**6. Live Heatmap Overlay:**
- Only generated when confidence is high (avoids wasted compute at 30 FPS).
- The Grad-CAM heatmap is resized to match the crop area.
- A **hot mask** filters out cold heatmap pixels (below 40% intensity) so only the truly important regions glow.
- The colored heatmap is blended onto the video at **60% opacity** in the hot regions.

---

### `cameraclassifier/mobilemain.py` — Mobile Browser Classifier (Flask Server)

**What it does:**  
The most complex file in the project. It turns the classifier into a **web application** accessible from any phone on the same network (or anywhere via ngrok). Instead of reading a webcam directly, it runs a Flask web server. The phone's browser camera streams frames to the server for analysis and the results (including the heatmap) are sent back and overlaid on the live video in the browser.

**Architecture:**
```
[Phone Camera] → [Browser JavaScript] → [HTTP POST /predict] → [Flask + TensorFlow]
                                                                        ↓
[Phone Screen] ← [Heatmap PNG + Label] ←────────────────────── [JSON Response]
```

**How it works step by step:**

**1. Model & Grad-CAM Setup (Lines 1–90):**
Identical to `main.py` — loads the model, splits it for Grad-CAM, and compiles `get_prediction_and_heatmap` with `@tf.function`.

**The improvement:** The Grad-CAM target is **dynamic** — it always tracks the detected animal, not just the "dog" class:
```python
target_score = tf.where(dog_score >= 0.5, dog_score, 1.0 - dog_score)
```
This means the heatmap highlights the cat's features when a cat is detected, and the dog's features when a dog is detected.

**2. Flask App & Embedded HTML (Lines 93–376):**
The entire frontend UI is embedded as a string inside the Python file (`HTML_TEMPLATE`). Key frontend features:
- **Full-screen camera view** using the browser's `getUserMedia` API.
- **A "Start Live AI Camera" button** — required on iOS because Safari won't start a camera without a direct user gesture.
- **Front/rear camera toggle** — allows switching between the selfie and back cameras.
- **Client-side prediction smoothing** — same rolling-average technique as the desktop app, but implemented in JavaScript (window of 8 frames).
- **Confidence bar** — a thin animated bar at the top that fills based on confidence.
- **Result badge** — shows "Searching...", "CAT (X%)" or "DOG (X%)" with smooth color transitions.
- **Live heatmap overlay** — a transparent PNG sent from the server is positioned precisely over the camera feed using CSS.
- **Safe area insets** — respects iPhone notch/home bar with `env(safe-area-inset-*)`.

**3. `/predict` Route (Lines 386–430):**
- Receives a **JPEG frame as a base64-encoded string** in a JSON POST body.
- Decodes the base64 → bytes → NumPy array → OpenCV image.
- Center-crops and resizes to 150×150.
- Runs `get_prediction_and_heatmap()`.
- If confidence ≥ 80%, calls `_encode_heatmap()` to generate the heatmap image.
- Returns JSON: `{ success, prediction, heatmap }`.

**4. `_encode_heatmap()` Helper (Lines 436–455):**
- Takes the raw heatmap tensor and the frame dimensions.
- Resizes the heatmap to match the crop area.
- Applies the **JET colormap** (blue→green→yellow→red).
- Creates an **RGBA image** the same size as the full video frame.
- Only where the heatmap is "hot" (>100/255) does it set the alpha channel to 180 (semi-transparent). Cold areas stay fully transparent.
- Encodes the final PNG as a base64 string to send over JSON.

**5. Startup & ngrok (Lines 461–503):**
- Parses CLI arguments (`--port`, `--host`).
- **Optionally** creates an ngrok tunnel if `pyngrok` is installed — this gives the server a public HTTPS URL accessible from anywhere in the world.
- **Optionally** generates and prints a **QR code** in the terminal if `qrcode` is installed — scan it with your phone to instantly open the app.
- If neither library is available, it prints manual instructions.
- Starts the Flask server with `werkzeug` logging silenced to `WARNING` level.

---

## 🔬 Core Technologies Explained

### Transfer Learning
Instead of training from scratch on millions of images (which would take days), MobileNetV2 — pre-trained by Google on 1.2M ImageNet images — is used as a starting feature extractor. Only the final decision layer is trained from scratch. This means the model learns in minutes with much less data.

### Grad-CAM (Gradient-weighted Class Activation Mapping)
A technique that creates a visual explanation of what part of an image a CNN focused on. By computing gradients of the prediction score with respect to the last convolutional layer's outputs, it identifies which spatial regions were most important to the classification decision.

### Data Augmentation
Random flips, rotations, and zooms are applied during training. This artificially multiplies the variety of the dataset, preventing the model from memorizing specific training images and forcing it to learn general features (ears, fur texture, snout shape) instead.

### `@tf.function`
A TensorFlow decorator that traces and compiles Python/TensorFlow code into a static computation graph in optimized C++. The first call is slow (the tracing step), but all subsequent calls bypass Python entirely and run at near-native speed — critical for real-time video inference.

---

## How to Run Each Part

### Prerequisites
```bash
pip install tensorflow opencv-python flask pillow matplotlib numpy
# Optional for mobile server:
pip install pyngrok qrcode
```

### Phase 1 — Digit Prototype
```bash
cd tenc
python main.py
```

### Phase 2 — Train the Cat/Dog Model
```bash
cd catdogclassify
python clean.py        # 1. Clean the dataset first
python main.py         # 2. Train (takes a few minutes with GPU)
python testmodel.py    # 3. Test on a single image
python predictvis.py   # 4. View the full diagnostic dashboard
```

### Phase 3 — Live Webcam Classifier (Desktop)
```bash
cd cameraclassifier
python main.py         # Requires cat_dog_model.keras to exist in this or parent dir
# Press 'q' to quit
```

### Phase 3 — Mobile Browser Classifier
```bash
cd cameraclassifier
python mobilemain.py
# Open http://localhost:5000 in a browser
# Or scan the QR code to open on your phone (requires ngrok)
```

---

## Model File Dependency Map

```
cat_dog_model.keras  (trained by catdogclassify/main.py)
        │
        ├── catdogclassify/testmodel.py    (loads it for single-image testing)
        ├── catdogclassify/predictvis.py   (loads it for the dashboard)
        ├── cameraclassifier/main.py       (loads it for live webcam inference)
        └── cameraclassifier/mobilemain.py (loads it for the Flask API server)
```

---

## Model Architecture Summary

| Layer | Type | Purpose |
|---|---|---|
| Input | (150, 150, 3) | Raw RGB image |
| RandomFlip | Augmentation | Mirror left/right randomly |
| RandomRotation | Augmentation | Rotate ±10° randomly |
| RandomZoom | Augmentation | Zoom in/out ±10% randomly |
| Rescaling | Preprocessing | Scale pixels from [0,255] to [-1,1] |
| MobileNetV2 | Feature Extractor | 155 frozen layers, pre-trained on ImageNet |
| GlobalAveragePooling2D | Pooling | Compress feature maps to 1D vector |
| Dropout(0.2) | Regularization | Prevent overfitting |
| Dense(1, sigmoid) | Output | Probability score: 0=Cat, 1=Dog |

---

*Built with TensorFlow, OpenCV, Flask,*
