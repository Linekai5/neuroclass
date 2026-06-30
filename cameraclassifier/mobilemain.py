import os
import sys
import base64
import io
import socket
import argparse
import numpy as np
import tensorflow as tf # type: ignore
import cv2 # type: ignore
from flask import Flask, render_template_string, request, jsonify # type: ignore
from PIL import Image

# =====================================================================
# CONFIGURATION
# =====================================================================
PORT = 5000
MODEL_FILENAME = "cat_dog_model.keras"

# =====================================================================
# 1. LOAD THE TRAINED MODEL
# =====================================================================
model_path = MODEL_FILENAME
if not os.path.exists(model_path):
    model_path = os.path.join("..", "catdogclassify", MODEL_FILENAME)
if not os.path.exists(model_path):
    raise FileNotFoundError(
        f"Could not find '{MODEL_FILENAME}'. "
        f"Tried: ./{MODEL_FILENAME} and {model_path}"
    )

print("Waking up the AI Brain for mobile streaming...")
model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# =====================================================================
# 2. EXTRACT BRAIN HALVES FOR FAST GRAD-CAM (DYNAMIC TRACKING)
# =====================================================================
print("Compiling Target Lock (Grad-CAM)...")

base_model = [layer for layer in model.layers if isinstance(layer, tf.keras.Model)][0]

prep_layers = []
for layer in model.layers:
    if isinstance(layer, tf.keras.Model):
        break
    prep_layers.append(layer)

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

@tf.function
def get_prediction_and_heatmap(img_tensor):
    """Returns (raw_prediction, heatmap) using dynamic binary class target assignment."""
    x = img_tensor
    for layer in prep_layers:
        x = layer(x, training=False)

    with tf.GradientTape() as tape:
        last_conv_output = last_conv_model(x, training=False)
        tape.watch(last_conv_output)
        preds = classifier_model(last_conv_output, training=False)
        
        # preds[:, 0] is the base Dog probability
        dog_score = preds[:, 0]
        
        # If it's a dog (>= 0.5), track the dog score. 
        # If it's a cat (< 0.5), track the cat score (1.0 - dog_score).
        target_score = tf.where(dog_score >= 0.5, dog_score, 1.0 - dog_score)

    grads = tape.gradient(target_score, last_conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = last_conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0.0)
    heatmap = heatmap / (tf.math.reduce_max(heatmap) + 1e-8)

    return preds[0][0], heatmap

print("Grad-CAM ready!")

# =====================================================================
# 3. FLASK APP & HTML UI
# =====================================================================
app = Flask(__name__)

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0,maximum-scale=1.0,user-scalable=no,viewport-fit=cover">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Pet Classifier</title>
<style>
  :root {
    --bg: #0a0a0a;
    --surface: rgba(10,10,10,0.75);
    --text: #ffffff;
    --accent-cat: #ff6b35;
    --accent-dog: #4da6ff;
    --searching: #ffb347;
    --radius: 16px;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  html, body {
    width: 100%; height: 100%;
    overflow: hidden;
    background-color: #0a0a0a;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    -webkit-tap-highlight-color: transparent;
    touch-action: manipulation;
  }
  #video-container { position: fixed; inset: 0; width: 100%; height: 100%; }
  #video { width: 100%; height: 100%; object-fit: cover; display: block; }
  #heatmap-overlay {
    position: fixed; inset: 0; width: 100%; height: 100%;
    pointer-events: none; z-index: 2; opacity: 0; transition: opacity 0.3s ease;
  }
  #heatmap-overlay.visible { opacity: 1; }
  #hud {
    position: fixed; top: 0; left: 0; right: 0; z-index: 10;
    padding: max(12px, env(safe-area-inset-top)) 16px 12px;
    display: flex; align-items: center; gap: 12px;
    background: linear-gradient(180deg, rgba(0,0,0,0.7) 0%, transparent 100%);
    pointer-events: none;
  }
  #result-badge {
    display: inline-flex; align-items: center; gap: 8px; padding: 10px 18px;
    border-radius: 40px; font-size: 18px; font-weight: 700; letter-spacing: 0.3px;
    color: #fff; backdrop-filter: blur(20px); -webkit-backdrop-filter: blur(20px);
    transition: background 0.4s ease, transform 0.3s ease;
  }
  #result-badge.searching { background: rgba(255,179,71,0.35); }
  #result-badge.cat       { background: rgba(255,107,53,0.55); }
  #result-badge.dog       { background: rgba(77,166,255,0.55); }
  #confidence-bar-wrap { flex: 1; height: 6px; border-radius: 3px; background: rgba(255,255,255,0.15); overflow: hidden; }
  #confidence-bar { height: 100%; width: 0%; border-radius: 3px; transition: width 0.35s ease, background 0.4s ease; background: var(--searching); }
  #switch-cam {
    position: fixed; bottom: max(24px, env(safe-area-inset-bottom)); right: 20px; z-index: 20;
    width: 64px; height: 52px; border-radius: 26px; border: none; background: rgba(255,255,255,0.15);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px); color: #fff; font-size: 14px; font-weight: bold;
    display: flex; align-items: center; justify-content: center; transition: background 0.2s;
  }
  #switch-cam:active { background: rgba(255,255,255,0.3); }
  
  #overlay-message {
    position: fixed; inset: 0; z-index: 50; display: flex; flex-direction: column;
    align-items: center; justify-content: center; background-color: #0a0a0a; color: #fff;
    font-size: 17px; gap: 20px; text-align: center; padding: 32px; transition: opacity 0.5s ease;
  }
  #overlay-message.hidden { opacity: 0; pointer-events: none; }
  
  #start-action-btn {
    padding: 16px 32px; font-size: 18px; font-weight: bold; border-radius: 40px;
    border: none; background: #fff; color: #000; cursor: pointer;
    box-shadow: 0 4px 20px rgba(255,255,255,0.15); transition: transform 0.2s;
  }
  #start-action-btn:active { transform: scale(0.95); }
  
  .spinner { display: none; width: 40px; height: 40px; border: 3px solid rgba(255,255,255,0.2); border-top-color: #fff; border-radius: 50%; animation: spin 0.8s linear infinite; }
  @keyframes spin { to { transform: rotate(360deg); } }
  #target-box { position: fixed; z-index: 3; pointer-events: none; border: 3px solid rgba(255,255,255,0.5); border-radius: 4px; transition: border-color 0.4s ease; }
</style>
</head>
<body>

<div id="overlay-message">
  <button id="start-action-btn">Start Live AI Camera</button>
  <div class="spinner" id="loading-spinner"></div>
  <div id="overlay-text">iOS requires physical interaction to boot cameras.</div>
</div>

<div id="video-container">
  <video id="video" autoplay playsinline muted></video>
  <canvas id="capture-canvas" style="display:none"></canvas>
</div>

<img id="heatmap-overlay" alt="" />
<div id="target-box"></div>

<div id="hud">
  <div id="result-badge" class="searching">Searching...</div>
  <div id="confidence-bar-wrap"><div id="confidence-bar"></div></div>
</div>

<button id="switch-cam" title="Switch camera">Switch</button>

<script>
  const video        = document.getElementById('video');
  const canvas       = document.getElementById('capture-canvas');
  const ctx          = canvas.getContext('2d');
  const heatmapImg   = document.getElementById('heatmap-overlay');
  const resultBadge  = document.getElementById('result-badge');
  const confBar      = document.getElementById('confidence-bar');
  const targetBox    = document.getElementById('target-box');
  const overlayMsg   = document.getElementById('overlay-message');
  const overlayText  = document.getElementById('overlay-text');
  const switchBtn    = document.getElementById('switch-cam');
  const startActionBtn = document.getElementById('start-action-btn');
  const loadingSpinner = document.getElementById('loading-spinner');

  let currentStream    = null;
  let facingMode       = 'environment'; 
  let predictTimer     = null;
  const PREDICT_EVERY  = 280;            
  const SMOOTH_WINDOW  = 8;              
  let predictionHistory = [];

  startActionBtn.addEventListener('click', () => {
    startActionBtn.style.display = 'none';
    loadingSpinner.style.display = 'block';
    overlayText.textContent = 'Initializing camera hardware...';
    startCamera();
  });

  function hideOverlay() { overlayMsg.classList.add('hidden'); }
  function showOverlay(text) { overlayText.textContent = text; overlayMsg.classList.remove('hidden'); }

  async function startCamera() {
    if (currentStream) { currentStream.getTracks().forEach(t => t.stop()); }

    const constraints = {
      video: { facingMode: { ideal: facingMode }, width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false
    };

    try {
      currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = currentStream;
      
      video.setAttribute('playsinline', 'true');
      video.setAttribute('autoplay', 'true');
      video.setAttribute('muted', 'true');
      
      await video.play();

      await new Promise(r => {
        if (video.videoWidth) r();
        else video.addEventListener('loadedmetadata', r, { once: true });
      });

      hideOverlay();
      updateTargetBox();
      startPredicting();
    } catch (err) {
      console.error(err);
      loadingSpinner.style.display = 'none';
      startActionBtn.style.display = 'block';
      overlayText.textContent = 'Camera access denied\\n\\nVerify that you granted camera permissions to Chrome/Safari.';
    }
  }

  function updateTargetBox() {
    const vw = video.videoWidth  || window.innerWidth;
    const vh = video.videoHeight || window.innerHeight;
    const displayW = window.innerWidth;
    const displayH = window.innerHeight;

    const videoRatio = vw / vh;
    const screenRatio = displayW / displayH;
    let renderedW, renderedH, offsetX, offsetY;

    if (videoRatio > screenRatio) {
      renderedH = displayH; renderedW = displayH * videoRatio;
      offsetX = (displayW - renderedW) / 2; offsetY = 0;
    } else {
      renderedW = displayW; renderedH = displayW / videoRatio;
      offsetX = 0; offsetY = (displayH - renderedH) / 2;
    }

    const cropDisplaySize = Math.min(renderedW, renderedH);
    const cropLeft = offsetX + (renderedW - cropDisplaySize) / 2;
    const cropTop  = offsetY + (renderedH - cropDisplaySize) / 2;

    targetBox.style.left   = cropLeft + 'px';
    targetBox.style.top    = cropTop + 'px';
    targetBox.style.width  = cropDisplaySize + 'px';
    targetBox.style.height = cropDisplaySize + 'px';

    window.__cropRect = { left: cropLeft, top: cropTop, width: cropDisplaySize, height: cropDisplaySize };
  }

  function startPredicting() {
    if (predictTimer) clearInterval(predictTimer);
    predictTimer = setInterval(captureAndPredict, PREDICT_EVERY);
  }

  async function captureAndPredict() {
    if (!video.videoWidth) return;

    const capW = 320;
    const capH = Math.round(capW * (video.videoHeight / video.videoWidth));
    canvas.width  = capW; canvas.height = capH;
    ctx.drawImage(video, 0, 0, capW, capH);

    const jpegBlob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', 0.75));
    if (!jpegBlob) return;

    const reader = new FileReader();
    const base64Promise = new Promise(r => { reader.onloadend = () => r(reader.result); });
    reader.readAsDataURL(jpegBlob);
    const dataUrl = await base64Promise;

    try {
      const resp = await fetch('/predict', {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'ngrok-skip-browser-warning': 'true' 
        },
        body: JSON.stringify({ image: dataUrl })
      });
      const data = await resp.json();
      if (!data.success) return;
      updateUI(data);
    } catch (err) {}
  }

  function updateUI(data) {
    const rawPred = data.prediction;
    predictionHistory.push(rawPred);
    if (predictionHistory.length > SMOOTH_WINDOW) { predictionHistory.shift(); }
    const smoothed = predictionHistory.reduce((a,b) => a+b, 0) / predictionHistory.length;

    const confidence = Math.abs(smoothed - 0.5) * 2;
    const confPct = Math.round(confidence * 100);
    confBar.style.width = confPct + '%';

    let state, label, barColor;
    if (confidence < 0.8) {
      state = 'searching'; label = 'Searching...'; barColor = 'var(--searching)';
    } else if (smoothed >= 0.90) {
      state = 'dog'; label = `DOG (${confPct}%)`; barColor = 'var(--accent-dog)';
    } else {
      state = 'cat'; label = `CAT (${confPct}%)`; barColor = 'var(--accent-cat)';
    }

    resultBadge.className = state;
    resultBadge.textContent = label;
    confBar.style.background = barColor;
    targetBox.style.borderColor = ({ searching: 'rgba(255,255,255,0.5)', cat: 'rgba(255,107,53,0.85)', dog: 'rgba(77,166,255,0.85)' })[state];

    if (data.heatmap && state !== 'searching') {
      heatmapImg.src = 'data:image/png;base64,' + data.heatmap;
      heatmapImg.classList.add('visible');
      const cr = window.__cropRect;
      if (cr) {
        heatmapImg.style.left = cr.left + 'px'; heatmapImg.style.top = cr.top + 'px';
        heatmapImg.style.width = cr.width + 'px'; heatmapImg.style.height = cr.height + 'px';
      }
    } else {
      heatmapImg.classList.remove('visible');
    }
  }

  switchBtn.addEventListener('click', () => {
    facingMode = (facingMode === 'environment') ? 'user' : 'environment';
    startCamera();
  });

  window.addEventListener('resize', () => { updateTargetBox(); });
  window.addEventListener('orientationchange', () => { setTimeout(updateTargetBox, 400); });
</script>
</body>
</html>"""

# =====================================================================
# 4. FLASK ROUTES
# =====================================================================

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        if not data or "image" not in data:
            return jsonify({"success": False, "error": "No image provided"}), 400

        image_b64 = data["image"]
        if "," in image_b64:
            image_b64 = image_b64.split(",", 1)[1]

        img_bytes = base64.b64decode(image_b64)
        np_arr = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return jsonify({"success": False, "error": "Invalid image"}), 400

        height, width = frame.shape[:2]

        crop_size = min(height, width)
        start_y = (height - crop_size) // 2
        start_x = (width - crop_size) // 2
        cropped = frame[start_y:start_y + crop_size, start_x:start_x + crop_size]

        img_resized = cv2.resize(cropped, (150, 150))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_tensor = tf.expand_dims(img_rgb, 0)

        raw_prediction, heatmap_tensor = get_prediction_and_heatmap(img_tensor)
        raw_prediction = float(raw_prediction)

        heatmap_b64 = None
        confidence = abs(raw_prediction - 0.5) * 2
        if confidence >= 0.8:
            heatmap_b64 = _encode_heatmap(heatmap_tensor.numpy(), crop_size, width, height)

        return jsonify({
            "success": True,
            "prediction": raw_prediction,
            "heatmap": heatmap_b64,
        })

    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500

# =====================================================================
# 5. HEATMAP ENCODING
# =====================================================================

def _encode_heatmap(heatmap, crop_size, frame_width, frame_height):
    heatmap_resized = cv2.resize(heatmap, (crop_size, crop_size))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    rgba = np.zeros((frame_height, frame_width, 4), dtype=np.uint8)
    hot_mask = heatmap_uint8 > 100

    start_y = (frame_height - crop_size) // 2
    start_x = (frame_width - crop_size) // 2

    rgba[start_y:start_y + crop_size, start_x:start_x + crop_size, :3] = colored_rgb
    rgba[start_y:start_y + crop_size, start_x:start_x + crop_size, 3] = np.where(hot_mask, 180, 0)

    img = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# =====================================================================
# 6. MAIN EXECUTION
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mobile Pet Classifier Server")
    parser.add_argument("--port", type=int, default=PORT, help=f"Port to listen on (default: {PORT})")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    args = parser.parse_args()

    print()
    print("=" * 56)
    print("  Mobile Pet Classifier Server Active")
    print("=" * 56)
    print(f"  Local Server Running On: http://localhost:{args.port}")
    print("=" * 56)

    try:
        from pyngrok import ngrok
        import qrcode
        
        # Open a ngrok tunnel to the local port
        public_url = ngrok.connect(args.port).public_url
        print(f"  Public ngrok URL: {public_url}")
        print("=" * 56)
        print("  Scan this QR Code with your mobile device:")
        print()
        
        # Generate and print QR code
        qr = qrcode.QRCode(version=1, box_size=10, border=2)
        qr.add_data(public_url)
        qr.make(fit=True)
        qr.print_ascii(invert=True)
        print("=" * 56)
        
    except ImportError:
        print("  Optional: Run 'pip install pyngrok qrcode' to automatically")
        print("  generate an ngrok URL and QR code in the terminal.")
        print("=" * 56)
        print(f"  STEP 2: Run 'ngrok http {args.port}' in another window.")
        print("=" * 56)

    import logging
    log = logging.getLogger("werkzeug")
    log.setLevel(logging.WARNING)

    app.run(host=args.host, port=args.port, debug=False)