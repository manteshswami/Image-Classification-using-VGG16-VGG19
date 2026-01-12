## 🧠 Image Classification using VGG16 & VGG19

 *Pre-trained VGG16 and VGG19 models are used to classify real-world images using deep learning and transfer learning.*

### 🚀 Project Overview

This project uses CNN-based pretrained models trained on ImageNet to detect and classify objects from input images such as flowers, animals, etc.

### 🛠 Tech Stack
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib

### 📌 Step-by-Step Implementation
🔹 Step 1 — Import Libraries
<pre><code>
  import tensorflow as tf 
  from tensorflow.keras.applications import VGG16, VGG19 
  from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions 
  from tensorflow.keras.preprocessing.image import load_img, img_to_array 
  import numpy as np 
  import matplotlib.pyplot as plt </code></pre>
🔹 Step 2 — Load Pretrained Models
<pre><code> 
  model_1 = VGG16(weights="imagenet") 
  model_2 = VGG19(weights="imagenet") 
</code></pre>
🔹 Step 3 — Create Prediction Function
<pre><code> 
  def predict_image(model, img_path):
    img = load_img(img_path, target_size=(224, 224)) 
    img_array = img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = preprocess_input(img_array) 
    prediction = model.predict(img_array) 
    return decode_predictions(prediction, top=5)[0] 
</code></pre>
🔹 Step 4 — Predict Using VGG16
<pre><code> 
  image_path = "/content/green_mamba.jpg" 
  predictions = predict_image(model_1, image_path) 
  for i, (img_id, label, score) in enumerate(predictions): 
  print(f"{i+1}. {label} - {score*100:.2f}%") 
</code></pre>
🔹 Step 5 — Display Result
<pre><code> 
  img = load_img(image_path) 
  plt.imshow(img) plt.title(f"Prediction: {predictions[0][1]} ({predictions[0][2]*100:.2f}%)") 
  plt.axis("off") 
  plt.show() 
</code></pre>
🔹 Step 6 — Predict Using VGG19
<pre><code> 
  image_path = "/content/ele.jpg" 
  predictions = predict_image(model_2, image_path) 
  for i, (img_id, label, score) in enumerate(predictions):
  print(f"{i+1}. {label} - {score*100:.2f}%") 
</code></pre>
🔹 Step 7 — Display Result
<code><pre>
  img = load_img(image_path) 
  plt.imshow(img) 
  plt.title(f"Prediction: {predictions[0][1]} ({predictions[0][2]*100:.2f}%)") 
  plt.axis("off") 
  plt.show() 
</code></pre>

### 🎯 Output
 *The model correctly predicts objects such as:*
- Green Mamba Snake
- African Elephant
  
### 🔄 Transfer learning
 *Updating soon*
