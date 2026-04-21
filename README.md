# Fashion MNIST Image Classifier (Streamlit App)

## 📌 Project Overview

This project is a **Deep Learning-based Fashion Item Classifier** built using a **Convolutional Neural Network (CNN)** trained on the **Fashion MNIST dataset**.

It also includes a **Streamlit web application** that allows users to upload an image and classify it into one of the fashion categories.

---

## 🧠 Model Details

* Dataset: **Fashion MNIST**
* Input Size: **28 × 28 grayscale images**
* Model Type: **CNN (Convolutional Neural Network)**
* Framework: **TensorFlow / Keras**
* Accuracy: ~**90–92%** on test data

### 🏷️ Classes

```text
0 - T-shirt/Top
1 - Trouser
2 - Pullover
3 - Dress
4 - Coat
5 - Sandal
6 - Shirt
7 - Sneaker
8 - Bag
9 - Ankle Boot
```

---

## ⚙️ Features

* Image classification using CNN
* Real-time prediction using Streamlit
* Image preprocessing pipeline
* Confidence-based prediction filtering
* Data augmentation for improved training
* Visualization of preprocessed input

---

## 📂 Project Structure

```
FashionMNIST/
│── main.py                      # Streamlit app
│── model_training.ipynb         # Model training notebook
│── trained_fashionMNIST_model.keras  # Saved model
│── README.md                   # Project documentation
```

---

## 🚀 How to Run the Project

### 1️⃣ Install Dependencies

```bash
pip install tensorflow streamlit numpy pillow matplotlib
```

---

### 2️⃣ Run the Streamlit App

```bash
streamlit run main.py
```

---

### 3️⃣ Open in Browser

```
http://localhost:8501
```

---

## 🖼️ Image Preprocessing

Before prediction, images are:

* Converted to **grayscale**
* Resized to **28×28**
* Centered using padding
* Background normalized (inverted if needed)
* Scaled between **0–1**

---

## ⚠️ Important Limitation

This model is trained on **Fashion MNIST**, which contains:

* Low-resolution (28×28) images
* Black background
* Centered clothing items

👉 Therefore, it may give **incorrect predictions on real-world images** due to:

* Different lighting
* Complex backgrounds
* Higher resolution
* Color differences

---

## 💡 Tips for Better Predictions

* Use **plain background images**
* Keep the object **centered**
* Avoid clutter and shadows
* Use **high-contrast images**

---

## 📈 Model Improvements

* Added **Data Augmentation**:

  * Rotation
  * Zoom
  * Width/Height shifts
* Used:

  * Batch Normalization
  * Dropout (0.5)

---

## 🔍 Sample Prediction Logic

From your Streamlit app: 

```python
predicted_class = np.argmax(probabilities)
confidence = probabilities[predicted_class]
```

Prediction is shown only if confidence ≥ 50%.

---

## 🧪 Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Streamlit
* PIL (Image Processing)
* Matplotlib

---

## 🎯 Future Improvements

* Use **transfer learning (MobileNetV2)**
* Train on **real-world fashion datasets**
* Increase input size (e.g., 128×128)
* Deploy on cloud (Streamlit Cloud)

---

## 🧸 Simple Explanation

This project teaches a model using **simple black-and-white clothing images**.
When tested on real photos, it may struggle because those images are more complex.

---

## ⭐ Final Note

✔️ Your model is correctly trained
✔️ Your app is working perfectly
⚠️ Wrong predictions are expected due to dataset limitations

---

## 🙌 Author

Developed as part of a Deep Learning / AI project using Fashion MNIST.
