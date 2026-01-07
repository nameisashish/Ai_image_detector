# 🧠 AI Image Detection App
**Detect whether an image is AI-generated or real — instantly.**

A Streamlit-based web application that analyzes uploaded images and predicts whether they are **AI-generated or authentic**. Built for fraud prevention, content verification, and trust-based platforms.

---

## 🚀 Features

- 📤 Image upload via browser (Localhost first)
- 🧠 AI vs Real image classification
- 🚫 Snapchat images are **not allowed**
- 🖼️ Supports **HEIC (.heic)** image format
- ⚡ Fast inference and lightweight backend
- 🎨 Clean, modern, and intuitive Streamlit UI

---

## 🧩 Tech Stack

- **Frontend**: Streamlit  
- **Backend**: Python  
- **ML/DL**: PyTorch / TensorFlow  
- **Image Processing**: Pillow, OpenCV  
- **HEIC Support**: pillow-heif  

---

## 📁 Project Structure

```
├── image_detector.py        # Core backend logic
├── app.py                   # Streamlit application
├── model/                   # Trained model files
├── utils/                   # Helper functions
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
```

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/ai-image-detector.git
cd ai-image-detector
```

### 2. Create and activate virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App (Localhost)

```bash
streamlit run app.py
```

Open in browser:
```
http://localhost:8501
```

---

## 📸 Supported Image Formats

| Format | Status |
|------|-------|
| JPG / JPEG | ✅ |
| PNG | ✅ |
| HEIC | ✅ |
| Snapchat Images | ❌ |

---

## 🔍 How It Works

1. User uploads an image  
2. Metadata and visual checks are performed  
3. Snapchat images are rejected  
4. Image is passed through the AI model  
5. Output:
   - **AI Generated Image**
   - **Real Image**

---

## 🛡️ Use Cases

- E-commerce refund fraud detection
- Fake image verification
- Social media moderation
- Trust-based customer support systems

---

## 🧪 Project Status

- ✅ Localhost version ready
- 🔄 UI enhancements in progress
- 🚀 Production deployment planned

---

## 🔮 Future Improvements

- Confidence score for predictions
- Explainability heatmaps
- REST API (FastAPI)
- Cloud deployment (AWS / GCP)

---

## 👨‍💻 Author

**Ashish Kishore**  
AI | Deep Learning | Applied Research

---

## 📜 License
 
Commercial use requires permission.
