🧠 AI Image Detection App
Detect whether an image is AI-generated or real — instantly.
A powerful Streamlit-based web application that analyzes uploaded images and predicts whether they are AI-generated or authentic. Built for fraud prevention, content verification, and trust-based platforms.
🚀 Features
📤 Image Upload (Localhost First)
Upload images directly via browser
🧠 AI vs Real Image Classification
Uses a trained deep learning model
🚫 Snapchat Images Blocked
Automatically detects and rejects Snapchat-generated images
🖼️ Supports HEIC Images
Converts .heic files seamlessly
⚡ Fast & Lightweight
Optimized for local testing and future production scaling
🎨 Legendary UI (Streamlit)
Clean, modern, and intuitive interface
🧩 Tech Stack
Frontend: Streamlit
Backend: Python
ML/DL: PyTorch / TensorFlow (model-dependent)
Image Processing: Pillow, OpenCV
HEIC Support: pillow-heif
📁 Project Structure
├── image_detector.py        # Core backend logic
├── app.py                   # Streamlit application
├── model/                   # Trained model files
├── utils/                   # Helper functions
├── requirements.txt         # Dependencies
└── README.md                # Project documentation
⚙️ Installation
1️⃣ Clone the Repository
git clone https://github.com/your-username/ai-image-detector.git
cd ai-image-detector
2️⃣ Create Virtual Environment (Recommended)
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
▶️ Running Locally (Localhost)
streamlit run app.py
Then open:
http://localhost:8501
📸 Supported Image Formats
Format	Supported
JPG / JPEG	✅
PNG	✅
HEIC	✅
Snapchat Images	❌ (Blocked)
🔍 How It Works
User uploads an image
App checks metadata & visual patterns
Snapchat images are rejected instantly
Image is passed through AI model
Output:
✅ AI Generated
📷 Real Image
🛡️ Use Cases
🛒 E-commerce refund fraud prevention
📱 Social media content moderation
📰 News & media verification
🧾 Trust-based customer support systems
🧪 Current Status
✅ Localhost version complete
🔄 UI polishing & optimization
🚀 Production deployment planned
🔮 Future Enhancements
Confidence score (%)
Image heatmap explanation
API version (FastAPI)
Cloud deployment (AWS/GCP)
👨‍💻 Author
Ashish
AI | Deep Learning | Applied Research
“Trust the model. Verify the image.”
