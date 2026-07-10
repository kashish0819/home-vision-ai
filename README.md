# 🩸 homevision-ai : Multimodal AI-Based Anemia Detection

AnemiaFusionNet is a deep learning-based healthcare application that predicts anemia by combining retinal eye images and clinical blood parameters. The project uses a multimodal fusion network built with PyTorch and provides a Flask-based web interface for real-time predictions and PDF report generation.

---

## ✨ Features

* 🧠 Multimodal AI model (Retinal Image + Clinical Data)
* 👁️ CNN-based retinal image analysis
* 📊 Clinical feature integration
* ⚡ Real-time anemia prediction
* 📄 Automatic PDF report generation
* 🌐 User-friendly Flask web application
* 💾 Trained model loading for instant inference

---

## 🛠️ Tech Stack

* Python
* Flask
* PyTorch
* TorchVision
* TIMM
* OpenCV
* NumPy
* Pandas
* Scikit-learn
* ReportLab
* HTML
* CSS
* JavaScript

---

## 📂 Project Structure

```text
homevision-ai/
│
├── AnemiaFusionNet_Web/      # Flask Web Application
├── ml/                       # AI Models & Prediction
├── data/                     # Dataset
├── saved_models/             # Trained Model Weights
├── database/                 # Prediction Records
├── utils/                    # Utility Functions
├── outputs/                  # Generated Reports
└── requirements.txt
```

---

## 🚀 Installation

```bash
git clone <repository-url>
cd homevision-ai

python -m venv venv

# Windows
venv\Scripts\activate

pip install -r requirements.txt
```

---

## ▶️ Run the Application

```bash
cd AnemiaFusionNet_Web
python app.py
```

Open your browser and visit:

```text
http://127.0.0.1:5000
```

---

## 🧪 Workflow

1. Upload a retinal eye image.
2. Enter patient clinical information.
3. The AI model extracts image features using a CNN.
4. Clinical features are fused with image features.
5. The model predicts **Normal** or **Anemia**.
6. View the confidence score and download the PDF report.

---

## 📊 Model

* CNN-based image feature extractor
* Clinical feature encoder
* Fusion Neural Network
* Binary Classification (Normal / Anemia)

---

## 📌 Future Improvements

* Grad-CAM explainability
* Multi-class anemia severity prediction
* Cloud deployment
* REST API
* Mobile application support
* Electronic Health Record (EHR) integration

---

## 👨‍💻 Author

**Kashish Gupta**

If you find this project useful, consider giving it a ⭐ on GitHub.
