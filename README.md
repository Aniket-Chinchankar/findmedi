
# 💊 Medicine Recommender System using OCR and Symptoms (Offline + AI Powered)

A smart AI-powered system that scans a medicine cover using offline OCR (Optical Character Recognition) and recommends appropriate medicines based on symptoms using semantic similarity via transformer models. The system is optimized to run on GPU (NVIDIA RTX 4060) and can be extended to use NPU acceleration.

---

## 🚀 Features

- 📸 **Offline OCR**: Extracts medicine name from a photo (taken from webcam/mobile camera)
- 🧠 **Symptom-Based Recommendation**: Uses sentence embeddings to recommend relevant medicines based on user-entered symptoms
- ⚡ **Hardware Acceleration**: Leverages GPU and NPU for faster inference
- 🔒 **Offline Capability**: Works without internet after initial model setup
- 📄 **ONNX Model Export**: Optimized model converted to ONNX for fast runtime execution

---

## 🖥️ Tech Stack

- Python 3.10+
- PyTorch (GPU)
- ONNX Runtime
- Streamlit (for frontend)
- Sentence-Transformers
- EasyOCR / Tesseract for offline OCR
- Pandas, NumPy, Scikit-learn

---

## 📂 Dataset

Used [11000+ Medicine Details Dataset](https://www.kaggle.com/datasets/singhnavjot2062001/11000-medicine-details) from Kaggle.

Contains:
- Medicine Name
- Uses (symptoms)
- Composition, Side Effects, Reviews, etc.

---

## 📷 How It Works

1. **Scan Image** using webcam or mobile camera.
2. **Extract Medicine Name** using OCR (EasyOCR/Tesseract).
3. **Search Dataset** for matched medicine.
4. **Recommend Similar Medicines** using symptom-based vector similarity via transformer model (`all-MiniLM-L6-v2`).

---

## 🔧 Installation

```bash
git clone https://github.com/your-username/medicine-recommender.git
cd medicine-recommender

# Create a virtual environment
python -m venv tf_env
.	f_env\Scriptsctivate      # Windows
# source tf_env/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

---

## 🧪 Run Locally

```bash
streamlit run app.py
```

> Make sure `medicines.csv` is in the root folder.

---

## 📦 Export Model to ONNX (Optional)

```python
from sentence_transformers import SentenceTransformer
import torch

model = SentenceTransformer('all-MiniLM-L6-v2')
model.eval()
dummy_input = torch.randint(0, 30522, (1, 128)).to('cuda')
torch.onnx.export(
    model._first_module().auto_model,
    dummy_input,
    "sentence_model.onnx",
    input_names=['input_ids'],
    output_names=['last_hidden_state'],
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'seq_len'},
        'last_hidden_state': {0: 'batch_size', 1: 'seq_len'}
    },
    opset_version=14
)
```

---

## 🌐 Deployment

You can deploy this app easily on **Streamlit Community Cloud**:

1. Push your code to a GitHub repo.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect GitHub repo and deploy.

---

## 🤝 Contributions

Pull requests are welcome. For major changes, please open an issue first.

---

## 📜 License

MIT License © 2025 [Your Name]
