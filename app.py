import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import onnxruntime as ort
from PIL import Image
import pytesseract
import io

# Load CSV
df = pd.read_csv("medicines.csv")[['Medicine Name', 'Uses']].dropna()
uses_list = df['Uses'].tolist()

# ONNX & Tokenizer
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
ort_session = ort.InferenceSession("sentence_model.onnx")

# OCR Function
def extract_text_from_image(image):
    image = Image.open(image)
    return pytesseract.image_to_string(image)

# Encode text using ONNX model
def encode_text_with_onnx(text):
    inputs = tokenizer(text, return_tensors='np', padding='max_length', truncation=True, max_length=128)
    input_ids = inputs['input_ids'].astype('int64')  # Fix type
    outputs = ort_session.run(None, {'input_ids': input_ids})
    return np.mean(outputs[0], axis=1)

# Precompute embeddings
@st.cache_resource
def get_all_embeddings():
    return np.vstack([encode_text_with_onnx(text) for text in uses_list])

embeddings = get_all_embeddings()

# Recommend function
def recommend_meds(symptom_text):
    query_emb = encode_text_with_onnx(symptom_text)
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(scores)[-5:][::-1]
    return df.iloc[top_indices]

# Streamlit UI
st.title("💊 Medicine Recommender")

st.write("📷 Upload a photo of a medicine (label) or describe the symptoms.")

option = st.radio("Choose input method:", ["Upload Image", "Enter Text"])

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload medicine image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        extracted_text = extract_text_from_image(uploaded_file)
        st.text_area("🧠 OCR Extracted Text", extracted_text)
        if st.button("Recommend"):
            results = recommend_meds(extracted_text)
            st.dataframe(results[['Medicine Name', 'Uses']])
else:
    user_text = st.text_input("Enter symptoms (e.g., fever, headache, cold)")
    if st.button("Recommend"):
        results = recommend_meds(user_text)
        st.dataframe(results[['Medicine Name', 'Uses']])
