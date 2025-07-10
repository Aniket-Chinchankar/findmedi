import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer
import onnxruntime as ort
from PIL import Image
import easyocr

# ---- Load CSV ----
@st.cache_data
def load_data():
    df = pd.read_csv("medicines.csv")[['Medicine Name', 'Uses']].dropna()
    return df, df['Uses'].tolist()

df, uses_list = load_data()

# ---- Load Tokenizer & ONNX Model ----
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    ort_session = ort.InferenceSession("sentence_model.onnx")
    return tokenizer, ort_session

tokenizer, ort_session = load_model()

# ---- OCR Function using EasyOCR ----
def extract_text_from_image(image_file):
    reader = easyocr.Reader(['en'], gpu=False)
    image = Image.open(image_file).convert("RGB")
    result = reader.readtext(np.array(image))
    return " ".join([text[1] for text in result])

# ---- ONNX Encoder ----
def encode_text_with_onnx(text):
    inputs = tokenizer(text, return_tensors='np', padding='max_length', truncation=True, max_length=128)
    input_ids = inputs['input_ids'].astype('int64')  # ONNX needs int64
    outputs = ort_session.run(None, {'input_ids': input_ids})
    return np.mean(outputs[0], axis=1)

# ---- Precompute Embeddings ----
@st.cache_resource
def get_all_embeddings():
    return np.vstack([encode_text_with_onnx(text) for text in uses_list])

embeddings = get_all_embeddings()

# ---- Recommender ----
def recommend_meds(symptom_text):
    query_emb = encode_text_with_onnx(symptom_text)
    scores = cosine_similarity(query_emb, embeddings)[0]
    top_indices = np.argsort(scores)[-5:][::-1]
    return df.iloc[top_indices]

# ---- Streamlit UI ----
st.set_page_config(page_title="Medicine Recommender", page_icon="💊")
st.title("💊 Smart Medicine Recommender")
st.markdown("Upload a **medicine image** or enter **symptoms** to get top recommended medicines.")

# Input Method Selection
option = st.radio("Choose Input Type:", ["📷 Upload Image", "✍️ Enter Symptoms"])

if option == "📷 Upload Image":
    uploaded_file = st.file_uploader("Upload Medicine Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)
        with st.spinner("🔍 Extracting text from image..."):
            extracted_text = extract_text_from_image(uploaded_file)
        st.text_area("🧠 Extracted Text from OCR", extracted_text, height=100)
        if st.button("💡 Recommend Medicines"):
            results = recommend_meds(extracted_text)
            st.success("✅ Recommendations based on extracted content:")
            st.dataframe(results[['Medicine Name', 'Uses']])

elif option == "✍️ Enter Symptoms":
    user_text = st.text_input("📝 Describe symptoms (e.g., headache, cold, cough)")
    if st.button("💡 Recommend Medicines"):
        results = recommend_meds(user_text)
        st.success("✅ Recommendations based on your symptoms:")
        st.dataframe(results[['Medicine Name', 'Uses']])
