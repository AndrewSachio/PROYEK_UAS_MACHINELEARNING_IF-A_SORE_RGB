import streamlit as st
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# KONFIGURASI STREAMLIT
# =====================================================
st.set_page_config(
    page_title="Analisis Sentimen LSTM vs BiLSTM",
    layout="centered"
)

# =====================================================
# LOAD MODEL & TOOL (INFERENCE ONLY)
# =====================================================
@st.cache_resource
def load_resources():
    model_lstm = load_model("model_lstm.h5", compile=False)
    model_bilstm = load_model("model_bilstm.h5", compile=False)

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    return model_lstm, model_bilstm, tokenizer, label_encoder


model_lstm, model_bilstm, tokenizer, label_encoder = load_resources()

MAX_LENGTH = 80

# =====================================================
# AKURASI MODEL (HASIL TRAINING)
# =====================================================
LSTM_ACCURACY = 0.87
BILSTM_ACCURACY = 0.89

# =====================================================
# PREPROCESSING
# =====================================================
def remove_double_char(text):
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def clean_text(text):
    text = text.lower()
    text = remove_double_char(text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokenize(text):
    return text.split()

# ❗ STOPWORD TANPA KATA NEGASI
STOPWORDS = {
    "yang","dan","di","ke","dari","ini","itu","untuk","dengan",
    "pada","adalah","karena","saya","aku","kamu","nya"
}

def remove_stopwords(tokens):
    return [w for w in tokens if w not in STOPWORDS]

# ✅ PENANGANAN NEGASI
def handle_negation(tokens):
    negations = {"tidak", "bukan", "jangan", "kurang"}
    result = []
    skip = False

    for i in range(len(tokens)):
        if skip:
            skip = False
            continue

        if tokens[i] in negations and i + 1 < len(tokens):
            result.append(tokens[i] + "_" + tokens[i + 1])
            skip = True
        else:
            result.append(tokens[i])

    return result

# =====================================================
# UI STREAMLIT
# =====================================================
st.title("📊 Analisis Sentimen Ulasan Roblox")
st.write(
    "Aplikasi ini menampilkan **seluruh proses preprocessing NLP**, "
    "**hasil prediksi LSTM & BiLSTM**, serta **akurasi model**."
)

text_input = st.text_area("Masukkan teks ulasan:")

if st.button("Analisis Sentimen"):
    if text_input.strip() == "":
        st.warning("Teks tidak boleh kosong.")
    else:
        # ===============================
        # PIPELINE NLP
        # ===============================
        cleaned = clean_text(text_input)
        tokens = tokenize(cleaned)
        tokens_no_stop = remove_stopwords(tokens)
        tokens_negation = handle_negation(tokens_no_stop)
        final_text = " ".join(tokens_negation)

        sequence = tokenizer.texts_to_sequences([final_text])
        padded = pad_sequences(sequence, maxlen=MAX_LENGTH, padding='post')

        # ===============================
        # PREDIKSI MODEL
        # ===============================
        prob_lstm = float(model_lstm.predict(padded, verbose=0)[0][0])
        prob_bilstm = float(model_bilstm.predict(padded, verbose=0)[0][0])

        label_lstm = label_encoder.inverse_transform([int(prob_lstm > 0.5)])[0]
        label_bilstm = label_encoder.inverse_transform([int(prob_bilstm > 0.5)])[0]

        # ===============================
        # OUTPUT PREPROCESSING
        # ===============================
        st.subheader("🔍 Tahapan Preprocessing")

        st.markdown("**Teks Asli**")
        st.code(text_input)

        st.markdown("**Cleaning Text**")
        st.code(cleaned)

        st.markdown("**Tokenisasi**")
        st.write(tokens)

        st.markdown("**Stopword Removal**")
        st.write(tokens_no_stop)

        st.markdown("**Handling Negation**")
        st.write(tokens_negation)

        st.markdown("**Final Text (Input Model)**")
        st.code(final_text)

        st.markdown("**Tokenizer Sequence**")
        st.write(sequence)

        st.markdown("**Padding**")
        st.write(padded.tolist())

        # ===============================
        # OUTPUT MODEL
        # ===============================
        st.subheader("🤖 Hasil Prediksi Model")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### 🔵 LSTM")
            st.write(f"Probabilitas : `{prob_lstm:.4f}`")
            st.write(f"Akurasi Model : `{LSTM_ACCURACY*100:.0f}%`")
            st.success(f"Sentimen: {label_lstm}") if label_lstm == "Positif" else st.error(f"Sentimen: {label_lstm}")

        with col2:
            st.markdown("### 🟣 BiLSTM")
            st.write(f"Probabilitas : `{prob_bilstm:.4f}`")
            st.write(f"Akurasi Model : `{BILSTM_ACCURACY*100:.0f}%`")
            st.success(f"Sentimen: {label_bilstm}") if label_bilstm == "Positif" else st.error(f"Sentimen: {label_bilstm}")
