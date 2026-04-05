import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer

# Page config
st.set_page_config(
    page_title="iPhone Sentiment Analyzer AI",
    page_icon="📱",
    layout="wide"
)

# Custom CSS (iPhone-inspired: sleek dark/blue)
st.markdown("""
<style>
.main { background-color: #000; color: white; }
.stButton > button {
    width: 100%; height: 3em; border-radius: 10px; 
    background: linear-gradient(45deg, #007AFF, #5856D6); color: white; font-weight: bold; border: none;
}
.metric-container { background-color: #1C1C1E; }
</style>
""", unsafe_allow_html=True)

# Load assets (cached)
@st.cache_resource
def load_assets():
    model = load_model('ann_model.keras')
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

try:
    model, tokenizer = load_assets()
    st.success("✅ Model & Tokenizer loaded (Accuracy ~70% on test)")
except Exception as e:
    st.error(f"❌ Load error: {e}. Run `python train_model.py` first!")

# Sidebar
with st.sidebar:
   # ✅ iPhone icon
    st.image("https://cdn-icons-png.flaticon.com/512/0/747.png", width=100)
    st.title("📱 iPhone Sentiment AI")
    st.info("Analyzes Amazon iPhone reviews → GOOD/BAD sentiment. Powered by ANN (TensorFlow). Trained on 3000+ reviews.")
    st.divider()
    st.warning("⚠️ Demo only. For awareness.")

# Main
st.title("🤖 iPhone Review Sentiment Analyzer")
st.write("Enter review title & description to predict if GOOD (≥4 stars) or BAD (<4).")

with st.form("sentiment_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📝 Review Title")
        review_title = st.text_area("Title", placeholder="e.g., No charger included!", height=100)
    
    with col2:
        st.subheader("💬 Review Description")
        review_desc = st.text_area("Description", placeholder="e.g., Disappointed but speed is great...", height=100)
    
    # Optional (from data)
    country_india = st.checkbox("Country: India", value=True)
    verified = st.checkbox("Verified Purchase", value=True)
    
    submitted = st.form_submit_button("🔮 Analyze Sentiment")

# Process & Predict
if submitted and 'model' in locals():
    full_review = review_title + " " + review_desc
    if full_review.strip():
        # Exact preprocess
        sample = tokenizer.texts_to_matrix([full_review], mode='binary')
        
        pred = model.predict(sample, verbose=0)[0][0]
        prob = float(pred)
        
        st.divider()
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            if prob > 0.5:
                st.success("### GOOD ✅")
                st.metric("Confidence", f"{prob:.1%}", delta="Positive Review")
            else:
                st.error("### BAD ❌")
                st.metric("Confidence", f"{1-prob:.1%}", delta="Negative Review")
        
        with res_col2:
            st.progress(prob)
            st.write(f"**Model Score:** {prob:.3f}")
            
            # Quick feedback
            if prob > 0.7:
                st.balloons()
            elif prob < 0.3:
                st.error("😞 Consider improvements.")
    else:
        st.warning("Enter a review!")

# Examples
with st.expander("📚 Example Reviews"):
    st.info("**Good:** 'Amazing speed, love it!'")
    st.warning("**Bad:** 'No charger, overpriced!'")

st.markdown("---")
st.caption("Built with Streamlit + TensorFlow. Deployed-ready.")

