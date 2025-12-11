import streamlit as st

st.set_page_config(page_title="Fraud Detection Demo", page_icon="💳", layout="centered")

st.title("💳 Credit Card Fraud Detection (Demo Version)")

st.write("""
This is a **demo version** of the Credit Card Fraud Detection app.

The original version requires a trained machine learning model and a large dataset,
which cannot be loaded on Streamlit Cloud without additional setup.
""")

st.subheader("🔮 Demo Prediction")

amount = st.slider("Transaction Amount ($)", 1, 2000, 120)
time = st.slider("Time of Transaction (seconds)", 0, 100000, 5000)
v1 = st.slider("V1 Feature", -10.0, 10.0, 0.5)

if st.button("Predict"):
    # Simple Fake Logic (Demo Only)
    if amount > 1000 or v1 < -3:
        st.error("⚠️ Prediction: FRAUD (Demo Logic)")
    else:
        st.success("✅ Prediction: SAFE (Demo Logic)")

st.markdown("---")
st.caption("Demo version created by Khushi Katwe")
