import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Student Performance Prediction",
    page_icon="🎓",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------- CSS ----------------
st.markdown("""
<style>

.stApp{
    background:linear-gradient(135deg,#0F2027,#203A43,#2C5364);
}

.main-title{
    text-align:center;
    color:white;
    font-size:48px;
    font-weight:bold;
}

.sub-title{
    text-align:center;
    color:#DDDDDD;
    font-size:20px;
}

.card{
    background:white;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 5px 20px rgba(0,0,0,0.2);
}

.pass{
    background:#d4edda;
    padding:30px;
    border-radius:15px;
    text-align:center;
    font-size:35px;
    font-weight:bold;
    color:green;
}

.fail{
    background:#f8d7da;
    padding:30px;
    border-radius:15px;
    text-align:center;
    font-size:35px;
    font-weight:bold;
    color:red;
}

footer{
    visibility:hidden;
}

</style>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------
st.sidebar.title("🎓 Student Dashboard")
st.sidebar.info("""
### AI Mini Project

Model:
- Logistic Regression

Technology:
- Python
- Scikit-Learn
- Streamlit
""")

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>🎓 Student Performance Prediction</div>", unsafe_allow_html=True)

st.markdown("<div class='sub-title'>Artificial Intelligence </div>", unsafe_allow_html=True)

st.write("")

# ---------------- INPUT ----------------
col1, col2 = st.columns(2)

with col1:
    study = st.slider(
        "📚 Study Time (Hours)",
        0.0,
        12.0,
        5.0,
        0.5
    )

with col2:
    marks = st.slider(
        "📝 Previous Grade",
        0,
        100,
        70
    )

st.write("")

if st.button("🚀 Predict Student Result", use_container_width=True):

    data = np.array([[study, marks]])

    scaled = scaler.transform(data)

    prediction = model.predict(scaled)[0]

    confidence = model.predict_proba(scaled)[0][prediction] * 100

    st.write("")
    st.subheader("📊 Prediction Dashboard")

    c1, c2, c3 = st.columns(3)

    c1.metric("📚 Study Hours", study)
    c2.metric("📝 Previous Grade", marks)
    c3.metric("🎯 Confidence", f"{confidence:.1f}%")

    st.progress(int(confidence))

    st.write("")

    if prediction == 1:

        st.markdown(
            "<div class='pass'>✅ PASS</div>",
            unsafe_allow_html=True
        )

        st.balloons()

    else:

        st.markdown(
            "<div class='fail'>❌ FAIL</div>",
            unsafe_allow_html=True
        )

st.write("")
st.write("")
st.markdown("---")

st.markdown(
"""
<center>

### 👨‍💻 Developed By

**Santosh Singh**

Artificial Intelligence Mini Project

</center>
""",
unsafe_allow_html=True
)
