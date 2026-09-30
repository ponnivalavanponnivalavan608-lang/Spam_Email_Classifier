import streamlit as st
import joblib


# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="SpamGuard AI",
    page_icon="🛡️",
    layout="centered"
)


# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown(
    """
    <style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .subtitle {
        text-align: center;
        font-size: 17px;
        margin-bottom: 25px;
    }

    .result-box {
        padding: 20px;
        border-radius: 15px;
        margin-top: 20px;
        text-align: center;
    }

    .footer {
        text-align: center;
        margin-top: 35px;
        font-size: 13px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ==========================================
# LOAD MODEL
# ==========================================

try:
    model = joblib.load("model/spam_model.pkl")
except Exception as error:
    st.error("Unable to load the trained model.")
    st.exception(error)
    st.stop()


# ==========================================
# HEADER
# ==========================================

st.markdown(
    '<div class="main-title">🛡️ SpamGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Machine Learning Spam Message Classifier'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# INFORMATION
# ==========================================

with st.container(border=True):

    st.write("### 📩 Analyze a Message")

    st.write(
        "Paste an SMS, email text, or other message below. "
        "The trained machine learning model will classify it "
        "as Spam or Not Spam."
    )


# ==========================================
# QUICK EXAMPLES
# ==========================================

examples = {
    "Custom message": "",
    "🚨 Spam example": (
        "Congratulations! You have won $1000. "
        "Click now to claim your free prize!"
    ),
    "✅ Normal example": (
        "Hi, are you coming to college tomorrow? "
        "We have a meeting at 10 AM."
    )
}

selected_example = st.selectbox(
    "Quick test:",
    list(examples.keys())
)


# ==========================================
# MESSAGE INPUT
# ==========================================

message = st.text_area(
    "Your message",
    value=examples[selected_example],
    height=180,
    placeholder="Type or paste your message here..."
)


# ==========================================
# PREDICTION
# ==========================================

if st.button(
    "🔍 Analyze Message",
    use_container_width=True
):

    if not message.strip():

        st.warning("⚠️ Please enter a message first.")

    else:

        # Prediction
        prediction = int(model.predict([message])[0])

        # Probabilities
        probabilities = model.predict_proba([message])[0]

        class_probabilities = dict(
            zip(model.classes_, probabilities)
        )

        not_spam_probability = class_probabilities.get(0, 0)
        spam_probability = class_probabilities.get(1, 0)


        # ==========================================
        # RESULT
        # ==========================================

        if prediction == 1:

            st.error("🚨 SPAM MESSAGE")

            st.write(
                f"Spam probability: "
                f"{spam_probability * 100:.2f}%"
            )

        else:

            st.success("✅ NOT SPAM")

            st.write(
                f"Not-spam probability: "
                f"{not_spam_probability * 100:.2f}%"
            )


        # ==========================================
        # PROBABILITY DETAILS
        # ==========================================

        st.write("### 📊 Model Probability")

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "✅ Not Spam",
                f"{not_spam_probability * 100:.2f}%"
            )

        with col2:
            st.metric(
                "🚨 Spam",
                f"{spam_probability * 100:.2f}%"
            )

        st.write("Not Spam")
        st.progress(not_spam_probability)

        st.write("Spam")
        st.progress(spam_probability)


# ==========================================
# FOOTER
# ==========================================

st.markdown(
    '<div class="footer">'
    'Built with Python • Scikit-learn • TF-IDF • Naive Bayes • Streamlit'
    '</div>',
    unsafe_allow_html=True
)