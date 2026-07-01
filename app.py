import streamlit as st
import joblib 
import matplotlib.pyplot as plt

st.title("Leo's AI Movie Review Analyzer")
st.write("This app predicts a movie review as positive or nagative.")

@st.cache_data
def loadMLmodel():
    model = joblib.load('sentiment_model.pkl')
    return model

model = loadMLmodel()

review = st.text_area("Enter a movie review to analyze:")

if st.button("Analyze"):
    if review:
        prediction = model.predict([review])
        predictionProb = model.predict_proba([review])
        negProb = (predictionProb[0][0]) * 100
        posProb = (predictionProb[0][1]) * 100

        if prediction[0] == "positive":
            st.success("LeoAI's Analysis: Positive 😁")
        else:
            st.error("LeoAI's Analysis: Negative 😟")

        fig, ax = plt.subplots(figsize=(5, 1))
        ax.barh(["Sentiment"], [negProb], color="brown", label=(f"Negative:{negProb: .2f}%"))
        ax.barh(["Sentiment"], [posProb], left=[negProb], color="green", label=(f"Positive:{posProb: .2f}%"))
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.5), ncol=2, fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Probability (%)")
        st.pyplot(fig)