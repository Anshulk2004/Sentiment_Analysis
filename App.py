import streamlit as st
import numpy as np
import pandas as pd 
import pickle
import nltk
from nltk.stem import WordNetLemmatizer
import re
nltk.download('wordnet')


#load modules 

with open('tfidf_vectorizer.pkl','rb') as f:
    tfidf_vectorizer = pickle.load(f)
    
with open('label_encoder.pkl','rb') as f:
    lb = pickle.load(f)
    
with open('Random_Forest.pkl','rb') as f:
    rf = pickle.load(f)        

stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_text2(text):
    lemmatizer = WordNetLemmatizer()
    text = re.sub("[^a-zA-Z]" , " " ,text)
    text = text.lower()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stopwords]
    return " ".join(text)



def predict_emotion(input_text):
    final_text = clean_text2(input_text)
    input_tfidf = tfidf_vectorizer.transform([final_text])

    final_label = rf.predict(input_tfidf)[0]
    final_emotion = lb.inverse_transform([final_label])[0]
    Emotion = np.max(rf.predict(input_tfidf))

    return final_emotion , Emotion

    


st.title("Predicting Emotions") 

st.write(" Emotions from JOY , FEAR , LOVE , ANGER , SADNESS , SURPRISE")

input_text = st.text_input("Write down the text")

if st.button("predict"):
    predicted_emotion , value = predict_emotion(input_text)
    st.write("Predicted Emotion:",predicted_emotion)
    st.write("Predicted Value",value)