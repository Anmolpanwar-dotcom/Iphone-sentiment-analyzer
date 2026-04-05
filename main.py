from fastapi import FastAPI
from pydantic import BaseModel
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = FastAPI()

# Model aur Tokenizer load karo
model = tf.keras.models.load_model('iphone_model.h5')
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

class Review(BaseModel):
    title: str
    description: str

@app.post("/predict")
def predict(review: Review):
    full_text = review.title + " " + review.description
    seq = tokenizer.texts_to_sequences([full_text])
    padded = pad_sequences(seq, maxlen=100, padding='post')
    
    prediction = model.predict(padded, verbose=0)[0][0]
    sentiment = "GOOD" if prediction > 0.5 else "BAD"
    
    return {"sentiment": sentiment, "confidence": float(prediction)}