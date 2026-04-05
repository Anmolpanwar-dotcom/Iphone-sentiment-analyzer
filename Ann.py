import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv('iphone.csv')
print(df.info())

print(df.describe())

print("number value",df.select_dtypes(include=['int','float64']).columns.tolist())
print("textvalues ",df.select_dtypes(include=['object']).columns.tolist())

df['reviewDescription']=df["reviewDescription"].fillna(df['reviewDescription'].mode()[0])
df['reviewUrl']=df["reviewUrl"].fillna(df['reviewUrl'].mode()[0])

print(df.isnull().sum())

df.drop(columns=['date','variant','reviewedIn','reviewUrl','variantAsin','productAsin'],inplace=True)
print(df.columns)

df['country'] = df['country'].apply(
    lambda x: 1 if str(x).strip().lower() == 'india' else 0)
y = df['ratingScore'].apply(lambda x: 1 if x >=4 else 0)

df['isVerified'] = df['isVerified'].map({True: 1, False: 0})
df['full_review'] = df['reviewTitle'].astype(str) + " " + df['reviewDescription'].astype(str)

tokenizer = Tokenizer(num_words=2000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['full_review'])
vocab_size = tokenizer.num_words
X = tokenizer.texts_to_matrix(df['full_review'], mode='binary')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(64, activation='relu', input_shape=(vocab_size,)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['accuracy'])

print("\n🚀 Training shuru ho rahi hai...")
model.fit(X_train, y_train, 
          epochs=10, 
          batch_size=32, 
          validation_data=(X_test, y_test))

loss, accuracy = model.evaluate(X_test, y_test)
print(f"\n✅ Final Accuracy: {accuracy*100:.2f}%")

model.save('ann_model.keras')
with open('tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print("\n✅ Model and tokenizer saved for Streamlit app!")
