from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np
import re

# ============================================
# Load Dataset
# ============================================

keras.utils.set_random_seed(42)

df = pd.read_csv("medical_comments_dataset.csv")

# print(df.head())
# print(df.shape)
# print(df.info())

# ============================================
# Data Cleaning
# ============================================

df = df.dropna()
df = df.drop_duplicates()

# print(df.isnull().sum())
# print(df.shape)

# ============================================
# Text Preprocessing
# ============================================

df["message"] = df["message"].str.lower()

df["message"] = df["message"].str.strip()
df["message"] = df["message"].str.replace(r"\s+", " ", regex=True)

df["message"] = df["message"].apply(
    lambda x: re.sub(r"[^\w\s]", "", x)
)

# ============================================
# Separate Features and Labels
# ============================================

X = df["message"]
y = df["package"]

# print(X.head())
# print(y.head())

# ============================================
# Label Encoding
# ============================================

encoder = LabelEncoder()
y = encoder.fit_transform(y)
# print(encoder.classes_)

# ============================================
# Tokenization
# ============================================

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(X)
# print(tokenizer.word_index)

# ============================================
# Convert Text to Sequences
# ============================================

X = tokenizer.texts_to_sequences(X)
# print(X[:5])

# ============================================
# Padding
# ============================================

X = pad_sequences(
    X,
    padding="post"
)
# print(X.shape)

# ============================================
# Train / Test Split
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)
# print(X_train.shape)
# print(X_test.shape)


# ============================================
# Build Neural Network
# ============================================

model = keras.Sequential([

    keras.layers.Embedding(
        input_dim=len(tokenizer.word_index) + 1,
        output_dim=32
    ),

    keras.layers.GlobalAveragePooling1D(),

    keras.layers.Dense( 16, activation="relu"),

    keras.layers.Dense( len(encoder.classes_), activation="softmax")
])

model.build(input_shape=(None, X.shape[1]))

# model.summary()

# ============================================
# Compile Model
# ============================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ============================================
# Train Model
# ============================================

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    validation_data=(X_test, y_test)
)

# ============================================
# Save Model & Objects
# ============================================

import joblib

model.save("medical_package_model.keras")

joblib.dump(tokenizer, "tokenizer.pkl")

joblib.dump(encoder, "encoder.pkl")

print("Model saved successfully!")