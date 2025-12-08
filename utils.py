import numpy as np
import pandas as pd
import joblib

def load_csv_images(path):
    df = pd.read_csv(path)
    labels = df['label'].values
    images = df.drop('label', axis=1).values
    images = images.reshape(-1, 28, 28, 1) / 255.0
    return images, labels

def load_label_encoder(path="label_encoder.joblib"):
    return joblib.load(path)

def decode_label(encoder, value):
    return encoder.inverse_transform([value])[0]

def label_to_letter(label):
    return chr(label + 65)
