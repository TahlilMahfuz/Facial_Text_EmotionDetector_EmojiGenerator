import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import model_from_json

# Load and preprocess data
csv_path_train = 'data_train.csv'
csv_path_test = 'data_test.csv'
data_train = pd.read_csv(csv_path_train, encoding='utf-8')
data_test = pd.read_csv(csv_path_test, encoding='utf-8')
data = pd.concat([data_train, data_test], ignore_index=True)

# Load text emotion detection model
predictor = tf.keras.models.load_model('Text Emotion Detection (BiLSTM).h5', compile=True)
text_class_names = ['happy', 'fear', 'anger', 'sad', 'neutral']

# Load facial emotion detection model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
facial_model = model_from_json(model_json)
facial_model.load_weights("facialemotionmodel.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facial_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

# Emojis dictionary
emojis = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊',
    'neutral': '😐', 'sad': '😢', 'surprise': '😲', 
    'joy': '😊', 'anger': '😠', 'sadness': '😢'
}

# Text preprocessing functions
def clean_text(data):
    data = re.sub(r"(#[\d\w\.]+)", '', data)
    data = re.sub(r"(@[\d\w\.]+)", '', data)
    data = word_tokenize(data)
    return data

def preprocess_text(text, max_len=500):
    texts = [' '.join(clean_text(text)) for text in data['Text']]
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    seq = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(seq, maxlen=max_len)
    return padded_sequence

# Define GUI functions and elements
text_emotion = ""
facial_emotion = ""

def predict_text_emotion():
    global text_emotion
    text = text_input.get()
    processed_text = preprocess_text(text)
    prediction = predictor.predict(processed_text)
    predicted_emotion_idx = np.argmax(prediction, axis=1)[0]
    text_emotion = text_class_names[predicted_emotion_idx]
    text_emotion_label.config(text=f"Emotion: {text_emotion}")

def compare_emotions():
    if text_emotion and facial_emotion:
        match_status = "Matched" if text_emotion == facial_emotion else "Not Matched"
        emoji_to_show = emojis.get(text_emotion, '😕') if match_status == "Matched" else '😕'
        match_status_label.config(text=f"Match Status: {match_status}")
        emoji_label.config(text=emoji_to_show)
    root.after(500, compare_emotions)

def update_image():
    global facial_emotion
    _, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        img_pixels = np.expand_dims(roi_gray, axis=0)
        img_pixels = np.expand_dims(img_pixels, axis=3)
        img_pixels = img_pixels / 255.0
        predictions = facial_model.predict(img_pixels)
        max_index = int(np.argmax(predictions))
        facial_emotion = facial_labels[max_index]
        cv2.putText(frame, facial_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.config(image=imgtk)
    video_label.after(10, update_image)

def start_webcam():
    global cap
    cap = cv2.VideoCapture(0)
    update_image()

# Initialize main window
root = tk.Tk()
root.title("Emotion Detection")

# Text input frame
text_frame = ttk.Frame(root, padding="10")
text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

text_input = ttk.Entry(text_frame, width=40)
text_input.grid(row=0, column=0, sticky=(tk.W, tk.E))
predict_button = ttk.Button(text_frame, text="Predict Emotion", command=predict_text_emotion)
predict_button.grid(row=1, column=0, sticky=tk.W)
text_emotion_label = ttk.Label(text_frame, text="Emotion: None")
text_emotion_label.grid(row=2, column=0, sticky=tk.W)

# Video frame
video_frame = ttk.Frame(root, padding="10")
video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
video_label = ttk.Label(video_frame)
video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Start webcam thread
start_webcam_thread = threading.Thread(target=start_webcam)
start_webcam_thread.daemon = True
start_webcam_thread.start()

# Status frame
status_frame = ttk.Frame(root, padding="10")
status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
match_status_label = ttk.Label(status_frame, text="Match Status: None")
match_status_label.grid(row=0, column=0, sticky=tk.W)

emoji_label = ttk.Label(status_frame, text="", font=("Helvetica", 36))
emoji_label.grid(row=0, column=1, sticky=tk.W)

# Start compare emotions thread
compare_emotions_thread = threading.Thread(target=compare_emotions)
compare_emotions_thread.daemon = True
compare_emotions_thread.start()

# Run the application
root.mainloop()
