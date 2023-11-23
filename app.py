import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
import threading
import numpy as np
from keras.models import load_model
from keras.models import model_from_json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

text_emotion = ""
facial_emotion = ""

json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
facial_model = model_from_json(model_json)
facial_model.load_weights("facialemotionmodel.h5")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
facial_labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

text_model = load_model('Text Emotion Detection (BiLSTM).h5')
text_class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
emojis = {
    'angry': '😠', 'disgust': '🤢', 'fear': '😨', 'happy': '😊',
    'neutral': '😐', 'sad': '😢', 'surprise': '😲', 
    'joy': '😊', 'anger': '😠', 'sadness': '😢'
}

def preprocess_text(text, max_len=500):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts([text])
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)
    return padded_sequence

def predict_text_emotion():
    global text_emotion
    text = text_input.get()
    processed_text = preprocess_text(text)
    prediction = text_model.predict(processed_text)
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

root = tk.Tk()
root.title("Emotion Detection")

text_frame = ttk.Frame(root, padding="10")
text_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

text_input = ttk.Entry(text_frame, width=40)
text_input.grid(row=0, column=0, sticky=(tk.W, tk.E))
predict_button = ttk.Button(text_frame, text="Predict Emotion", command=predict_text_emotion)
predict_button.grid(row=1, column=0, sticky=tk.W)
text_emotion_label = ttk.Label(text_frame, text="Emotion: None")
text_emotion_label.grid(row=2, column=0, sticky=tk.W)

video_frame = ttk.Frame(root, padding="10")
video_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
video_label = ttk.Label(video_frame)
video_label.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

start_webcam_thread = threading.Thread(target=start_webcam)
start_webcam_thread.daemon = True
start_webcam_thread.start()

status_frame = ttk.Frame(root, padding="10")
status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
match_status_label = ttk.Label(status_frame, text="Match Status: None")
match_status_label.grid(row=0, column=0, sticky=tk.W)

emoji_label = ttk.Label(status_frame, text="", font=("Helvetica", 36))
emoji_label.grid(row=0, column=1, sticky=tk.W)


compare_emotions_thread = threading.Thread(target=compare_emotions)
compare_emotions_thread.daemon = True
compare_emotions_thread.start()


root.mainloop()
