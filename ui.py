import cv2
import tensorflow as tf
import numpy as np
import customtkinter as ctk
import mediapipe as mp
from PIL import Image, ImageTk
import os
import csv
from datetime import datetime
import time

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

model = tf.keras.models.load_model("isl_model.h5")
class_names = list("abcdefghijklmnopqrstuvwxyz")

LOG_DIR = "logs"
IMG_DIR = os.path.join(LOG_DIR, "images")
CSV_FILE = os.path.join(LOG_DIR, "predictions.csv")

os.makedirs(IMG_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction", "confidence"])

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

app = ctk.CTk()
app.title("ISL Recognition System")
app.geometry("750x700")
app.resizable(False, False)

cap = cv2.VideoCapture(0)

video_label = ctk.CTkLabel(app, text="")
video_label.pack(pady=20)

text_box = ctk.CTkTextbox(
    app,
    width=500,
    height=70,
    corner_radius=15,
    font=("Segoe UI", 20)
)
text_box.pack(pady=10)

current_frame = None
last_prediction_time = 0
last_prediction = ""

def update_frame():
    global current_frame

    ret, frame = cap.read()

    if ret:
        frame = cv2.flip(frame, 1)
        current_frame = frame.copy()

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    auto_predict()
    video_label.after(10, update_frame)

def auto_predict():
    global last_prediction_time

    current_time = time.time()

    if current_time - last_prediction_time > 2:
        predict()
        last_prediction_time = current_time

def predict():
    global current_frame, last_prediction

    if current_frame is None:
        return

    frame = current_frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        h, w, _ = frame.shape

        x_list = []
        y_list = []

        for hand_landmarks in result.multi_hand_landmarks:
            for lm in hand_landmarks.landmark:
                x_list.append(int(lm.x * w))
                y_list.append(int(lm.y * h))

        x1, x2 = min(x_list), max(x_list)
        y1, y2 = min(y_list), max(y_list)

        padding = 30

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return

        roi_original = roi.copy()

        roi = cv2.GaussianBlur(roi, (5,5), 0)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)

        roi = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        roi = cv2.resize(roi, (224,224))

        roi = tf.keras.applications.mobilenet_v2.preprocess_input(roi)
        roi = np.expand_dims(roi, axis=0)

        prediction = model.predict(roi, verbose=0)

        predicted_class = class_names[np.argmax(prediction)]
        confidence = float(np.max(prediction)) * 100

        if confidence < 70:
            return

        if predicted_class == last_prediction:
            return

        last_prediction = predicted_class

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{timestamp}.jpg"
        filepath = os.path.join(IMG_DIR, filename)

        cv2.imwrite(filepath, roi_original)

        with open(CSV_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([filename, predicted_class, f"{confidence:.2f}"])

        text_box.delete("1.0", "end")
        text_box.insert(
            "end",
            f"{predicted_class.upper()}   ({confidence:.2f}%)"
        )

    else:
        text_box.delete("1.0", "end")
        text_box.insert("end", "No hand detected")

def clear_text():
    text_box.delete("1.0", "end")

def quit_app():
    cap.release()
    app.destroy()

button_frame = ctk.CTkFrame(app, fg_color="transparent")
button_frame.pack(pady=20)

clear_btn = ctk.CTkButton(
    button_frame,
    text="Clear",
    width=130,
    height=45,
    fg_color="#ff9900",
    command=clear_text
)
clear_btn.grid(row=0, column=0, padx=20)

quit_btn = ctk.CTkButton(
    button_frame,
    text="Quit",
    width=130,
    height=45,
    fg_color="#cc0000",
    command=quit_app
)
quit_btn.grid(row=0, column=1, padx=20)

update_frame()
app.mainloop()