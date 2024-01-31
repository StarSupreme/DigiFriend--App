import os
import random
from flask import Flask, render_template, request, redirect, url_for, session
from chatterbot import ChatBot
from gtts import gTTS
from chatterbot.trainers import ListTrainer
from kivy.core.audio import SoundLoader
import numpy as np
import librosa
import sounddevice as sd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from PIL import Image
import requests
import cv2
from fer import FER
from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
from gtts import gTTS
import os
import speech_recognition as sr
import threading
import time

app = Flask(__name__)
app.secret_key = 'MANBEARPIG_MUDMAN888'

# Initialize the chatbot once, not in every request
bot = ChatBot('MyBot')

def perform_real_time_prediction():
    global final_emotion                                                                    
    # Function to extract features using Librosa
    def extract_features(audio_data, sample_rate):
        # Extract MFCC features from audio data
        mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T, axis=0)
        return mfccs

    # Function to generate synthetic audio data for different emotions
    def generate_audio_data(emotion, duration, sample_rate):
        t = np.linspace(0, duration, int(duration * sample_rate), endpoint=False)

        if emotion == 'happy':
            # Generate synthetic happy audio (frequency-modulated sine wave)
            frequency = np.interp(np.random.random(), [0, 1], [220, 630])
            audio_data = np.sin(2 * np.pi * frequency * t + 5 * np.sin(2 * np.pi * 0.25 * t))
        elif emotion == 'sad':
            # Generate synthetic sad audio (sine wave with varying amplitude)
            frequency = np.interp(np.random.random(), [0, 1], [100, 700])
            audio_data = np.sin(2 * np.pi * 220 * t) * np.interp(t, [0, duration], [1, 0])
        elif emotion == 'angry':
            # Generate synthetic angry audio (frequency-modulated sawtooth wave)
            frequency = np.interp(np.random.random(), [0, 1], [300, 700])
            audio_data = librosa.core.tone(frequency, sr=sample_rate, duration=duration) + 0.3 * np.sin(
                2 * np.pi * 0.5 * t)
        else:
            # Generate synthetic neutral audio (white noise with varying amplitude)
            audio_data = np.interp(np.random.rand(int(duration * sample_rate)), [0, 1], [-1, 1])

        return audio_data

    # Generate synthetic data for different emotions
    sample_rate = 22050  # Sampling rate
    duration = 3  # Duration for each synthetic audio (seconds)

    emotions = ['happy', 'sad', 'angry', 'neutral']
    X = []
    y = []

    for emotion in emotions:
        for _ in range(50):  # Generate 50 samples for each emotion
            audio_data = generate_audio_data(emotion, duration, sample_rate)
            features = extract_features(audio_data, sample_rate)
            X.append(features)
            y.append(emotion)

    # Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train RandomForestClassifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_classifier.fit(X_train, y_train)

    # Predict on test set
    y_pred = rf_classifier.predict(X_test)

    # Evaluate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    # Function for real-time emotion prediction
    def predict_emotion(audio_data, sample_rate):
        features = extract_features(audio_data, sample_rate)
        # Reshape features to match the shape the model expects
        features = features.reshape(1, -1)
        # Perform emotion prediction using your trained model
        predicted_emotion = rf_classifier.predict(features)
        return predicted_emotion[0]

    # Define parameters for real-time audio capturing
    duration = 5  # Duration in seconds for capturing audio
    sample_rate = 22050  # Sampling rate
    channels = 1  # Mono audio input

    # Predict emotion from voice using the trained classifier
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, blocking=True)
    dominant_emotion_voice = predict_emotion(audio_data[:, 0], sample_rate)
    emotions_voice = dominant_emotion_voice
    # Initialize the FER detector
    detector = FER(mtcnn=True)

    # Initialize variables to store emotion intensities
    emotion_intensities = {
        'happy': 0,
        'sad': 0,
        'fear': 0,
        'disgust': 0,
        'angry': 0,
        'surprise': 0,
        'neutral': 0
    }
    emotion_intensitiess = {
        'happy': 0,
        'sad': 0,
        'fear': 0,
        'disgust': 0,
        'angry': 0,
        'surprise': 0,
        'neutral': 0
    }

    # Function to update emotion intensities for voice predictions
    def update_emotion_intensities_voice(emotion):
        emotion_intensities[emotion] += 1  # You can use any value or weight based on your requirement

    # Predict emotion from voice using the trained classifier
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, blocking=True)
    dominant_emotion_voice = predict_emotion(audio_data[:, 0], sample_rate)
    emotions_voice = dominant_emotion_voice

    # Update emotion intensities based on voice prediction
    update_emotion_intensities_voice(emotions_voice)

    # URLs for emoji images
    emoji_urls = {
        'happy': 'https://i.postimg.cc/nhYrr6vh/Remove-background-project-2.png',
        'sad': 'https://i.postimg.cc/k4Fm9tz1/Remove-background-project-3.png',
        'fear': 'https://i.postimg.cc/C14W9pn9/Remove-background-project-4.png',
        'disgust': 'https://i.postimg.cc/SxMk9yyL/Remove-background-project-8.png',
        'angry': 'https://i.postimg.cc/SNYgMpNq/Remove-background-project-7.png',
        'surprise': 'https://i.postimg.cc/6qyYckrr/Remove-background-project-6.png',
        'neutral': 'https://i.postimg.cc/NFJZf844/Remove-background-project-5.png',
    }

    def fetch_image_with_alpha(url):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                image = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_UNCHANGED)
                # Check if the image has an alpha channel
                if image is not None and image.shape[
                    2] == 3:  # If the image has 3 channels (no alpha), add an alpha channel
                    b, g, r = cv2.split(image)
                    alpha = np.ones_like(b) * 255
                    image = cv2.merge((b, g, r, alpha))
                return image
            else:
                print(f"Failed to fetch image from {url}. Status code: {response.status_code}")
                return None
        except Exception as e:
            print(f"Error fetching image from {url}: {e}")
            return None

    # Load emoji images from URLs with alpha channel
    emoji_images = {emotion: fetch_image_with_alpha(url) for emotion, url in emoji_urls.items()}

    def update_emotion_intensities_face(dominant_emotion_face):
        emotion_intensitiess[dominant_emotion_face] += 1

    # Initialize video capture object outside the loop
    cap = cv2.VideoCapture(0)

    # Capture frame-by-frame
    ret, frame = cap.read()

    # Resize the frame to a smaller resolution for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  # Adjust the scaling factor as needed

    # Detect emotions in the resized frame
    result = detector.detect_emotions(small_frame)
    if result:
        for face in result:
            emotions_face = face['emotions']
            dominant_emotion_face = max(emotions_face, key=emotions_face.get)

            # Update emotion intensities based on the dominant emotion from face prediction
            update_emotion_intensities_face(dominant_emotion_face)
    # Predict emotions from face and voice
    if result:
        for face in result:
            emotions_face = face['emotions']
            dominant_emotion_face = max(emotions_face, key=emotions_face.get)

            # Define a weight for each source of emotion (face and voice)
            weight_face = 0.6  # Weight for face emotion
            weight_voice = 0.4  # Weight for voice emotion

            # Normalize the emotions' scores to a common scale (e.g., 0 to 1)
            normalized_emotion_face = emotions_face[dominant_emotion_face] / sum(emotions_face.values())
            normalized_emotion_voice = emotion_intensities.get(emotions_voice, 0) / sum(emotion_intensities.values())

            # Calculate combined emotion scores considering weights
            combined_emotion_score = (normalized_emotion_face * weight_face) + (normalized_emotion_voice * weight_voice)

            # Choose the final emotion based on the combined score
            final_emotion = dominant_emotion_face if combined_emotion_score >= 0.5 else emotion_voice

            # Predict emotion from voice using the trained classifier
            emotion_voice = predict_emotion(audio_data[:, 0], sample_rate)

            # Choose dominant emotion considering both face and voice
            if dominant_emotion_face == dominant_emotion_voice:
                final_emotion = dominant_emotion_face
            elif dominant_emotion_face != dominant_emotion_voice and dominant_emotion_face == "happy" or "neutral" or "surprise" and dominant_emotion_voice == "happy" or "neutral":
                final_emotion = "happy"
            elif dominant_emotion_face != dominant_emotion_voice and dominant_emotion_face == "disgust" or "fear" or "sad" and dominant_emotion_voice == "sad" or "angry":
                final_emotion = "sad"
            elif dominant_emotion_face != dominant_emotion_voice and dominant_emotion_face == "fear" or "surprise" and dominant_emotion_voice == "sad" or "neutral":
                final_emotion = "fear"
            elif dominant_emotion_face != dominant_emotion_voice and dominant_emotion_face == "neutral" or "angry" or "sad" and dominant_emotion_voice == "sad" or "angry" or "neutral":
                final_emotion = "sad"
            break

        print("Emotion: " + final_emotion)
    return final_emotion


def get_corpus_file_path(emotion):
    base_path = "./chats/"
    global text

    if final_emotion == 'neutral':
        text = "Hi, how are you?"

    elif final_emotion == 'happy':
        text = "Someone seems happy!"

    elif final_emotion == 'sad':
        text = "Someone is sad, what is the matter?"

    elif final_emotion == 'angry':
        text = "You have a bad temper now, what's the matter?"

    elif final_emotion == 'disgust':
        text = "What is that?!"

    elif final_emotion == 'surprise':
        text = "OMG! What just happened?"

    elif final_emotion == 'fear':
        text = "Woah! What happened?!"

    return os.path.join(base_path, f"{emotion}.txt" if emotion != 'neutral' else "chat.txt")


def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save('output.mp3')

    # Load and play the audio
    sound = SoundLoader.load('output.mp3')
    if sound:
        sound.play()


@app.route('/')
def home():
    global final_emotion
    # Train the bot with a random emotion when the homepage is loaded
    perform_real_time_prediction()
    corpus_file = get_corpus_file_path(final_emotion)
    with open(corpus_file, 'r') as file:
        training_data = file.readlines()
    trainer = ListTrainer(bot)
    trainer.train(training_data)
    speak(text)

    # Speak the intro line only if the page is loaded for the first time
    if not session.get('page_loaded'):
        session['page_loaded'] = True

    return render_template('index.html', final_emotion=final_emotion)

@app.route('/listen', methods=['POST'])
def listen():
    # Handle the form data sent from the client
    user_input = request.form.get('user_input')
    if user_input:
        response = bot.get_response(user_input).text
        return response
    return "No user input received", 400

if __name__ == '__main__':
    app.run(debug = True)