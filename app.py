import queue
from typing import List, NamedTuple

import av
import cv2
import firebase_admin
import mediapipe as mp
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import whisper
from audiorecorder import audiorecorder
from firebase_admin import credentials, firestore
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
import queue
import os
import glob
from textblob import TextBlob
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.express as px
import whisper
from io import BytesIO

# np.set_printoptions(suppress=True)


def firebase():
    if not firebase_admin._apps:
        cred = credentials.Certificate(
            'static/signai-ea450-firebase-adminsdk-pj29a-1469176a2d.json')
        app = firebase_admin.initialize_app(cred)
    else:
        app = firebase_admin.get_app()
    db = firestore.client()
    return app, db


def cache_query_param():
    try:
        query_param = st.experimental_get_query_params()
        user_id = query_param['user'][0]
        st.session_state['key'] = user_id
    except Exception as e:
        st.error("Please enter the user id, or try logging in from the home page")
        user_id = st.text_input("Enter your user id", key="user_id")
        st.session_state['key'] = user_id
        if user_id:
            st.experimental_set_query_params(user=user_id)
            st.experimental_rerun()

@st.cache(ttl=24*60*60, allow_output_mutation=True)
def load_model():
    num_classes = 26
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=32, kernel_size=5, strides=1,
                               padding="causal", activation="relu", input_shape=(63, 1)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=64, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=128, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(filters=256, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=256, kernel_size=5,
                               strides=1, padding="causal", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Dropout(rate=0.2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(num_classes, activation='softmax')])

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    model.load_weights('model/model_SIBI.h5')
    return model


def extract_feature(annotated_image):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(static_image_mode=False, model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        results = hands.process(cv2.flip(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 1))
        image_height, image_width, _ = annotated_image.shape

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract hand landmarks
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.append(landmark.x * image_width)
                    landmarks.append(landmark.y * image_height)
                    landmarks.append(landmark.z)

                yield landmarks
        else:
            yield None


def predict(annotated_image, model):
    features = next(extract_feature(annotated_image))
    if features is None:
        return None, None, annotated_image

    input_IMG = np.array([features])

    predictions = model.predict(input_IMG)
    char = chr(np.argmax(predictions) + 65)
    confidence = np.max(predictions) / np.sum(predictions)

    if confidence > 0.4:
        cv2.putText(annotated_image, char, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(annotated_image, str(confidence), (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return char, confidence, annotated_image

    return None, None, annotated_image


def process_video():
    cap = cv2.VideoCapture(0)
    image_placeholder = st.empty()  # Create an empty placeholder for the image
    model = load_model()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_image = frame.copy()
        # Process each frame directly in the predict function
        _, _, output_IMG = predict(annotated_image, model)
        annotated_image_bytes = cv2.imencode(".jpg", output_IMG)[1].tobytes()
        # Update the image placeholder with the new image
        image_placeholder.image(annotated_image_bytes, channels="BGR", use_column_width=True, caption="Video Feed")

    cap.release()



def sign_detection(db, user_id):
    st.image("static/sign.jpg")
    process_video()


@st.cache(ttl=24*60*60, allow_output_mutation=True)
def audio_model():
    model = whisper.load_model("tiny.en")
    return model


def speech_detection():
    # Record audio
    audio = audiorecorder(start_prompt="Start recording", stop_prompt="Stop recording", pause_prompt="", show_visualizer=True, key=None)

    if len(audio) > 0:
        # Convert AudioSegment object to WAV format
        wav_data = BytesIO()
        audio.export(wav_data, format="wav")
        wav_data.seek(0)  # Go back to the beginning of the BytesIO object

        # Display audio using Streamlit
        st.audio(wav_data, format="audio/wav", sample_rate=audio.frame_rate)

        # Save the audio to a file
        with open("audio_proc/audio.wav", "wb") as f:
            f.write(wav_data.read())

        # Perform model inference
        model = whisper.load_model("base")
        result = model.transcribe("audio_proc/audio.wav")

        # Display success message
        placeholder = st.empty()
        placeholder.success("Model loaded successfully")

        # Process the transcribed text
        text = result["text"]
        print("TEXT : ", text)

        # Create video based on the transcribed text
        if text:
            original_text = text
            first_word = original_text.split()[0].upper()
            filename = f'video_proc/{first_word}.webm'

            # Check if the image file for the first letter exists
            first_letter_image_path = f"static/sign_alpha/{first_word[0]}.jpg"
            if not os.path.exists(first_letter_image_path):
                st.error(f"Image file for the first letter '{first_word[0]}' not found.")
                return  # Exit the function

            frame = cv2.imread(first_letter_image_path)
            if frame is None:
                st.error("Failed to load image file.")
                return  # Exit the function

            height, width, _ = frame.shape
            fourcc = cv2.VideoWriter_fourcc(*'VP90')
            video = cv2.VideoWriter(filename, fourcc, 1, (width, height))

            for i in text:
                if i.isalpha():
                    image_path = f"static/sign_alpha/{i}.jpg"
                    if os.path.exists(image_path):
                        video.write(cv2.imread(image_path))
                    else:
                        st.warning(f"Image file for letter '{i}' not found.")

            cv2.destroyAllWindows()
            video.release()
            st.header(original_text)
            st.video(filename)

    # Remove temporary files
    files = glob.glob('video_proc/*')
    for f in files:
        os.remove(f)

    files = glob.glob('audio_proc/*')
    for f in files:
        os.remove(f)

    return 0


def show_database(db, user_name):
    users_ref = db.collection(u'users')
    query = users_ref.where(u'name', u'==', user_name).get()

    for doc in query:
        doc_ref = db.collection(u'users').document(doc.id)
        doc = doc_ref.get()
        user_det = doc.to_dict()
        st.header("User Details")
        if user_det is not None and all(key in user_det for key in ['name', 'dob', 'email']):
            user_df = pd.DataFrame({"Type": ['Name', 'DOB', 'Email'], "Value": [
                                   user_det['name'], user_det['dob'], user_det['email']]})
            st.dataframe(user_df)

            st.header('Sign Detected')
            sign_df = pd.DataFrame(columns=['Alphabet', 'Confidence'])
            sign_ref = doc_ref.collection(u'sign-detected').stream()
            for sign in sign_ref:
                res = sign.to_dict()
                for alphabet, prob in res.items():
                    sign_df = pd.concat([sign_df, pd.DataFrame(
                        {"Alphabet": [alphabet], "Confidence": [prob]})], ignore_index=True)

            st.dataframe(sign_df)


def main():
    image = Image.open("static/vid_call.jpg")
    logo = Image.open("static/logo.png")
    st.set_page_config(page_title="SpeechSign", page_icon=logo)

    st.image(image)
    st.title("@SpeechSign")
    cache_query_param()

    if st.session_state.key:
        user_id = st.session_state.key

        app, db = firebase()
        st.sidebar.title("Select the process to your convinience")
        st.sidebar.markdown("Select the conversion method accordingly:")
        algo = st.sidebar.selectbox(
            "Select the Operation", options=["Sign-to-Speech", "Speech-to-Sign", "Access Database", "Sign Recog Model Architecture", "Feedback", "Feedback analysis"]
        )

        if algo == "Sign-to-Speech":
            sign_detection(db, user_id=user_id)
        elif algo == "Speech-to-Sign":
            speech_detection()
        elif algo == "Access Database":
            show_database(db, user_name=user_id)
            # Add a refresh button to update the page
            if st.button("Refresh"):
                st.experimental_rerun()
        elif algo == "Sign Recog Model Architecture":
            st.title("Sign Recog Model Architecture")
            st.image("static/arch.png")

        elif algo == "Feedback":
            st.title("Feedback")
            st.write("Please provide your feedback below")
            feedback = st.text_area("Feedback")
            if st.button("Submit"):
                doc_ref = db.collection(u'feedback').document(user_id)
                doc_ref.set({u'feedback': feedback})
                st.success("Feedback submitted successfully")
                st.experimental_rerun()

        elif algo == "Feedback analysis":
            # analysis of feedback that is coming from the database, includes the feedback of all the users
            st.title("Feedback Analysis")
            st.set_option('deprecation.showPyplotGlobalUse', False)

            # get the feedback from the database
            feedback_ref = db.collection(u'feedback').stream()
            feedback = []
            for feed in feedback_ref:
                feedback.append(feed.to_dict()['feedback'])

            # convert the feedback into a dataframe
            feedback_df = pd.DataFrame(feedback, columns=['feedback'])

            # perform the sentiment analysis
            sentiment = []

            for i in range(len(feedback_df)):
                sentiment.append(
                    TextBlob(feedback_df['feedback'][i]).sentiment.polarity)

            feedback_df['sentiment'] = sentiment

            # plot the sentiment analysis
            fig = px.histogram(feedback_df, x="sentiment",
                               nbins=20, title="Sentiment Analysis")
            st.plotly_chart(fig)

            # plot the word cloud
            wordcloud = WordCloud(width=800, height=500, random_state=21,
                                  max_font_size=110).generate(' '.join(feedback))
            plt.figure(figsize=(10, 7))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            st.pyplot()

            # plot the word frequency
            word_freq = pd.Series(
                ' '.join(feedback).split()).value_counts()[:20]
            fig = px.bar(word_freq, x=word_freq.index,
                         y=word_freq.values, title="Word Frequency")
            st.plotly_chart(fig)

    else:
        st.write("Please login to continue")


if __name__ == "__main__":
    main()
