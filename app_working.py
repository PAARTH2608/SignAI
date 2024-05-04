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


def extract_feature(video_stream):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    with mp_hands.Hands(static_image_mode=False, model_complexity=0,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands:
        while True:
            ret, image = video_stream.read()
            if not ret:
                break

            results = hands.process(cv2.flip(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 1))
            image_height, image_width, _ = image.shape

            if results.multi_hand_landmarks:
                annotated_image = cv2.flip(image.copy(), 1)
                for hand_landmarks in results.multi_hand_landmarks:
                # Wrist Hand /  Pergelangan Tangan
                    wristX = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x * image_width
                    wristY = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y * image_height
                    wristZ = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].z

                    # Thumb Finger / Ibu Jari
                    thumb_CmcX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].x * image_width
                    thumb_CmcY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].y * image_height
                    thumb_CmcZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_CMC].z

                    thumb_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].x * image_width
                    thumb_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].y * image_height
                    thumb_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_MCP].z

                    thumb_IpX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].x * image_width
                    thumb_IpY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].y * image_height
                    thumb_IpZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_IP].z

                    thumb_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * image_width
                    thumb_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * image_height
                    thumb_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].z

                    # Index Finger / Jari Telunjuk
                    index_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x * image_width
                    index_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y * image_height
                    index_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].z

                    index_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].x * image_width
                    index_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y * image_height
                    index_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].z

                    index_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].x * image_width
                    index_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y * image_height
                    index_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].z

                    index_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width
                    index_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height
                    index_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].z

                    # Middle Finger / Jari Tengah
                    middle_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * image_width
                    middle_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * image_height
                    middle_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].z

                    middle_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].x * image_width
                    middle_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y * image_height
                    middle_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].z

                    middle_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].x * image_width
                    middle_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].y * image_height
                    middle_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP].z

                    middle_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].x * image_width
                    middle_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y * image_height
                    middle_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].z

                    # Ring Finger / Jari Cincin
                    ring_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].x * image_width
                    ring_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].y * image_height
                    ring_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_MCP].z

                    ring_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].x * image_width
                    ring_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y * image_height
                    ring_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].z

                    ring_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].x * image_width
                    ring_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].y * image_height
                    ring_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_DIP].z

                    ring_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].x * image_width
                    ring_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y * image_height
                    ring_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].z

                    # Pinky Finger / Jari Kelingking
                    pinky_McpX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].x * image_width
                    pinky_McpY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].y * image_height
                    pinky_McpZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_MCP].z

                    pinky_PipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].x * image_width
                    pinky_PipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y * image_height
                    pinky_PipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].z

                    pinky_DipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].x * image_width
                    pinky_DipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].y * image_height
                    pinky_DipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_DIP].z

                    pinky_TipX = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].x * image_width
                    pinky_TipY = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y * image_height
                    pinky_TipZ = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].z

                    # Draw the Skeleton
                    mp_drawing.draw_landmarks(
                        annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # draw bounding box
                    bounding_box = {}
                    bounding_box['x_min'] = int(min(wristX, thumb_CmcX, thumb_IpX, thumb_TipX, index_McpX, index_PipX, index_DipX, index_TipX, middle_McpX,
                                                middle_PipX, middle_DipX, middle_TipX, ring_McpX, ring_PipX, ring_DipX, ring_TipX, pinky_McpX, pinky_PipX, pinky_DipX, pinky_TipX))
                    bounding_box['y_min'] = int(min(wristY, thumb_CmcY, thumb_IpY, thumb_TipY, index_McpY, index_PipY, index_DipY, index_TipY, middle_McpY,
                                                middle_PipY, middle_DipY, middle_TipY, ring_McpY, ring_PipY, ring_DipY, ring_TipY, pinky_McpY, pinky_PipY, pinky_DipY, pinky_TipY))
                    bounding_box['x_max'] = int(max(wristX, thumb_CmcX, thumb_IpX, thumb_TipX, index_McpX, index_PipX, index_DipX, index_TipX, middle_McpX,
                                                middle_PipX, middle_DipX, middle_TipX, ring_McpX, ring_PipX, ring_DipX, ring_TipX, pinky_McpX, pinky_PipX, pinky_DipX, pinky_TipX))
                    bounding_box['y_max'] = int(max(wristY, thumb_CmcY, thumb_IpY, thumb_TipY, index_McpY, index_PipY, index_DipY, index_TipY, middle_McpY,
                                                middle_PipY, middle_DipY, middle_TipY, ring_McpY, ring_PipY, ring_DipY, ring_TipY, pinky_McpY, pinky_PipY, pinky_DipY, pinky_TipY))

                    cv2.rectangle(annotated_image, (bounding_box['x_min'], bounding_box['y_min']), (
                        bounding_box['x_max'], bounding_box['y_max']), (0, 255, 0), 2)
                
                yield annotated_image
            else:
                yield None


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


def predict(frame, model):
    (wristX, wristY, wristZ,
     thumb_CmcX, thumb_CmcY, thumb_CmcZ,
     thumb_McpX, thumb_McpY, thumb_McpZ,
     thumb_IpX, thumb_IpY, thumb_IpZ,
     thumb_TipX, thumb_TipY, thumb_TipZ,
     index_McpX, index_McpY, index_McpZ,
     index_PipX, index_PipY, index_PipZ,
     index_DipX, index_DipY, index_DipZ,
     index_TipX, index_TipY, index_TipZ,
     middle_McpX, middle_McpY, middle_McpZ,
     middle_PipX, middle_PipY, middle_PipZ,
     middle_DipX, middle_DipY, middle_DipZ,
     middle_TipX, middle_TipY, middle_TipZ,
     ring_McpX, ring_McpY, ring_McpZ,
     ring_PipX, ring_PipY, ring_PipZ,
     ring_DipX, ring_DipY, ring_DipZ,
     ring_TipX, ring_TipY, ring_TipZ,
     pinky_McpX, pinky_McpY, pinky_McpZ,
     pinky_PipX, pinky_PipY, pinky_PipZ,
     pinky_DipX, pinky_DipY, pinky_DipZ,
     pinky_TipX, pinky_TipY, pinky_TipZ,
     output_IMG) = extract_feature(frame)

    input_IMG = np.array([[[wristX], [wristY], [wristZ],
                           [thumb_CmcX], [thumb_CmcY], [thumb_CmcZ],
                           [thumb_McpX], [thumb_McpY], [thumb_McpZ],
                           [thumb_IpX], [thumb_IpY], [thumb_IpZ],
                           [thumb_TipX], [thumb_TipY], [thumb_TipZ],
                           [index_McpX], [index_McpY], [index_McpZ],
                           [index_PipX], [index_PipY], [index_PipZ],
                           [index_DipX], [index_DipY], [index_DipZ],
                           [index_TipX], [index_TipY], [index_TipZ],
                           [middle_McpX], [middle_McpY], [middle_McpZ],
                           [middle_PipX], [middle_PipY], [middle_PipZ],
                           [middle_DipX], [middle_DipY], [middle_DipZ],
                           [middle_TipX], [middle_TipY], [middle_TipZ],
                           [ring_McpX], [ring_McpY], [ring_McpZ],
                           [ring_PipX], [ring_PipY], [ring_PipZ],
                           [ring_DipX], [ring_DipY], [ring_DipZ],
                           [ring_TipX], [ring_TipY], [ring_TipZ],
                           [pinky_McpX], [pinky_McpY], [pinky_McpZ],
                           [pinky_PipX], [pinky_PipY], [pinky_PipZ],
                           [pinky_DipX], [pinky_DipY], [pinky_DipZ],
                           [pinky_TipX], [pinky_TipY], [pinky_TipZ]]])

    predictions = model.predict(input_IMG)
    char = chr(np.argmax(predictions)+65)
    confidence = np.max(predictions)/np.sum(predictions)

    if confidence > 0.4:
        cv2.putText(output_IMG, char, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(output_IMG, str(confidence), (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return char, confidence, output_IMG

    return None, None, output_IMG


class Detection(NamedTuple):
    name: str
    prob: float


class VideoTransformer(VideoProcessorBase):

    result_queue: "queue.Queue[List[Detection]]"

    def __init__(self) -> None:
        self.threshold1 = 224
        self.result_queue = queue.Queue()
        self.data = np.ndarray(shape=(1, 240, 240, 3), dtype=np.float32)

    def _predict_image(self, image):
        result: List[Detection] = []
        model = load_model()
        label, confidence, output_img = predict(image, model)
        if label is not None:
            result.append(Detection(label, float(confidence)))
        return result, output_img

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        self.frame = frame
        result, output_img = self._predict_image(
            frame.to_ndarray(format="bgr24"))
        self.result_queue.put(result)

        return av.VideoFrame.from_ndarray(output_img, format="bgr24")


def process_video():
    cap = cv2.VideoCapture(0)  # Use the webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process each frame using extract_feature
        for annotated_image in extract_feature(cap):
            if annotated_image is not None:
                # Display the annotated image in Streamlit
                st.image(annotated_image, channels="BGR", use_column_width=True)

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
