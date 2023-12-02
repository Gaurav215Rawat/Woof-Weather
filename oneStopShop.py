import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.layers import Dropout, Flatten, BatchNormalization, Dense, Activation
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from efficientnet.keras import EfficientNetB0
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import librosa
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the pre-trained models
image_model = load_model('../savedModels/finalImageModel.h5')
audio_model = load_model('../saved_models/finalAudio.h5')

# Function to preprocess an image
def preprocess_image(img_path_or_file):
    # Load the image
    if isinstance(img_path_or_file, str):
        # If img_path_or_file is a file path
        img = image.load_img(img_path_or_file, target_size=(224, 224))
    else:
        # If img_path_or_file is an UploadedFile
        img = Image.open(img_path_or_file).convert('RGB')

    # Convert the image to a numpy array
    img_array = image.img_to_array(img)

    # Resize the image to the model's input size
    img_array = tf.image.resize(img_array, (224, 224))

    # Normalize the pixel values to the range [0, 1]
    img_array /= 255.0

    # Expand the dimensions to match the model input shape
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

# Function to preprocess an audio file
def preprocess_audio(audio_path):
    # Load the audio file
    audio, sample_rate = librosa.load(audio_path, res_type='kaiser_fast')

    # Extract MFCC features
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    # Pad sequences to a fixed length (adjust as needed)
    mfccs_padded = pad_sequences([mfccs_scaled_features], maxlen=40, padding='post', truncating='post')

    return mfccs_padded

# Define the Streamlit interface
def main():
    st.title("Mixed Sentiment Analysis App")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Mixed Sentiment Analysis App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Option to select image or audio prediction
    option = st.radio("Select Prediction Type", ('Image Prediction', 'Audio Prediction'))

    if option == 'Image Prediction':
        # Input for image
        image_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        processed_image = None  # Initialize processed_image

        if image_file is not None and image_file.type in ["image/jpeg", "image/jpg", "image/png"]:
            processed_image = preprocess_image(image_file)
        else:
            st.warning("Please upload a valid image file (jpg, jpeg, png).")

        # Display the selected image
        if image_file is not None:
            st.image(image_file, caption="Selected Image", use_column_width=True)

        result = ""

        # Perform prediction when the user clicks the "Predict" button
        if st.button("Predict"):
            if processed_image is None:
                st.warning("Please upload an image.")
            else:
                # Make predictions using the image model
                predictions = image_model.predict(processed_image)

                # Get the predicted class label
                class_indices = {0: 'angry', 1: 'happy', 2: 'sad'}  # Update with your actual class labels
                predicted_class_index = np.argmax(predictions)
                predicted_class_label = class_indices[predicted_class_index]

                result = f"The sentiment for the image is {predicted_class_label}"

        # Display the result
        if processed_image is not None:
            st.success(result)
        else:
            st.warning("Image processing failed.")

    elif option == 'Audio Prediction':
        # Input for audio file
        audio_file = st.file_uploader("Choose an audio file...", type=["wav"])

        processed_audio = None  # Initialize processed_audio

        if audio_file is not None and audio_file.type == "audio/wav":
            processed_audio = preprocess_audio(audio_file)
        else:
            st.warning("Please upload a valid audio file (wav).")

        result = ""

        # Perform prediction when the user clicks the "Predict" button
        if st.button("Predict"):
            if processed_audio is None:
                st.warning("Please upload an audio file.")
            else:
                # Make predictions using the audio model
                predictions = audio_model.predict(processed_audio)

                # Get the predicted class label
                class_indices = {0: 'ANGER', 1: 'HAPPY', 2: 'SAD'}  # Update with your actual class labels
                predicted_class_index = np.argmax(predictions)
                predicted_class_label = class_indices[predicted_class_index]

                result = f"The predicted emotion is {predicted_class_label}"

        # Display the result
        if processed_audio is not None:
            st.success(result)
        else:
            st.warning("Audio processing failed.")

    if st.button("About"):
        st.text("Let's Learn")
        st.text("Built with Streamlit")

if __name__ == '__main__':
    main()
