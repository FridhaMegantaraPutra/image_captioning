import numpy as np
import streamlit as st
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.sequence import pad_sequences
from keras.applications.densenet import DenseNet201, preprocess_input

# Load the trained model
caption_model = tf.keras.models.load_model("model.keras")

# Load the features dictionary
features = np.load("features.npy", allow_pickle=True).item()

# Load the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.word_index = np.load("tokenizer_word_index.npy", allow_pickle=True).item()

# Load the DenseNet201 model for feature extraction
feature_extraction_model = tf.keras.models.load_model('densenet201.h5')

def extract_features(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = feature_extraction_model.predict(img)
    return features

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(image_path):
    feature = extract_features(image_path)
    
    in_text = "startseq"
    for i in range(32):  # max_length should match your training max_length
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=32)  # Adjust maxlen as needed

        y_pred = caption_model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)
        
        word = idx_to_word(y_pred, tokenizer)
        
        if word is None:
            break
            
        in_text += " " + word
        
        if word == 'endseq':
            break
            
    return in_text.replace("endseq", "by : megantara").strip()  # Clean the output

# Streamlit application
st.title("Image Captioning with Deep Learning")
st.write("Upload an image to get a predicted caption.")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded image temporarily
    image_path = "./temp_image.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write("")
    
    if st.button("Predict Caption"):
        predicted_caption = predict_caption(image_path)
        st.write("Predicted Caption:", predicted_caption)
