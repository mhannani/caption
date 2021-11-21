import os
import json
import requests
import SessionState
import streamlit as st
import webbrowser
from PIL import Image

st.title("Welcome to Caption")
st.header("Identify what's in your photos!")
GitHub = "https://github.com/mhannani/caption"


# supposing the pred_button got not clicked yet
pred_button = False

st.sidebar.title(
    "Caption"
)

caption_length = st.sidebar.slider('The caption length', 0, 50, 30)

# Pick the model version
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("CNN_INCEP_V3_LSTM_WITHOUT_ATT",
     "CNN_INCEP_V3_GRU_WITH_ATT",
     "CNN_INCEP_V3_LSTM_WITH_ATT")
)

# github_icon = Image.open("assets/icons/github.png")
if st.sidebar.button('Fork on GitHub'):
    webbrowser.open_new_tab(GitHub)

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Let's begin by uploading an image of something here",
                                 type=["png", "jpeg", "jpg"])

# setup the state of the app
session_state = SessionState.get(pred_button=False)

# The logic of the app flow
if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()
else:
    session_state.uploaded_image = uploaded_file.read()
    print(type(uploaded_file.read()))
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")


if pred_button:
    session_state.pred_button = True
else:
    session_state.pred_button = False
