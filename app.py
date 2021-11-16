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

github_icon = Image.open("assets/icons/github.png")
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
    st.image(session_state.uploaded_image, use_column_width=True)
    pred_button = st.button("Predict")


if pred_button:
    session_state.pred_button = True
else:
    session_state.pred_button = False