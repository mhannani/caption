from app_utils import SessionState
import streamlit as st
from app_utils.gt_captions import gt_captions
from app_utils.evaluation import blue_score
import webbrowser
from app_utils.generate_caption import generate_caption

GitHub = "https://github.com/mhannani/caption"

# supposing the pred_button got not clicked yet
pred_button = False

st.sidebar.title(
    "Caption"
)
# github_icon = Image.open("static/icons/github.png")
if st.sidebar.button('Fork on GitHub'):
    webbrowser.open_new_tab(GitHub)

st.title("Welcome to Caption")

# Pick the model version
choose_model = st.sidebar.selectbox(
    "Pick model you'd like to use",
    ("CNN_INCEP_V3_LSTM_WITHOUT_ATT",
     "CNN_INCEP_V3_GRU_WITH_ATT",
     "CNN_INCEP_V3_LSTM_WITH_ATT")
)

# caption length parameter
caption_length = st.sidebar.slider('The caption length', 0, 50, 30)


# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Let's begin by uploading an image",
                                 type=["png", "jpeg", "jpg"])


# setup the state of the app
session_state = SessionState.get(pred_button=False)

# The logic of the app flow

if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()

else:
    session_state.uploaded_image = uploaded_file.read()
    session_state.filename = uploaded_file.name
    st.image(session_state.uploaded_image)
    st.subheader('~Ground Truth captions')
    st.table(gt_captions(session_state.filename))
    pred_button = st.sidebar.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True
else:
    session_state.pred_button = False

if session_state.pred_button:
    st.subheader('~Generated caption')
    caption = generate_caption(uploaded_file, caption_length)
    blue_score(session_state.filename, None)
    st.success(caption)
