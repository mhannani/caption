from app_utils import SessionState
import streamlit as st
from PIL import Image
from app_utils.gt_captions import gt_captions
from app_utils.load_assets import local_css, remote_css, icon
from app_utils.evaluation import calculate_blue_score
import webbrowser
from app_utils.generate_caption import generate_caption

GitHub = "https://github.com/mhannani/caption"
WebApp = "https://caption.mhannani.com/"

# github_icon = Image.open("static/icons/github.png")
icon(["fab fa-github fa-2x", "fas fa-globe fa-2x"], [GitHub, WebApp])

fw_cdn = "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta2/css/all.min.css"
blue_score_dict_weights = {1: (1, 0, 0, 0), 2: (0, 1, 0, 0), 3: (0, 0, 1, 0), 4: (0, 0, 0, 1)}
# supposing the pred_button got not clicked yet
pred_button = False
show_gt_captions = False
default_blue_weight = ""
n_grams = 0
captions = []

local_css("static/app.css")
remote_css(fw_cdn)

st.sidebar.subheader("General Settings")

# Pick the model version
choose_model = st.sidebar.selectbox(
    "1. Pick model you'd like to use",
    ("CNN_INCEP_V3_LSTM_WITHOUT_ATT",
     "CNN_INCEP_V3_GRU_WITH_ATT",
     "CNN_INCEP_V3_LSTM_WITH_ATT")
)

# caption length parameter
caption_length = st.sidebar.slider('2. The caption length', 0, 50, 30)

st.title("Welcome to Caption")

# File uploader allows user to add their own image
uploaded_file = st.file_uploader(label="Let's begin by uploading an image",
                                 type=["png", "jpeg", "jpg"])

# setup the state of the app
session_state = SessionState.get(pred_button=False, show_gt_captions=False)


# The logic of the app flow

if not uploaded_file:
    st.warning("Please upload an image.")
    st.stop()

else:
    session_state.uploaded_image = uploaded_file.read()
    session_state.filename = uploaded_file.name
    st.image(session_state.uploaded_image)
    captions = gt_captions(session_state.filename)
    if not captions.empty:
        show_gt_captions = st.sidebar.radio("Show ground truth captions ?", ("Yes", "No"))
        st.sidebar.subheader("Evaluation Settings")

        # whether to used the default weighted BLEU-4
        default_blue_weight = st.sidebar.radio("1. Use the default weights of BLUE score ?", ("Yes", "No"))

        # n-grams parameters
        if default_blue_weight == "No":
            n_grams = st.sidebar.slider('2. Number of n-grams', 1, 4, 4)

    session_state.gt_captions_button = show_gt_captions

    if show_gt_captions == "Yes" and not captions.empty:
        st.subheader('~ Ground Truth captions')
        st.table(captions)


if uploaded_file:
    pred_button = st.sidebar.button("Predict")

# Did the user press the predict button?
if pred_button:
    session_state.pred_button = True
else:
    session_state.pred_button = False

if session_state.pred_button:
    st.subheader('~ Generated caption')
    caption = generate_caption(uploaded_file, caption_length)

    st.success(caption)
    if not captions.empty:
        if default_blue_weight == "Yes":
            weights = (1. / 4., 1. / 4., 1. / 4., 1. / 4.)
        else:
            weights = blue_score_dict_weights[n_grams]

        blue_score = calculate_blue_score(session_state.filename, caption, weights)
        st.subheader('~ The BLUE score')
        st.info("{:.10f}".format(blue_score))
