# Model weights (Google Drive):
# https://drive.google.com/file/d/1c5hy7IHJ2E0C15fUdar_vN7IHKqc3_NE/view?usp=drive_link

import os
import streamlit as st
import torch
import numpy as np
from PIL import Image
from dl_rough_final_gan import ECCVGenerator, preprocess_img, postprocess_tens
import torch.utils.model_zoo as model_zoo
import gdown  # <--- NEW

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
PRETRAINED_URL = "https://colorizers.s3.us-east-2.amazonaws.com/colorization_release_v2-9b330a0b.pth"

# ---- Google Drive settings for your GAN model ----
MODEL_ID = "1c5hy7IHJ2E0C15fUdar_vN7IHKqc3_NE"
MODEL_PATH = "gan_generator_epoch_50.pth"


def download_model_if_needed():
    """Download GAN weights from Google Drive if not present."""
    if os.path.exists(MODEL_PATH):
        return  # already there

    # Direct download URL for Google Drive
    url = f"https://drive.google.com/uc?id={MODEL_ID}"
    gdown.download(url, MODEL_PATH, quiet=False)


@st.cache_resource
def load_models():
    # ------- pretrained model from URL -------
    pretrained_gen = ECCVGenerator().to(DEVICE)
    state_pre = model_zoo.load_url(
        PRETRAINED_URL,
        map_location=DEVICE,
        check_hash=True
    )
    pretrained_gen.load_state_dict(state_pre)
    pretrained_gen.eval()

    # ------- GAN generator (download if needed) -------
    download_model_if_needed()
    gan_gen = ECCVGenerator().to(DEVICE)
    state_gan = torch.load(MODEL_PATH, map_location=DEVICE)
    gan_gen.load_state_dict(state_gan)
    gan_gen.eval()

    return pretrained_gen, gan_gen


pretrained_generator, gan_generator = load_models()

st.set_page_config(layout="wide")
st.title("Image Colorizer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load original image
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    tens_orig_l, tens_rs_l = preprocess_img(img_np, HW=(256, 256), return_ab=False)
    tens_rs_l = tens_rs_l.unsqueeze(0).to(DEVICE)  # (1,1,256,256)

    # inference
    with torch.no_grad():
        # pretrained model prediction
        pred_ab_pre = pretrained_generator(tens_rs_l)[0].cpu()

        # GAN fine-tuned model prediction
        pred_ab_gan = gan_generator(tens_rs_l)[0].cpu()

    # postprocess
    colorized_pre = postprocess_tens(tens_orig_l, pred_ab_pre)
    colorized_pre_img = (colorized_pre * 255).astype(np.uint8)

    colorized_gan = postprocess_tens(tens_orig_l, pred_ab_gan)
    colorized_gan_img = (colorized_gan * 255).astype(np.uint8)

    # display side by side
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Ground Truth")
        st.image(img_np)

    with col2:
        st.subheader("Pretrained")
        st.image(colorized_pre_img)

    with col3:
        st.subheader("Fine-tuned GAN")
        st.image(colorized_gan_img)
