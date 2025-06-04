import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import os
from torchvision import transforms

from scripts.models.R2AU_dynamic import R2AttU_Net
from torchvision import models
from scripts.preprocessing.preprocessing import preprocess_image

# --- PAGE SETUP ---
st.set_page_config(page_title="Image Segmentation & Classification App", layout="centered")
st.title("🧠 Medical Image Segmentation & Classification")
st.write("Upload a **grayscale image** to get the predicted segmentation mask and risk classification.")

# --- LOAD MODELS ---
@st.cache_resource
def load_segmentation_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model", "merged_model.pth")
    model = R2AttU_Net(
        img_ch=1,
        output_ch=1,
        t=4,
        base_filters=16,
        depth=5,
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

@st.cache_resource
def load_classification_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model", "classif_model.pth")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    class_names = ["Low Risk", "High Risk"]
    num_classes = len(class_names)
    model.fc = torch.nn.Linear(num_ftrs, num_classes)  # Update for multiple classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

segmentation_model, device = load_segmentation_model()
classification_model, _ = load_classification_model()

# --- IMAGE UPLOAD ---
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # --- SAVE UPLOADED IMAGE TO FOLDER ---
    upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_image")
    os.makedirs(upload_folder, exist_ok=True)
    img_save_path = os.path.join(upload_folder, uploaded_file.name)
    with open(img_save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # --- PREPROCESS USING FILE PATH ---
    img_tensor = preprocess_image(img_save_path)  # Pass the path
    img_tensor = img_tensor.to(device)

    # --- SEGMENTATION PREDICTION ---
    with torch.no_grad():
        seg_output = segmentation_model(img_tensor)
        seg_prediction = torch.sigmoid(seg_output)

    # --- POSTPROCESS SEGMENTATION ---
    pred_mask = (seg_prediction.cpu().numpy() > 0.5).astype("uint8")[0, 0]

    # --- RESIZE IMAGES ---
    original_image = cv2.imread(img_save_path, cv2.IMREAD_GRAYSCALE)
    original_image_resized = cv2.resize(original_image, (pred_mask.shape[1], pred_mask.shape[0]))

    # --- APPLY MASK TO ORIGINAL IMAGE ---
    masked_image = cv2.bitwise_and(original_image_resized, original_image_resized, mask=pred_mask)
    thermal_image = cv2.applyColorMap(masked_image, cv2.COLORMAP_JET)

    # --- CLASSIFICATION PREDICTION ---
    from PIL import Image

    # Convert NumPy array to PIL image
    masked_image_pil = Image.fromarray(masked_image)

    # Apply transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # Ensure these values match training
    ])

    masked_tensor = transform(masked_image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        classification_output = classification_model(masked_tensor)  # Output logits for both classes
        probabilities = torch.softmax(classification_output, dim=1)  # Convert logits to probabilities
        risk_prediction = probabilities[0, 1].item()  # Probability for "High Risk" class
        risk_status = "High Risk" if risk_prediction > 0.5 else "Low Risk"

    # --- DISPLAY RESULTS ---
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Original Image")
        st.image(original_image_resized, channels="GRAY", use_container_width=True)

    with col2:
        st.subheader("Predicted Mask")
        st.image(pred_mask * 255, channels="GRAY", use_container_width=True)

    with col3:
        st.subheader("Masked Image")
        st.image(thermal_image, use_container_width=True)

    with col4:
        st.subheader("Risk Classification")
        st.write(f"**Risk Status:** {risk_status}")
        st.write(f"**Risk Score:** {risk_prediction:.2f}")

    # --- DOWNLOAD OPTION ---
    result_img = Image.fromarray(pred_mask * 255)
    buf = BytesIO()
    result_img.save(buf, format="PNG")
    byte_im = buf.getvalue()
    st.download_button(
        label="Download Mask",
        data=byte_im,
        file_name="predicted_mask.png",
        mime="image/png"
    )