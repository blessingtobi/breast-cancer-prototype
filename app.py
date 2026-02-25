import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import cv2
import torch.nn.functional as F
import gdown
import os

# ---------------------------------
# Download Model from Google Drive (if not exists)
# ---------------------------------

MODEL_URL = "https://drive.google.com/uc?id=1gqMYicKV391QIjj3h_TpfXiCvKURQQQt"
MODEL_PATH = "model.pth"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

# ---------------------------------
# Load Model (Stable)
# ---------------------------------

@st.cache_resource
def load_model():
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 2)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))

    # Disable ALL inplace ReLU to avoid autograd errors
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    model.eval()
    return model

model = load_model()
class_names = ["Benign", "Malignant"]

# ---------------------------------
# Image Transform
# ---------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---------------------------------
# Grad-CAM Function (Stable)
# ---------------------------------

def generate_gradcam(model, image_tensor):

    model.eval()
    gradients = []
    activations = []

    target_layer = model.features.denseblock4.denselayer16.conv2

    def forward_hook(module, input, output):
        activations.append(output.clone())

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0].clone())

    f_handle = target_layer.register_forward_hook(forward_hook)
    b_handle = target_layer.register_full_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    f_handle.remove()
    b_handle.remove()

    if len(gradients) == 0 or len(activations) == 0:
        return np.zeros((224, 224))

    grad = gradients[0]
    act = activations[0]

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze().clone()
    cam = F.relu(cam)

    cam = cam.detach().cpu().numpy()

    cam -= np.min(cam)
    if np.max(cam) != 0:
        cam /= np.max(cam)

    cam = cv2.resize(cam, (224, 224))
    cam = cv2.GaussianBlur(cam, (5, 5), 0)

    return cam

# ---------------------------------
# Streamlit UI
# ---------------------------------

st.title("AI-Based Breast Cancer Detection System — Prototype")
st.write("Upload a mammogram image to get prediction and Grad-CAM visualization.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_tensor = transform(image).unsqueeze(0)

    # ----------------------------
    # Prediction
    # ----------------------------
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, prediction = torch.max(probs, 1)

    st.subheader("Prediction Result")
    st.write(f"Class: **{class_names[prediction.item()]}**")
    st.write(f"Confidence: {confidence.item()*100:.2f}%")

    # ----------------------------
    # Grad-CAM
    # ----------------------------
    cam = generate_gradcam(model, input_tensor)

    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = heatmap.astype(np.float32) / 255.0

    original = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    original = cv2.resize(original, (224, 224)).astype(np.float32) / 255.0
    original = np.stack([original] * 3, axis=-1)

    overlay = 0.5 * heatmap + 0.5 * original
    overlay = np.clip(overlay, 0, 1)

    st.subheader("Grad-CAM Visualization")
    st.image(overlay, use_column_width=True)