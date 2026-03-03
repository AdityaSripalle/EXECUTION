import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Brain Tumor Detection",
    layout="wide"
)

st.title("🧠 Brain Tumor Classification System")
st.markdown("Hybrid Deep Learning + Grad-CAM Visualization")

# -------------------------------
# Device
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Model Definition (EDIT if needed)
# -------------------------------
class HybridModel(nn.Module):
    def __init__(self, num_classes=4):
        super(HybridModel, self).__init__()

        from torchvision.models import mobilenet_v3_small
        self.backbone = mobilenet_v3_small(weights=None)
        self.backbone.classifier[3] = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.backbone(x)

# -------------------------------
# Load model
# -------------------------------
@st.cache_resource
def load_model():
    model = HybridModel(num_classes=4)
    model.load_state_dict(
        torch.load("best_hybrid_model.pth", map_location=device)
    )
    model.to(device)
    model.eval()
    return model

model = load_model()

# -------------------------------
# Class labels (EDIT if needed)
# -------------------------------
class_names = [
    "Glioma",
    "Meningioma",
    "No Tumor",
    "Pituitary"
]

# -------------------------------
# Image transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# -------------------------------
# Grad-CAM
# -------------------------------
def generate_gradcam(model, image_tensor):
    gradients = []
    activations = []

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    def forward_hook(module, inp, out):
        activations.append(out)

    # hook last conv layer
    target_layer = model.backbone.features[-1]
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_backward_hook(backward_hook)

    output = model(image_tensor)
    pred_class = output.argmax(dim=1)

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0].cpu().data.numpy()[0]
    acts = activations[0].cpu().data.numpy()[0]

    weights = np.mean(grads, axis=(1, 2))
    cam = np.zeros(acts.shape[1:], dtype=np.float32)

    for i, w in enumerate(weights):
        cam += w * acts[i]

    cam = np.maximum(cam, 0)
    cam = cv2.resize(cam, (224, 224))
    cam = cam - cam.min()
    cam = cam / cam.max()

    fh.remove()
    bh.remove()

    return cam

# -------------------------------
# Prediction function
# -------------------------------
def predict_image(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, 1)

    cam = generate_gradcam(model, img)

    return pred.item(), conf.item(), cam

# -------------------------------
# Batch upload
# -------------------------------
uploaded_files = st.file_uploader(
    "📤 Upload MRI Images",
    type=["jpg", "png", "jpeg"],
    accept_multiple_files=True
)

# -------------------------------
# Processing
# -------------------------------
if uploaded_files:

    for file in uploaded_files:
        image = Image.open(file).convert("RGB")

        col1, col2 = st.columns(2)

        # Prediction
        pred, conf, cam = predict_image(image)

        with col1:
            st.image(image, caption="Original MRI", use_column_width=True)

            st.metric(
                label="Prediction",
                value=class_names[pred]
            )

            st.progress(float(conf))

            st.write(f"Confidence: **{conf*100:.2f}%**")

        # Heatmap
        with col2:
            img_np = np.array(image.resize((224, 224)))

            heatmap = cv2.applyColorMap(
                np.uint8(255 * cam),
                cv2.COLORMAP_JET
            )

            overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

            st.image(overlay, caption="🔥 Tumor Heatmap", use_column_width=True)

st.markdown("---")
st.markdown("✅ Hybrid CNN + Grad-CAM | Local Medical AI System")
