# skin_cancer_app.py
import streamlit as st
import torch
import torch.nn as nn
import timm
import numpy as np
from pathlib import Path
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from io import BytesIO
from datetime import datetime

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="Skin Cancer Classifier",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------
# Constants
# ---------------------------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
CLASS_DESCRIPTIONS = {
    'MEL': 'Melanoma - malignant skin cancer that develops from melanocytes',
    'NV': 'Melanocytic Nevus - common mole, usually benign',
    'BCC': 'Basal Cell Carcinoma - most common type of skin cancer',
    'AKIEC': 'Actinic Keratosis / Intraepithelial Carcinoma - pre-cancerous or early stage skin cancer',
    'BKL': 'Benign Keratosis - non-cancerous skin growth (seborrheic keratosis, solar lentigo)',
    'DF': 'Dermatofibroma - common benign skin nodule',
    'VASC': 'Vascular Lesion - abnormality of blood vessels (hemangiomas, angiomas)'
}

# ---------------------------
# Models
# ---------------------------
class SkinCancerModel(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 7):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        feature_dim = getattr(self.backbone, "num_features", None)
        if feature_dim is None:
            # Fallback for some timm models
            if hasattr(self.backbone, "num_features"):
                feature_dim = self.backbone.num_features
            else:
                raise RuntimeError("Could not determine feature dimension for backbone.")
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(feature_dim),
            nn.Dropout(0.5),
            nn.Linear(feature_dim, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

def load_state_dict_flexible(state):
    if isinstance(state, dict):
        return state.get('model_state_dict') or state.get('state_dict') or state
    return state

class EnsembleModel(nn.Module):
    def __init__(self, model_names, models_dir: Path, num_classes=7):
        super().__init__()
        self.models = nn.ModuleList([SkinCancerModel(m, num_classes).to(DEVICE) for m in model_names])
        for i, m_name in enumerate(model_names):
            path = models_dir / f"{m_name}_best.pth"
            if not path.exists():
                raise FileNotFoundError(f"Model weights not found: {path}")
            state = torch.load(path, map_location=DEVICE)
            state_dict = load_state_dict_flexible(state)
            # Allow minor head/bn differences
            missing, unexpected = self.models[i].load_state_dict(state_dict, strict=False)
            if unexpected:
                st.warning(f"[{m_name}] Unexpected keys in state_dict ignored: {unexpected}")
            if missing:
                st.warning(f"[{m_name}] Missing keys when loading state_dict: {missing}")
            self.models[i].eval()

    @torch.no_grad()
    def forward(self, x, temperature: float = 1.0):
        logits_list = []
        for model in self.models:
            logits_list.append(model(x))
        stacked = torch.stack(logits_list)
        avg_logits = torch.mean(stacked, dim=0)
        if temperature and temperature > 1e-6:
            avg_logits = avg_logits / float(temperature)
        return torch.softmax(avg_logits, dim=1)

def find_target_layer_for_cam(model: SkinCancerModel):
    # 1) ConvNeXt: stages[-1].blocks[-1]
    try:
        return model.backbone.stages[-1].blocks[-1]
    except Exception:
        pass
    # 2) Swin: layers/stages[-1].blocks[-1]
    for attr in ("layers", "stages"):
        if hasattr(model.backbone, attr):
            last_stage = getattr(model.backbone, attr)[-1]
            if hasattr(last_stage, "blocks") and len(last_stage.blocks) > 0:
                return last_stage.blocks[-1]
    # 3) EfficientNet-like
    if hasattr(model.backbone, "blocks") and len(model.backbone.blocks) > 0:
        return model.backbone.blocks[-1]
    # 4) Last Conv2d as a robust fallback
    last_conv = None
    for m in model.backbone.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is not None:
        return last_conv
    raise RuntimeError("Could not find a suitable convolutional layer for Grad-CAM.")

# ---------------------------
# UI
# ---------------------------
st.title("Skin Cancer Classification with XAI")
st.markdown("""
This app classifies skin lesions into **7 classes** and explains its decision with **Grad-CAM**.
**Note:** This is a research/demo tool and not a medical device.
""")

# Sidebar
st.sidebar.header("Model Settings")
temperature = st.sidebar.slider("Temperature Scaling", 0.1, 1.0, 0.4, 0.1,
                                help="Lower values sharpen the softmax and increase apparent confidence.")
use_ensemble = st.sidebar.checkbox("Use Full Ensemble (3 models)", True,
                                   help="EfficientNetV2-S + ConvNeXt-Tiny + Swin-Tiny")
use_tta = st.sidebar.checkbox("Use Test-Time Augmentation", True,
                              help="Average predictions over flips/rotations/brightness.")

st.sidebar.header("Class Information")
for c in CLASS_NAMES:
    st.sidebar.markdown(f"**{c}**: {CLASS_DESCRIPTIONS[c]}")

# ---------------------------
# Load models (cached)
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_models_cached(use_ens: bool):
    models_dir = Path.cwd() / "trained_models"
    if not models_dir.exists():
        # Optional fallback path (adjust if needed)
        alt = Path(r"C:\Users\sudha\Downloads\ISIC\skin_cancer_project\trained_models")
        if alt.exists():
            models_dir = alt
    if not models_dir.exists():
        raise FileNotFoundError("Could not find 'trained_models' directory with weight files.")

    model_names = ["tf_efficientnetv2_s", "convnext_tiny", "swin_tiny_patch4_window7_224"] if use_ens \
        else ["tf_efficientnetv2_s"]

    ensemble = EnsembleModel(model_names, models_dir).to(DEVICE)

    # XAI model: use EfficientNetV2-S for CAM target
    xai_name = "tf_efficientnetv2_s"
    xai_model = SkinCancerModel(xai_name).to(DEVICE)
    xai_path = models_dir / f"{xai_name}_best.pth"
    xai_state = load_state_dict_flexible(torch.load(xai_path, map_location=DEVICE))
    xai_missing, xai_unexp = xai_model.load_state_dict(xai_state, strict=False)
    if xai_unexp:
        st.warning(f"[XAI] Unexpected keys ignored: {xai_unexp}")
    if xai_missing:
        st.warning(f"[XAI] Missing keys: {xai_missing}")
    xai_model.eval()

    target_layer = find_target_layer_for_cam(xai_model)
    return ensemble, xai_model, target_layer, model_names, models_dir

# ---------------------------
# Preprocessing
# ---------------------------
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ---------------------------
# File upload
# ---------------------------
uploaded_file = st.file_uploader("Upload a dermatoscopic image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.info("👆 Upload a skin lesion image to get started")
    st.markdown("### About this application")
    st.markdown("""
- Dataset: ISIC 2018 (HAM10000)
- Backbones: **EfficientNetV2-S**, **ConvNeXt-Tiny**, **Swin-Tiny**
- XAI: **Grad-CAM** over EfficientNetV2-S's last conv block
- Inference: Ensemble + optional **TTA** + **temperature scaling**
""")
else:
    # Load once here (cached)
    try:
        with st.spinner("Loading models..."):
            ensemble, xai_model, target_layer, model_names, models_dir = load_models_cached(use_ensemble)
    except Exception as e:
        st.error(f"Model loading error: {e}")
        st.stop()

    st.success(f"Using {'ensemble of ' + str(len(model_names)) + ' models' if use_ensemble else 'single model'} "
               f"with {'TTA' if use_tta else 'no TTA'} on **{DEVICE.type.upper()}**")

    # Read image
    img = Image.open(uploaded_file).convert("RGB")

    # TTA views
    if use_tta:
        tta_imgs = [
            img,
            TF.hflip(img),
            TF.vflip(img),
            TF.rotate(img, 90),
            TF.adjust_brightness(img, 1.2),
        ]
    else:
        tta_imgs = [img]

    # Predict
    probs_accum = []
    with torch.no_grad():
        for im in tta_imgs:
            t = val_transform(im).unsqueeze(0).to(DEVICE)
            probs_accum.append(ensemble(t, temperature=temperature))
    ensemble_probs = torch.mean(torch.stack(probs_accum, dim=0), dim=0)  # [1, C]

    confidence, pred_idx = torch.max(ensemble_probs, 1)
    pred_class = CLASS_NAMES[pred_idx.item()]

    # Top-3
    topk = min(3, len(CLASS_NAMES))
    topk_conf, topk_idx = torch.topk(ensemble_probs, k=topk, dim=1)
    topk_conf = topk_conf[0].tolist()
    topk_idx = topk_idx[0].tolist()
    top3_text = " | ".join([f"{CLASS_NAMES[i]}: {p:.1%}" for i, p in zip(topk_idx, topk_conf)])

    # Grad-CAM (NOTE: no use_cuda argument)
    input_tensor = val_transform(img).unsqueeze(0).to(DEVICE)
    targets = [ClassifierOutputTarget(pred_idx.item())]
    try:
        with GradCAM(model=xai_model, target_layers=[target_layer]) as cam:
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
        overlay = show_cam_on_image(
            np.array(img.resize((224, 224))) / 255.0, grayscale_cam, use_rgb=True
        )
    except Exception as e:
        st.warning(f"Grad-CAM failed ({e}). Showing only original image.")
        overlay = np.array(img.resize((224, 224)))

    # Layout
    col1, col2 = st.columns(2)
    with col1:
        st.image(img.resize((224, 224)), caption="Uploaded Image", use_container_width=True)
    with col2:
        st.image(overlay, caption="XAI Explanation (Grad-CAM)", use_container_width=True)

    # Prediction info
    st.header(f"Prediction: {pred_class} ({CLASS_DESCRIPTIONS[pred_class].split(' - ')[0]})")
    st.subheader(f"Confidence: {confidence.item():.2%}")

    st.subheader("Top-3 Predictions")
    for i, (conf, idx) in enumerate(zip(topk_conf, topk_idx)):
        name = CLASS_NAMES[idx]
        st.markdown(f"**{i+1}. {name}**: {conf:.2%} — {CLASS_DESCRIPTIONS[name].split(' - ')[0]}")

    # Downloadable combined figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(np.array(img.resize((224, 224))))
    axes[0].set_title('Uploaded Image')
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title('XAI Explanation (Grad-CAM)')
    axes[1].axis('off')

    plt.suptitle(f"Pred: {pred_class} (Confidence: {confidence.item():.2%})\nTop-3: {top3_text}", fontsize=14)
    plt.tight_layout()

    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.download_button(
        label="💾 Save Results",
        data=buf,
        file_name=f"skin_cancer_prediction_{pred_class}_{timestamp}.png",
        mime="image/png"
    )

    # Details
    st.markdown("---")
    st.markdown(f"### About {pred_class}\n{CLASS_DESCRIPTIONS[pred_class]}")

    with st.expander("Technical Details"):
        st.markdown(f"- **Device**: {DEVICE}")
        st.markdown(f"- **Models**: {', '.join(model_names)}")
        st.markdown(f"- **Temperature**: {temperature}")
        st.markdown(f"- **TTA**: {'Enabled' if use_tta else 'Disabled'}; **#views**: {len(tta_imgs)}")
        st.markdown("**All class probabilities:**")
        probs = ensemble_probs[0].tolist()
        for name, prob in zip(CLASS_NAMES, probs):
            st.markdown(f"- {name}: {prob:.4f} ({prob:.2%})")

# import streamlit as st
# import torch
# import torch.nn as nn
# import timm
# import numpy as np
# from pathlib import Path
# import torchvision.transforms as transforms
# import torchvision.transforms.functional as TF
# from PIL import Image
# import matplotlib.pyplot as plt
# from pytorch_grad_cam import GradCAM
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# import base64
# from io import BytesIO
# from datetime import datetime

# # Set page config
# st.set_page_config(
#     page_title="Skin Cancer Classifier", 
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Constants
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# CLASS_NAMES = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
# CLASS_DESCRIPTIONS = {
#     'MEL': 'Melanoma - malignant skin cancer that develops from melanocytes',
#     'NV': 'Melanocytic Nevus - common mole, usually benign',
#     'BCC': 'Basal Cell Carcinoma - most common type of skin cancer',
#     'AKIEC': 'Actinic Keratosis / Intraepithelial Carcinoma - pre-cancerous or early stage skin cancer',
#     'BKL': 'Benign Keratosis - non-cancerous skin growth (seborrheic keratosis, solar lentigo)',
#     'DF': 'Dermatofibroma - common benign skin nodule',
#     'VASC': 'Vascular Lesion - abnormality of blood vessels (hemangiomas, angiomas)'
# }

# # Model definitions
# class SkinCancerModel(nn.Module):
#     def __init__(self, model_name: str, num_classes: int = 7):
#         super().__init__()
#         self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
#         feature_dim = self.backbone.num_features
#         self.classifier = nn.Sequential(
#             nn.BatchNorm1d(feature_dim),
#             nn.Dropout(0.5),
#             nn.Linear(feature_dim, num_classes)
#         )

#     def forward(self, x):
#         features = self.backbone(x)
#         return self.classifier(features)

# def load_state_dict_flexible(state):
#     # Accept various checkpoint formats
#     if isinstance(state, dict):
#         return state.get('model_state_dict') or state.get('state_dict') or state
#     return state

# class EnsembleModel(nn.Module):
#     def __init__(self, model_names, models_dir, num_classes=7):
#         super().__init__()
#         self.models = nn.ModuleList([SkinCancerModel(m, num_classes).to(DEVICE) for m in model_names])
#         for i, m_name in enumerate(model_names):
#             path = models_dir / f"{m_name}_best.pth"
#             if not path.exists():
#                 raise FileNotFoundError(f"Model weights not found: {path}")
#             state = torch.load(path, map_location=DEVICE)
#             state_dict = load_state_dict_flexible(state)
#             self.models[i].load_state_dict(state_dict)
#             self.models[i].eval()

#     def forward(self, x, temperature: float = 1.0):
#         logits_list = []
#         with torch.no_grad():
#             for model in self.models:
#                 logits_list.append(model(x))
#         stacked = torch.stack(logits_list)
#         avg_logits = torch.mean(stacked, dim=0)
#         if temperature and temperature > 1e-6:
#             avg_logits = avg_logits / float(temperature)
#         return torch.softmax(avg_logits, dim=1)

# def find_target_layer_for_cam(model: SkinCancerModel):
#     # Prefer architecture-specific last conv block
#     # 1) convnext: stages[-1].blocks[-1]
#     try:
#         return model.backbone.stages[-1].blocks[-1]
#     except Exception:
#         pass
#     # 2) swin: layers[-1]/stages[-1].blocks[-1]
#     for attr in ("layers", "stages"):
#         if hasattr(model.backbone, attr):
#             last_stage = getattr(model.backbone, attr)[-1]
#             if hasattr(last_stage, "blocks"):
#                 return last_stage.blocks[-1]
#     # 3) efficientnet-like: blocks[-1]
#     if hasattr(model.backbone, "blocks"):
#         try:
#             return model.backbone.blocks[-1]
#         except Exception:
#             pass
#     # 4) robust fallback: last Conv2d in the backbone
#     last_conv = None
#     for m in model.backbone.modules():
#         if isinstance(m, nn.Conv2d):
#             last_conv = m
#     if last_conv is not None:
#         return last_conv
#     raise RuntimeError("Could not find a suitable convolutional layer for Grad-CAM.")

# # App title and description
# st.title("Skin Cancer Classification with XAI")
# st.markdown("""
# This application uses deep learning to classify skin lesions into 7 categories with explainable AI visualization.
# Upload an image to get a prediction with visual explanation of which areas influenced the model's decision.
# """)

# # Sidebar with model settings
# st.sidebar.header("Model Settings")
# temperature = st.sidebar.slider(
#     "Temperature Scaling", 
#     0.1, 1.0, 0.4, 0.1, 
#     help="Lower values increase prediction confidence (0.4 recommended)"
# )

# use_ensemble = st.sidebar.checkbox(
#     "Use Full Ensemble (3 models)", 
#     True,
#     help="Use all 3 models for better accuracy (recommended)"
# )

# use_tta = st.sidebar.checkbox(
#     "Use Test-Time Augmentation", 
#     True,
#     help="Apply multiple transformations for robust prediction (recommended)"
# )

# # Information section in sidebar
# st.sidebar.header("Class Information")
# for class_code in CLASS_NAMES:
#     st.sidebar.markdown(f"**{class_code}**: {CLASS_DESCRIPTIONS[class_code]}")

# # Load models
# @st.cache_resource
# def load_models(use_ensemble=True):
#     # Resolve models directory
#     models_dir = Path.cwd() / "trained_models"
#     if not models_dir.exists():
#         # Fallback to known absolute path
#         models_dir = Path(r"C:\Users\sudha\Downloads\ISIC\skin_cancer_project\trained_models")
#         if not models_dir.exists():
#             st.error(f"Model directory not found: {models_dir}")
#             st.stop()
    
#     st.info(f"Loading models from: {models_dir}")
    
#     # Select models based on ensemble setting
#     if use_ensemble:
#         model_names = ["tf_efficientnetv2_s", "convnext_tiny", "swin_tiny_patch4_window7_224"]
#     else:
#         model_names = ["tf_efficientnetv2_s"]
    
#     # Create and load ensemble
#     try:
#         ensemble = EnsembleModel(model_names, models_dir).to(DEVICE)
        
#         # Load XAI model (always use efficientnet for visualization)
#         xai_backbone_name = "tf_efficientnetv2_s"
#         xai_model = SkinCancerModel(xai_backbone_name).to(DEVICE)
#         xai_path = models_dir / f"{xai_backbone_name}_best.pth"
#         state = torch.load(xai_path, map_location=DEVICE)
#         xai_state = load_state_dict_flexible(state)
#         xai_model.load_state_dict(xai_state)
#         xai_model.eval()
        
#         target_layer = find_target_layer_for_cam(xai_model)
        
#         return ensemble, xai_model, target_layer, model_names
#     except Exception as e:
#         st.error(f"Error loading models: {str(e)}")
#         st.stop()

# # Image preprocessing
# val_transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # File uploader
# uploaded_file = st.file_uploader("Upload a skin lesion image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Load models
#     with st.spinner("Loading models..."):
#         ensemble, xai_model, target_layer, model_names = load_models(use_ensemble)
    
#     # Display model configuration
#     st.success(f"Using {'ensemble of ' + str(len(model_names)) + ' models' if use_ensemble else 'single model'} with {'TTA' if use_tta else 'no TTA'}")
    
#     # Display the uploaded image
#     img = Image.open(uploaded_file).convert("RGB")
    
#     # Create TTA images if enabled
#     if use_tta:
#         tta_imgs = [
#             img,                          # Original
#             TF.hflip(img),                # Horizontal flip
#             TF.vflip(img),                # Vertical flip
#             TF.rotate(img, 90),           # 90-degree rotation
#             TF.adjust_brightness(img, 1.2) # Brightness adjustment
#         ]
#     else:
#         tta_imgs = [img]
    
#     # Process with ensemble
#     probs_accum = []
#     for im in tta_imgs:
#         t = val_transform(im).unsqueeze(0).to(DEVICE)
#         probs_accum.append(ensemble(t, temperature=temperature))
    
#     # Average over views, then get prediction
#     ensemble_probs = torch.mean(torch.stack(probs_accum, dim=0), dim=0)
    
#     confidence, pred_idx = torch.max(ensemble_probs, 1)
#     pred_class = CLASS_NAMES[pred_idx.item()]
    
#     # Top-3
#     topk_conf, topk_idx = torch.topk(ensemble_probs, k=min(3, len(CLASS_NAMES)), dim=1)
#     topk_conf = topk_conf[0].tolist()
#     topk_idx = topk_idx[0].tolist()
#     top3_text = " | ".join([f"{CLASS_NAMES[i]}: {p:.1%}" for i, p in zip(topk_idx, topk_conf)])
    
#     # GradCAM
#     input_tensor = val_transform(img).unsqueeze(0).to(DEVICE)
#     targets = [ClassifierOutputTarget(pred_idx.item())]
#     with GradCAM(model=xai_model, target_layers=[target_layer]) as cam:
#         grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0, :]
#         visualization = show_cam_on_image(
#             np.array(img.resize((224, 224))) / 255.0,
#             grayscale_cam,
#             use_rgb=True
#         )
    
#     # Display results in columns
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.image(img.resize((224, 224)), caption="Uploaded Image", use_container_width=True)
    
#     with col2:
#         st.image(visualization, caption="XAI Explanation", use_container_width=True)
    
#     # Display prediction
#     st.header(f"Prediction: {pred_class} ({CLASS_DESCRIPTIONS[pred_class].split(' - ')[0]})")
#     st.subheader(f"Confidence: {confidence.item():.2%}")
    
#     # Display top-3 predictions
#     st.subheader("Top-3 Predictions:")
#     for i, (conf, idx) in enumerate(zip(topk_conf, topk_idx)):
#         class_name = CLASS_NAMES[idx]
#         st.markdown(f"**{i+1}. {class_name}**: {conf:.2%} - {CLASS_DESCRIPTIONS[class_name].split(' - ')[0]}")
    
#     # Create a combined image for download
#     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#     axes[0].imshow(np.array(img.resize((224, 224))))
#     axes[0].set_title('Uploaded Image')
#     axes[0].axis('off')
    
#     axes[1].imshow(visualization)
#     axes[1].set_title('XAI Explanation')
#     axes[1].axis('off')
    
#     plt.suptitle(
#         f"Pred: {pred_class} (Confidence: {confidence.item():.2%})\nTop-3: {top3_text}",
#         fontsize=14
#     )
#     plt.tight_layout()
    
#     # Save figure to BytesIO
#     buf = BytesIO()
#     plt.savefig(buf, format='png', dpi=150)
#     buf.seek(0)
    
#     # Create download button
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     st.download_button(
#         label="💾 Save Results",
#         data=buf,
#         file_name=f"skin_cancer_prediction_{pred_class}_{timestamp}.png",
#         mime="image/png",
#         help="Download the prediction results as an image"
#     )
    
#     # Additional information about the prediction
#     st.markdown("---")
#     st.markdown(f"### About {pred_class}: {CLASS_DESCRIPTIONS[pred_class]}")
    
#     # Technical details
#     with st.expander("Technical Details"):
#         st.markdown(f"**Device used**: {DEVICE}")
#         st.markdown(f"**Models used**: {', '.join(model_names)}")
#         st.markdown(f"**Temperature scaling**: {temperature}")
#         st.markdown(f"**Test-time augmentation**: {'Enabled' if use_tta else 'Disabled'}")
#         st.markdown(f"**Number of TTA views**: {len(tta_imgs)}")
        
#         # Show all class probabilities
#         st.markdown("**All class probabilities:**")
#         probs = ensemble_probs[0].tolist()
#         for i, (name, prob) in enumerate(zip(CLASS_NAMES, probs)):
#             st.markdown(f"- {name}: {prob:.4f} ({prob:.2%})")
# else:
#     # Show sample images when no file is uploaded
#     st.info("👆 Upload a skin lesion image to get started")
    
#     # Display information about the model
#     st.markdown("### About this application")
#     st.markdown("""
#     This application uses deep learning to classify skin lesions into 7 different categories.
    
#     The model was trained on the ISIC 2018 Challenge Task 3 dataset (HAM10000), which contains over 10,000 dermatoscopic images.
    
#     The classification is performed using an ensemble of state-of-the-art convolutional neural networks:
#     - EfficientNetV2-S
#     - ConvNeXt Tiny
#     - Swin Transformer Tiny
    
#     For explainability, we use Grad-CAM to highlight the regions of the image that influenced the model's decision.
#     """)
