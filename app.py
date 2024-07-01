import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Assuming CustomModel and Configuration are properly defined
# Make sure to import or define these classes

@st.cache_resource
def load_model():
    configuration = Configuration()
    model = CustomModel(configuration).cuda()
    model_path = "/kaggle/input/my-se-resnext-model/se_resnext50_32x4d_19_loss0.05_score0.89_val_loss0.28_val_score0.80.pt"
    model.load_state_dict(torch.load(model_path, map_location='cuda'))
    model.eval()
    return model

def pad_to_multiple(image, multiple=32):
    height, width = image.shape[-2:]
    pad_height = (multiple - height % multiple) % multiple
    pad_width = (multiple - width % multiple) % multiple
    padding = (pad_width // 2, pad_width - pad_width // 2, pad_height // 2, pad_height - pad_height // 2)
    padded_image = F.pad(image, padding, mode='constant', value=0)
    return padded_image, padding

def unpad_from_multiple(padded_image, padding):
    left, right, top, bottom = padding
    height, width = padded_image.shape[-2:]
    return padded_image[..., top:height-bottom, left:width-right]

def process_image(img):
    # Convert PIL Image to NumPy array
    img_array = np.array(img)
    
    # Check and convert the image to grayscale if it's in RGB
    if img_array.ndim == 3 and img_array.shape[-1] == 3:
        img = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        img = img_array  # Use the image as is if it's already grayscale
    
    # Resize the image to the required dimensions
    img = cv2.resize(img, (512, 512))
    
    # Add a channel dimension and convert to tensor
    img = np.expand_dims(img, axis=0)  # Add channel dimension
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


def perform_inference(model, images):
    model.eval()
    with torch.no_grad():
        images = images.to('cuda', dtype=torch.float)
        padded_images, padding = pad_to_multiple(images)
        preds = model(padded_images)
        preds = unpad_from_multiple(preds, padding)
        preds = preds.sigmoid() > 0.5
    return preds.cpu().squeeze()

def visualize_prediction(image, prediction, threshold=0.5):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.imshow(image, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    thresholded_pred = prediction > threshold
    colored_img = np.stack([image]*3, axis=-1)
    red_mask = np.zeros_like(colored_img)
    red_mask[:, :, 0] = 165
    mask = ~thresholded_pred
    overlay = colored_img.copy()
    overlay[mask] = red_mask[mask]

    ax2.imshow(overlay)
    ax2.set_title('Prediction Overlay')
    ax2.axis('off')

    plt.tight_layout()
    return fig

# Streamlit app
st.title('Blood Vessel Segmentation App')

# File uploader for 5 images
uploaded_files = st.file_uploader("Upload 5 consecutive image slices", type=["png", "jpg", "jpeg", 'tif'], accept_multiple_files=True)

if len(uploaded_files) == 5:
    # Process uploaded images
    images = torch.stack([process_image(Image.open(file)) for file in uploaded_files])
    
    # Load model and perform inference
    model = load_model()
    predictions = perform_inference(model, images)
    
    # Let user choose which slice to visualize
    slice_index = st.selectbox("Choose which slice to visualize:", [1, 2, 3, 4, 5]) - 1
    
    # Visualize chosen slice and prediction
    chosen_image = images[slice_index].squeeze().numpy()
    chosen_prediction = predictions[slice_index].squeeze().numpy()
    
    fig = visualize_prediction(chosen_image, chosen_prediction)
    st.pyplot(fig)

else:
    st.warning("Please upload exactly 5 image slices.")
