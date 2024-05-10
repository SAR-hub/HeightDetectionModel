import os
import numpy as np
import torch
import cv2
from torchvision import transforms
from network import UNet as HUNet

def process_image(image_path):
    # Load model
    model_h = HUNet(128)
    pretrained_model_h = torch.load('/Users/syedaamna/Downloads/HeightDetection/model_ep_48.pth.tar', map_location=torch.device('cpu'))
    model_h.load_state_dict(pretrained_model_h["state_dict"])

    # Read Image
    X = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB).astype('float32')
    scale = 128 / max(X.shape[:2])
    
    X_scaled = cv2.resize(X, (0,0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR) 
    
    # Padding if necessary
    if X_scaled.shape[1] > X_scaled.shape[0]:
        p_a = (128 - X_scaled.shape[0])//2
        p_b = (128 - X_scaled.shape[0])-p_a
        X = np.pad(X_scaled, [(p_a, p_b), (0, 0), (0,0)], mode='constant')
    elif X_scaled.shape[1] <= X_scaled.shape[0]:
        p_a = (128 - X_scaled.shape[1])//2
        p_b = (128 - X_scaled.shape[1])-p_a
        X = np.pad(X_scaled, [(0, 0), (p_a, p_b), (0,0)], mode='constant') 
    
    X /= 255
    X = transforms.ToTensor()(X).unsqueeze(0)
    
    model_h.eval()
    with torch.no_grad():
        _, _, h_p = model_h(X)
    
    return h_p.item()

# Get the image file path
image_folder = "/Users/syedaamna/Downloads/HeightDetection/temp_images/"
image_files = [file for file in os.listdir(image_folder) if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png')]

# Check if only one image file is present
if len(image_files) != 1:
    raise ValueError("Expected exactly one image file in the directory")

image_path = os.path.join(image_folder, image_files[0])

# Predict height
predicted_height = process_image(image_path)
predicted_height_cm = predicted_height * 100
print(f"Predicted Height: {predicted_height_cm:.2f}")
