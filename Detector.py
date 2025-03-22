import numpy as np
import streamlit as st
import cv2
import torch
import os
from torchvision import transforms
from PIL import Image
import tempfile
import torch.nn as nn
import torch.optim as optim
import timm

class Discriminator(nn.Module):
    def __init__(self, num_classes=1):
        super(Discriminator, self).__init__()
        # Load XceptionNet as the backbone
        self.backbone = timm.create_model('legacy_xception', pretrained=True, features_only=True)
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Fully connected layer for binary classification
        self.fc = nn.Linear(self.backbone.feature_info.channels()[-1], num_classes)
        
        # Sigmoid activation for binary classification
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Extract features from XceptionNet
        features = self.backbone(x)[-1]  # Use the last feature map
        
        # Global average pooling
        pooled = self.global_pool(features)
        pooled = pooled.view(pooled.size(0), -1)
        
        # Fully connected layer
        output = self.fc(pooled)
        
        # Sigmoid activation
        output = self.sigmoid(output)
        return output

# --- Preprocessing and Video Frame Extraction ---
def preprocess_frame(frame, transform):
    """Preprocess a single video frame."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    frame = Image.fromarray(frame)  # Convert to PIL image
    frame = transform(frame)  # Apply transformations
    return frame

def process_video(video_path, model, transform, device, target_fps, max_frames=1000):
    """Process the video to classify each frame as real or fake."""
    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        st.error(f"Error: Could not open video {video_path}")
        return 0.5  # Default to "uncertain"

    fps = vidcap.get(cv2.CAP_PROP_FPS)  # Original frame rate of the video
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))  # Total frames in the video

    st.write(f"Video: {video_path}")
    st.write(f"Original FPS: {int(fps)}, Total Frames: {total_frames}")

    # Calculate the interval between frames to achieve the target FPS
    frame_interval = int(fps // target_fps)
    if frame_interval < 1:
        frame_interval = 1  # Ensure at least 1 frame is processed

    frame_predictions = []
    frame_count = 0
    processed_frame_count = 0

    while True:
        success, frame = vidcap.read()
        if not success:
            break  # Exit loop if no frame is read

        # Process frames at the target FPS
        if frame_count % frame_interval == 0:
            if processed_frame_count >= max_frames:
                st.warning(f"Reached maximum frame limit ({max_frames}) for video: {video_path}")
                break

            try:
                frame = preprocess_frame(frame, transform)
                frame = frame.unsqueeze(0).to(device)  # Add batch dimension

                # Predict with the model
                with torch.no_grad():
                    output = model(frame)
                    prediction = torch.sigmoid(output).item()  # Get probability of being fake

                frame_predictions.append(prediction)
                processed_frame_count += 1
            except Exception as e:
                st.error(f"Error processing frame {frame_count}: {e}")
                continue

        frame_count += 1

    vidcap.release()

    # Aggregate predictions (e.g., average probability)
    avg_prediction = np.mean(frame_predictions) if frame_predictions else 0.5
    return avg_prediction

# --- Streamlit App ---
def main():
    st.title("Deepfake Detection App")
    st.write("Upload a video to check if it's real or a deepfake.")

    # File uploader
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
            tmp_file.write(uploaded_file.read())
            video_path = tmp_file.name

        # Load the pre-trained Discriminator model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "discriminator_epoch_6.pth"  # Ensure the model is saved with this name
        model = Discriminator()
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only = True))
        model.to(device)
        model.eval()

        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),  # Converts to [0, 1] range
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

        # Process the video
        avg_prediction = process_video(video_path, model, transform, device, target_fps=1, max_frames=1000)

        # Display results
        st.write("### Results")
        if avg_prediction == 0.5:
            st.error("Error in opening video")
            return
        if avg_prediction < 0.5:
            st.success(f"The video is classified as **Deepfake** (Average prediction: {avg_prediction:.4f})")
        else:
            st.success(f"The video is classified as **Real** (Average prediction: {avg_prediction:.4f})")

        # Clean up the temporary file
        os.unlink(video_path)

if __name__ == "__main__":
    main()
