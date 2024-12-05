import os
import numpy as np
import spiegelib as spgl
from sklearn.preprocessing import StandardScaler
import pickle

# Ensure identical configurations for MFCC extractor
mfcc_extractor = spgl.features.MFCC(num_mfccs=13, frame_size=2048, hop_size=1024, time_major=True)

# Path to evaluation folder
folder_path = 'evaluation'
mfcc_feature_list = []

# Debugging: Track file processing and shapes
print("Starting feature extraction from evaluation WAV files...")

# Extract features from all WAV files in the evaluation folder
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        audio = spgl.AudioBuffer(file_path)
        
        # Extract MFCC features
        mfcc_features = mfcc_extractor(audio)
        mfcc_feature_list.append(mfcc_features)

        # Debugging: Print shape of features for each file
        print(f"Processed {filename}: MFCC Shape {mfcc_features.shape}")

# Convert feature list to array
mfcc_features_array = np.array(mfcc_feature_list)
print("MFCC Features Array Shape:", mfcc_features_array.shape)  # Debugging

# Load the scaler used during training
with open('mfcc_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Apply scaling
mfcc_features_scaled = scaler.transform(
    mfcc_features_array.reshape(-1, mfcc_features_array.shape[-1])
).reshape(mfcc_features_array.shape)

# Debugging: Verify shape after scaling
print("MFCC Features Scaled Shape:", mfcc_features_scaled.shape)

# Save scaled features
np.save("mfcc_test_features_scaled.npy", mfcc_features_scaled)

print("Test feature extraction and scaling complete.")

# Debugging: Compare shapes with training features
train_features = np.load("mfcc_features_scaled.npy")
print("Training Features Shape:", train_features.shape)

# Debugging: Ensure compatibility
if train_features.shape[1:] != mfcc_features_scaled.shape[1:]:
    print("Warning: Shape mismatch between training and test feature arrays!")
    print(f"Training Shape: {train_features.shape[1:]}, Test Shape: {mfcc_features_scaled.shape[1:]}")
else:
    print("Shapes are compatible between training and test feature arrays.")
