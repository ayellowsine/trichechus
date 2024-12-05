import os
import numpy as np
import spiegelib as spgl
# import random
from sklearn.preprocessing import StandardScaler
import pickle

# Define MFCC feature extractor for feature extraction and training consistency
mfcc_extractor = spgl.features.MFCC(num_mfccs=13, frame_size=2048, hop_size=1024, time_major=True)
# Configure `lstm_extractor` identically for compatibility with training and evaluation scripts
lstm_extractor = spgl.features.MFCC(num_mfccs=13, frame_size=2048, hop_size=1024, time_major=True, scale=True)

# Define STFT feature extractor
stft_extractor = spgl.features.STFT(fft_size=512, hop_size=256, output='magnitude', time_major=True)

# Path to folder of WAV files
folder_path = '11-05'
mfcc_feature_list = []
stft_feature_list = []

# Loop through WAV files and extract features
for filename in os.listdir(folder_path):
    if filename.endswith(".wav"):
        file_path = os.path.join(folder_path, filename)
        audio = spgl.AudioBuffer(file_path)
        
        # Extract and store MFCC features
        mfcc_features = mfcc_extractor(audio)
        mfcc_feature_list.append(mfcc_features)
        
        # Extract and store STFT features
        stft_features = stft_extractor(audio)
        stft_feature_list.append(stft_features)

# Convert lists to arrays for analysis and saving
mfcc_features_array = np.array(mfcc_feature_list)
stft_features_array = np.array(stft_feature_list)

# Save to separate .npy files
np.save("mfcc_features.npy", mfcc_features_array)
np.save("stft_features.npy", stft_features_array)

# Optional: Scale the MFCC features and save the scaler
scaler = StandardScaler()
# Fit scaler on flattened MFCC features (excluding batch dimension)
scaler.fit(mfcc_features_array.reshape(-1, mfcc_features_array.shape[-1]))

# Apply scaling and reshape back to original dimensions
mfcc_features_scaled = scaler.transform(
    mfcc_features_array.reshape(-1, mfcc_features_array.shape[-1])
).reshape(mfcc_features_array.shape)

# Save the scaled MFCC features
np.save("mfcc_features_scaled.npy", mfcc_features_scaled)

# Save the scaler for future use in training and evaluation
with open('mfcc_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("Feature extraction, scaling, and saving complete.")
