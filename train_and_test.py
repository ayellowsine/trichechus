import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pickle

# Load feature data
mfcc_features = np.load("mfcc_features_scaled.npy")  # Already scaled MFCC features
params = np.load("trainParams.npy")  # MIDI parameters: pitch + six CC values

# Split into training and testing datasets
trainFeatures, testFeatures, trainParams, testParams = train_test_split(
    mfcc_features, params, test_size=0.2, random_state=42
)

# Save training and testing datasets
np.save("mfcc_train_features_scaled.npy", trainFeatures)
np.save("mfcc_test_features_scaled.npy", testFeatures)
np.save("trainParams.npy", trainParams)
np.save("testParams.npy", testParams)

print("Training and testing datasets saved.")

# Scale MIDI parameter data
param_scaler = MinMaxScaler(feature_range=(0, 127))  # MIDI values range from 0 to 127
param_scaler.fit(trainParams)  # Fit scaler on training data

# Save parameter scaler for later use
with open("params_scaler.pkl", "wb") as f:
    pickle.dump(param_scaler, f)

print("Parameter scaler saved as params_scaler.pkl.")

# (Optional) Verify the scaling
scaled_params = param_scaler.transform(trainParams)
print(f"First 5 scaled parameter rows:\n{scaled_params[:5]}")
