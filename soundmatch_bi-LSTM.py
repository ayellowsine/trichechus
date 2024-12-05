import os
import numpy as np
import tensorflow as tf
import spiegelib as spgl
from tabulate import tabulate
import pickle

# Define the rms_error loss function
def rms_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Load the trained model
bi_lstm = tf.keras.models.load_model(
    './saved_models/simple_fm_bi_lstm.h5',
    custom_objects={
        "HighwayLayer": spgl.estimator.HighwayLayer,
        "rms_error": rms_error
    }
)

# Load parameter scaler
with open('params_scaler.pkl', 'rb') as f:
    params_scaler = pickle.load(f)

# Helper function to extract ground truth parameters from filenames
def extract_ground_truth(filename):
    name_parts = os.path.splitext(filename)[0]
    pitch = int(name_parts[:2], 16)
    cc_values = [int(name_parts[i:i + 2], 16) for i in range(2, 14, 2)]
    return [pitch] + cc_values

# MIDI parameter names
PARAMETER_NAMES = [
    "Pitch",
    "Generator Waveform",
    "Generator Spectrum",
    "Generator Inversion",
    "Sub Content",
    "Sub FM Rate",
    "Noise Level"
]

# Convert pitch to note names
def pitch_to_note(pitch):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    note = pitch % 12
    return f"{note_names[note]}{octave}"

# Process evaluation folder
evaluation_folder = './evaluation'
files = [f for f in os.listdir(evaluation_folder) if f.endswith('.wav')]

for file_idx, filename in enumerate(files):
    print(f"Processing file {file_idx + 1}/{len(files)}: {filename}")
    
    # Load the target audio
    file_path = os.path.join(evaluation_folder, filename)
    target_audio = spgl.AudioBuffer(file_path)
    
    # Extract ground truth parameters
    true_params = extract_ground_truth(filename)
    true_pitch = pitch_to_note(true_params[0])
    true_ccs = true_params[1:]
    
    # Predict parameters using the model
    input_features = spgl.features.MFCC(num_mfccs=13, frame_size=2048, hop_size=1024, time_major=True)
    normalized_features = input_features(target_audio).reshape(1, -1, 13)  # Adjust shape for LSTM input
    normalized_prediction = bi_lstm.predict(normalized_features)[0]
    predicted_patch = params_scaler.inverse_transform([normalized_prediction])[0]

    # Separate pitch and CC predictions
    predicted_pitch = pitch_to_note(int(predicted_patch[0]))
    predicted_ccs = [int(value) for value in predicted_patch[1:]]
    
    # Calculate accuracy
    accuracies = []
    for pred, actual in zip([predicted_patch[0]] + predicted_ccs, [true_params[0]] + true_ccs):
        if actual == 0:
            accuracy = 100.0 if pred == 0 else 0.0
        else:
            accuracy = max(0, 100.0 - abs((pred - actual) / actual) * 100)
        accuracies.append(accuracy)
    
    # Tabulate results
    rows = []
    rows.append(["Pitch", predicted_pitch, true_pitch, f"{accuracies[0]:.2f}%"])
    for i, param_name in enumerate(PARAMETER_NAMES[1:], start=1):
        rows.append([param_name, predicted_ccs[i - 1], true_ccs[i - 1], f"{accuracies[i]:.2f}%"])
    
    print(f"\nResults for {filename}:")
    print(tabulate(rows, headers=["Parameter", "Predicted", "Actual", "Accuracy"], tablefmt="grid"))

    print("=" * 50)  # Divider for clarity between files
