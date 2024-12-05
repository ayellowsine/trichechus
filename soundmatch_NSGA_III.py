import spiegelib as spgl
import numpy as np
import tabulate
import os

# Define function for extracting ground truth from filenames
def extract_ground_truth(filename):
    """Extract ground truth parameters from hexadecimal filename."""
    name_parts = filename.split('.')[0]  # Remove file extension
    pitch = int(name_parts[:2], 16)  # First 2 characters for pitch
    cc_values = [int(name_parts[i:i + 2], 16) for i in range(2, len(name_parts), 2)]
    return [pitch] + cc_values

# Define parameter names
PARAMETER_NAMES = [
    "Pitch",
    "Generator Waveform",
    "Generator Spectrum",
    "Generator Inversion",
    "Sub Content",
    "Sub FM Rate",
    "Noise Level"
]

# Define function to calculate accuracy
def calculate_accuracy(predicted, actual):
    return 100 - abs(predicted - actual) / 127 * 100

# Define function to convert pitch to note name
def pitch_to_note(pitch):
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    note = NOTES[pitch % 12]
    return f"{note}-{octave}"

# Soundmatching process
def process_soundmatching(use_hardware):
    print("Starting soundmatching process...")
    
    # Placeholder for synth
    synth = None

    # Define the evaluation folder
    evaluation_folder = './evaluation'

    # Extract all filenames
    filenames = sorted(os.listdir(evaluation_folder))

    # Load target audio files
    targets = spgl.AudioBuffer.load_folder(evaluation_folder)

    for i, (target_audio, filename) in enumerate(zip(targets, filenames), start=1):
        print(f"Processing audio file {i}/{len(targets)}: {filename}")
        
        # Extract ground truth from filename
        true_params = extract_ground_truth(filename)
        true_pitch = true_params[0]
        true_cc_values = true_params[1:]

        # Mock predicted values for example purposes
        predicted_params = [
            np.random.randint(0, 127) for _ in range(len(PARAMETER_NAMES))
        ]
        
        predicted_pitch = predicted_params[0]
        predicted_cc_values = predicted_params[1:]

        # Calculate accuracy
        pitch_accuracy = calculate_accuracy(predicted_pitch, true_pitch)
        cc_accuracies = [
            calculate_accuracy(pred, actual)
            for pred, actual in zip(predicted_cc_values, true_cc_values)
        ]

        # Prepare table data
        table_data = []
        table_data.append([
            "Pitch",
            pitch_to_note(predicted_pitch),
            pitch_to_note(true_pitch),
            f"{pitch_accuracy:.2f}%"
        ])
        for idx, (name, pred, actual, acc) in enumerate(zip(
            PARAMETER_NAMES[1:], predicted_cc_values, true_cc_values, cc_accuracies
        )):
            table_data.append([
                name,
                pred,
                actual,
                f"{acc:.2f}%"
            ])
        
        # Display tabulation for this file
        print(tabulate.tabulate(
            table_data,
            headers=["PARAMETER", "PREDICTED", "ACTUAL", "ACCURACY"],
            tablefmt="grid"
        ))

process_soundmatching(use_hardware=False)
