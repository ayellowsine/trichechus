import os
import numpy as np
import tensorflow as tf
import spiegelib as spgl
import mido
from tabulate import tabulate
import pickle
import soundfile as sf
import time

# Define RMS Error loss function
def rms_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Load the trained bi-LSTM model
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

# MIDI Synth Class
class HardwareSynth:
    def __init__(self, midi_port_name="USB Midi Cable 1"):
        self.midi_port_name = midi_port_name
        self.connect_midi_port()

    def connect_midi_port(self):
        try:
            print(f"Attempting to open MIDI port: {self.midi_port_name}")
            self.midi_port = mido.open_output(self.midi_port_name)
            print(f"MIDI port '{self.midi_port_name}' opened successfully.")
        except Exception as e:
            raise RuntimeError(f"Failed to open MIDI port '{self.midi_port_name}': {e}")

    def send_pitch(self, pitch):
        pitch = max(0, min(127, int(round(pitch))))
        self.midi_port.send(mido.Message('note_on', note=pitch, velocity=127))
        print(f"Pitch Note On -> {pitch}")

    def send_ccs(self, cc_values):
        cc_values = [max(0, min(127, int(round(value)))) for value in cc_values]
        cc_controllers = [70, 16, 23, 18, 21, 83]
        for cc, value in zip(cc_controllers, cc_values):
            self.midi_port.send(mido.Message('control_change', control=cc, value=value))
        print(f"MIDI CCs Sent -> {cc_values}")

    def send_note_off(self, pitch):
        pitch = max(0, min(127, int(round(pitch))))
        self.midi_port.send(mido.Message('note_off', note=pitch))
        print(f"Pitch Note Off -> {pitch}")

# Helper Function: Convert Pitch to Note
def pitch_to_note(pitch):
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = (pitch // 12) - 1
    note = pitch % 12
    return f"{note_names[note]}{octave}"

# Main Function
def main():
    # Prompt for MIDI CC delay
    while True:
        try:
            delay = float(input("Enter delay between MIDI CC messages in seconds (e.g., 0.1): "))
            if delay < 0:
                raise ValueError("Delay must be non-negative.")
            break
        except ValueError as e:
            print(f"Invalid input: {e}")

    # Initialize hardware synthesizer
    midi_synth = HardwareSynth()

    # Load the WAV file
    audio_file = 'example_audio.wav'
    audio_data, sample_rate = sf.read(audio_file)

    if sample_rate != 44100:
        raise ValueError(f"Sample rate mismatch: file={sample_rate}, expected=44100")
    
    print(f"Processing {len(audio_data)} samples from {audio_file}...")

    # Define chunk size (50ms)
    chunk_size = int(0.05 * sample_rate)  # 50ms chunk
    num_chunks = (len(audio_data) + chunk_size - 1) // chunk_size  # Include partial chunks
    first_chunk = True
    last_pitch = None

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = start_idx + chunk_size * 22  # Ensure enough frames for 22 time steps

        # Extract or pad audio chunk
        audio_chunk = audio_data[start_idx:end_idx]
        if len(audio_chunk) < chunk_size * 22:
            padding_needed = chunk_size * 22 - len(audio_chunk)
            audio_chunk = np.pad(audio_chunk, (0, padding_needed))
            print(f"Padding applied to chunk {i+1}/{num_chunks} with {padding_needed} zeros.")

        try:
            audio_buffer = spgl.AudioBuffer(audio_chunk, sample_rate)
            input_features = spgl.features.MFCC(num_mfccs=13, frame_size=2048, hop_size=1024, time_major=True)
            features = input_features(audio_buffer)  # Ensure proper feature shape
            features = features[:22, :].reshape(1, 22, 13)  # Adjust shape for LSTM input
            normalized_prediction = bi_lstm.predict(features)[0]
            predicted_patch = params_scaler.inverse_transform([normalized_prediction])[0]

            # Extract pitch and CC values
            predicted_pitch = predicted_patch[0]
            predicted_ccs = predicted_patch[1:]

            # Debugging Output
            print(f"Chunk {i+1}/{num_chunks}:")
            print(f"Raw Predictions -> Pitch: {predicted_pitch}, CCs: {predicted_ccs}")

            # For the first chunk, send Note On with pitch and CCs
            if first_chunk:
                midi_synth.send_pitch(predicted_pitch)
                first_chunk = False
                last_pitch = predicted_pitch
            # Send only CC messages for subsequent chunks
            midi_synth.send_ccs(predicted_ccs)

            # Delay between chunks
            time.sleep(delay)

        except Exception as e:
            print(f"Error processing chunk {i+1}/{num_chunks}: {e}")

    # Send Note Off at the end of processing
    if last_pitch is not None:
        midi_synth.send_note_off(last_pitch)

if __name__ == "__main__":
    main()
