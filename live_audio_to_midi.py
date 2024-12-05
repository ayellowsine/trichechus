import os
import numpy as np
import tensorflow as tf
import spiegelib as spgl
import mido
import sounddevice as sd
import time
import pickle
from tensorflow.keras.layers import Layer
import threading

# Custom HighwayLayer Implementation
class HighwayLayer(Layer):
    def __init__(self, units=None, activation='relu', transform_gate_bias=-2.0, **kwargs):
        super(HighwayLayer, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)
        self.transform_gate_bias = transform_gate_bias

    def build(self, input_shape):
        if self.units is None:
            self.units = input_shape[-1]

        self.dense_H = tf.keras.layers.Dense(self.units, activation=self.activation)
        self.dense_T = tf.keras.layers.Dense(self.units, activation='sigmoid', bias_initializer=tf.keras.initializers.Constant(self.transform_gate_bias))

    def call(self, inputs):
        H = self.dense_H(inputs)
        T = self.dense_T(inputs)
        C = 1.0 - T
        return H * T + inputs * C

    def get_config(self):
        config = super(HighwayLayer, self).get_config()
        config.update({
            'units': self.units,
            'activation': tf.keras.activations.serialize(self.activation),
            'transform_gate_bias': self.transform_gate_bias
        })
        return config

    @classmethod
    def from_config(cls, config):
        units = config.pop('units', None)
        return cls(units=units, **config)

# Define RMS Error loss function
def rms_error(y_true, y_pred):
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

# Load Model and Scaler
def load_model_and_scaler():
    try:
        print("Loading model and scaler...")
        custom_objects = {
            "rms_error": rms_error,
            "HighwayLayer": HighwayLayer,
        }
        model = tf.keras.models.load_model("saved_models/simple_fm_bi_lstm.h5", custom_objects=custom_objects)
        with open('params_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        print("Model and scaler loaded successfully.")
        return model, scaler
    except Exception as e:
        print(f"Error loading model or scaler: {e}")
        raise RuntimeError(f"Cannot proceed without the model or scaler: {e}")

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

    def closest_valid_value(self, value, valid_values):
        return min(valid_values, key=lambda x: abs(x - value))

    def send_midi(self, cc_values):
        # Clamp values to valid MIDI range (0–127)
        cc_values = [max(0, min(127, int(round(value)))) for value in cc_values]

        # Map CC 16 to the closest valid value
        cc16_values = [0, 8, 15, 23, 30, 38, 45, 53, 60, 68, 75, 83, 90, 98, 105, 113, 120]  # Custom values for CC16 (Gen Spectrum)

        cc_values[0] = self.closest_valid_value(cc_values[0], cc16_values)  # CC 16

        # Send CC values (CC16, CC18, CC21, CC83)
        cc_controllers = [16, 18, 21, 83]
        for cc, value in zip(cc_controllers, cc_values):
            self.midi_port.send(mido.Message('control_change', control=cc, value=value))

        # Return formatted string for display
        return f"CC16: {cc_values[0]} CC18: {cc_values[1]} CC21: {cc_values[2]} CC83: {cc_values[3]}"

    def close_port(self):
        if hasattr(self, 'midi_port') and self.midi_port:
            self.midi_port.close()
            print(f"MIDI port '{self.midi_port_name}' closed.")

# VU Meter Function
def vu_meter(indata, midi_display, display_mode):
    """Displays a primitive VU meter based on audio input amplitude and MIDI output values."""
    rms = np.sqrt(np.mean(indata**2))
    meter_length = 50
    level = int(rms * meter_length)
    filled = "█" * level
    empty = "▒" * (meter_length - level)
    vu_bar = f"|{filled}{empty}|"

    if display_mode == "s":
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"{vu_bar} {midi_display}", end="\r", flush=True)
    elif display_mode == "w":
        print(f"{vu_bar} {midi_display}")

# Audio Processing Function
def process_audio_chunk(audio_chunk, midi_synth, model, scaler, display_mode, gain_boost, cc_multiplication_factors):
    try:
        audio_chunk = audio_chunk * gain_boost
        audio_chunk = np.pad(audio_chunk, (0, max(0, 22 * 2048 - len(audio_chunk))))
        input_features = spgl.features.MFCC(num_mfccs=13, frame_size=2048, hop_size=1024, time_major=True)
        audio_buffer = spgl.AudioBuffer(audio_chunk, sample_rate=44100)
        features = input_features(audio_buffer)
        features = features[:22, :].reshape(1, 22, 13)

        normalized_prediction = model.predict(features, verbose=0)[0]
        predicted_patch = scaler.inverse_transform([normalized_prediction])[0]

        predicted_ccs = [predicted_patch[2], predicted_patch[4], predicted_patch[5], predicted_patch[6]]
        predicted_ccs = [value * factor for value, factor in zip(predicted_ccs, cc_multiplication_factors)]
        midi_display = midi_synth.send_midi(predicted_ccs)
        vu_meter(audio_chunk, midi_display, display_mode)

    except Exception as e:
        print("Error processing audio:", e)

# Main function
def main():
    midi_synth = None
    try:
        print("Available input devices:")
        for i, device in enumerate(sd.query_devices()):
            if device['max_input_channels'] > 0:
                print(f"{i}: {device['name']}")
        device_id = int(input("Select input device by index: "))

        mode = input("Select mode: (f) fixed delay or (t) threshold: ").lower()
        if mode == "f":
            delay = float(input("Enter delay between MIDI CC messages in seconds (e.g., 0.1): "))
        elif mode == "t":
            threshold_db = float(input("Enter threshold in dB for triggering MIDI messages (e.g., -40): "))
        else:
            print("Invalid mode selected. Defaulting to threshold mode.")
            mode = "t"
            threshold_db = float(input("Enter threshold in dB for triggering MIDI messages (e.g., -40): "))

        gain_boost = float(input("Enter gain boost factor (e.g., 1.0 for no boost, 2.0 to double): "))

        cc_multiplication_factors = []
        cc_descriptions = [
            "CC16 - Generator Spectrum",
            "CC18 - Generator Inversion",
            "CC21 - Sub-Oscillator Harmonic Content",
            "CC83 - Noise Level"
        ]
        for cc_desc in cc_descriptions:
            cc_factor = float(input(f"Enter multiplication factor for {cc_desc} (e.g., 1.0 for no scaling): "))
            cc_multiplication_factors.append(cc_factor)

        display_mode = input("Select display mode: (s) standard or (w) waterfall: ").lower()

        midi_synth = HardwareSynth()
        model, scaler = load_model_and_scaler()
        sample_rate = 44100
        chunk_size = int(0.05 * sample_rate)

        latest_audio_chunk = [None]

        def audio_callback(indata, frames, time, status):
            if status and "overflow" not in str(status):
                print(f"Audio Input Error: {status}", flush=True)
            latest_audio_chunk[0] = indata.flatten()

        with sd.InputStream(device=device_id, channels=1, samplerate=sample_rate, blocksize=chunk_size, callback=audio_callback):
            print("Processing live audio input. Press Ctrl+C to stop.")
            while True:
                try:
                    if latest_audio_chunk[0] is not None:
                        if mode == "f":
                            process_audio_chunk(latest_audio_chunk[0], midi_synth, model, scaler, display_mode, gain_boost, cc_multiplication_factors)
                            time.sleep(delay)
                        elif mode == "t":
                            rms = np.sqrt(np.mean(latest_audio_chunk[0]**2))
                            rms_db = 20 * np.log10(rms + 1e-10)
                            if rms_db > threshold_db:
                                process_audio_chunk(latest_audio_chunk[0], midi_synth, model, scaler, display_mode, gain_boost, cc_multiplication_factors)
                                time.sleep(0.05)
                    time.sleep(0.001)

                except KeyboardInterrupt:
                    print("\nStopping audio processing.")
                    break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if midi_synth:
            midi_synth.close_port()

if __name__ == "__main__":
    main()
