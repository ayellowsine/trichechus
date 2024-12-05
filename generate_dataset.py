import mido
import time
import sounddevice as sd
from scipy.io.wavfile import write

def send_midi_panic_message(out_port):
    """Send a MIDI panic message to stop any hanging notes."""
    for channel in range(16):
        out_port.send(mido.Message('control_change', channel=channel, control=123, value=0))  # All Notes Off
        out_port.send(mido.Message('control_change', channel=channel, control=64, value=0))   # Sustain Pedal Off
    print("MIDI panic message sent (All Notes Off and Sustain Off).")

def send_midi_messages(out_port, pitch, cc_values):
    """Send MIDI CC values first, then the pitch to ensure CC settings are applied."""
    # Send CC messages for each controller with the specified values
    cc_controllers = [70, 16, 23, 18, 21, 83]  # CC controllers for 6 parameters
    for i, cc_value in enumerate(cc_values):
        out_port.send(mido.Message('control_change', control=cc_controllers[i], value=cc_value))
        print(f"Control Change: CC {cc_controllers[i]}, Value {cc_value}")

    # Send Note On after CC values
    out_port.send(mido.Message('note_on', note=pitch, velocity=127))
    print(f"Note On: pitch {pitch}")

def record_audio(filename: str, duration: int = 0.5, sample_rate: int = 44100):
    """Record audio and save as a WAV file."""
    print(f"Recording for {duration} seconds...")
    sd.default.device = 1
    # Capture audio data
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()  # Wait until recording is finished

    # Save as WAV file
    write(filename, sample_rate, audio_data)
    print(f"Recording saved as {filename}")

def generate_filename_from_pitch_and_cc(pitch, cc_values):
    """Generate a filename based on the pitch and CC values in hexadecimal format."""
    pitch_hex = f"{pitch:02X}"  # Convert pitch to 2-digit hex
    cc_hex = ''.join(f"{value:02X}" for value in cc_values)  # Convert each CC value to 2-digit hex
    return f"{pitch_hex}{cc_hex}.wav"

def cycle_midi_parameters(out_port, cc4_offset=0, cc5_offset=0, cc6_offset=0):
    """Cycle through pitch and CC values, send panic messages, and record audio with encoded filenames."""
    pitch_range = range(60, 61)  # Updated pitch values from 36 to 71
    
    # Define custom values for the first, second, and third CC parameters
    custom_cc1_values = [26, 51, 77, 102, 127]  # Custom values for the first CC parameter (Gen Waveform)
    custom_cc2_values = [0, 8, 15, 23, 30, 38, 45, 53, 60, 68, 75, 83, 90, 98, 105, 113, 120]  # Updated custom values for the second CC parameter (Gen Spectrum)
    custom_cc3_values = [0, 22, 43, 64, 85, 106, 127]  # Custom values for the third CC parameter (Sub FM Rate)
    
    # Define step skipping and optional offsets for the fourth, fifth, and sixth CC parameters
    skip_step_cc4 = 63  # Example: skip every Nth step for the fourth CC parameter (Gen Inversion)
    skip_step_cc5 = 63   # Example: skip every Nth step for the fifth CC parameter (Sub Content)
    skip_step_cc6 = 63  # Example: skip every Nth step for the sixth CC parameter (Noise Level)
    cc4_offset = 0
    cc5_offset = 64
    cc6_offset = 0

    # Combine CC parameters into ranges, applying offsets where provided
    cc_ranges = [
        custom_cc1_values,  # First CC parameter uses custom values
        custom_cc2_values,  # Second CC parameter uses custom values
        custom_cc3_values,  # Third CC parameter with custom values
        range(cc4_offset, 128, skip_step_cc4),  # Fourth CC with step skipping and offset
        range(cc5_offset, 128, skip_step_cc5),  # Fifth CC with step skipping and offset
        range(cc6_offset, 128, skip_step_cc6)   # Sixth CC range with skipping and offset
    ]
    
    for pitch in pitch_range:
        for cc1 in cc_ranges[0]:
            for cc2 in cc_ranges[1]:
                for cc3 in cc_ranges[2]:
                    for cc4 in cc_ranges[3]:
                        for cc5 in cc_ranges[4]:
                            for cc6 in cc_ranges[5]:
                                cc_values = [cc1, cc2, cc3, cc4, cc5, cc6]
                                
                                # Generate filename based on pitch and CC values
                                filename = generate_filename_from_pitch_and_cc(pitch, cc_values)
                                
                                # Send the MIDI messages
                                send_midi_messages(out_port, pitch, cc_values)
                                
                                # Record audio with the generated filename
                                record_audio(filename, duration=0.5, sample_rate=44100)
                                
                                # Send a MIDI panic message after each note cycle
                                send_midi_panic_message(out_port)
                                
                                # Add a short delay if needed
                                time.sleep(0.5)

if __name__ == "__main__":
    # Open an output port
    out_port = mido.open_output('USB Midi Cable 1')  # Replace with your MIDI device name
    try:
        # Run cycle with optional offsets for CC4, CC5, and CC6
        cycle_midi_parameters(out_port, cc4_offset=0, cc5_offset=32, cc6_offset=64)
    except KeyboardInterrupt:
        print("Stopping the MIDI message cycling.")
    finally:
        out_port.close()
