import pyaudio
import time
import numpy as np
import librosa # Not strictly used for pitch detection here, but often used with it.

# Define constants for audio processing
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024  # Power of 2 for FFT

# Scales Dictionary
Scales = {
    "C": 261.63, "C#": 277.18, "D": 293.66, "D#": 311.13, "E": 329.63, "F": 349.23,
    "F#": 369.99, "G": 392.00, "G#": 415.30, "A": 440.00, "A#": 466.16, "B": 493.88,
    "C1": 523.25, "C#1": 554.37, "D1": 587.33, "D#1": 622.25, "E1": 659.25, "F1": 698.46,
    "F#1": 739.99, "G1": 783.99, "G#1": 830.61, "A1": 880.00, "A#1": 932.33, "B1": 987.77,
    "C2": 1046.50
}
out_dict = {value: key for key, value in Scales.items()}

def detect_pitch(data_chunk, current_rate):
    try:
        audio_np = np.frombuffer(data_chunk, dtype=np.int16) / 32768.0
        if not np.any(audio_np): # Check for silence or empty buffer
            return None, None

        magnitude = np.abs(np.fft.fft(audio_np))
        frequency_axis = np.fft.fftfreq(len(magnitude), 1.0 / current_rate) # More accurate frequency axis

        # Filter out negative frequencies and DC offset
        positive_freq_indices = np.where(frequency_axis > 0)[0]
        if len(positive_freq_indices) == 0:
            return None, None

        left_magnitude = magnitude[positive_freq_indices]
        left_frequency = frequency_axis[positive_freq_indices]

        if len(left_magnitude) == 0:
            return None, None

        peak_index = np.argmax(left_magnitude)
        fundamental_frequency = left_frequency[peak_index]

        if fundamental_frequency <= 0: # Should be positive due to filtering
            return None, None

        diff = []
        scale_frequencies = list(Scales.values())
        for note_freq in scale_frequencies:
            diff.append(abs(note_freq - fundamental_frequency))

        if not diff:
            return None, fundamental_frequency

        min_diff_index = np.argmin(diff)
        closest_scale_freq = scale_frequencies[min_diff_index]
        detected_note = out_dict.get(closest_scale_freq)

        return detected_note, fundamental_frequency

    except Exception as e:
        # Print error within the function for debugging specific pitch detection issues
        # but allow main loop to handle broader error reporting to console
        # print(f"Error in detect_pitch: {e}")
        return None, None


def record_and_detect_pitch():
    audio = pyaudio.PyAudio()
    stream = None

    print("Initializing audio...")
    try:
        # Check for available devices (optional, but good for debugging)
        # device_count = audio.get_device_count()
        # if device_count == 0:
        #     print("No audio devices found. Please ensure a microphone is connected.")
        #     return
        # print(f"Found {device_count} audio devices.")
        # for i in range(device_count):
        #     print(audio.get_device_info_by_index(i))


        # Attempt to open the stream
        try:
            stream = audio.open(format=FORMAT,
                                channels=CHANNELS,
                                rate=RATE,
                                input=True,
                                frames_per_buffer=CHUNK,
                                input_device_index=None) # Use default input device
        except IOError as e:
            # This IOError can occur if no input device is found or if there's an issue with portaudio
            print(f"Error opening audio stream: {e}")
            print("Please ensure a microphone is connected and configured correctly.")
            print("You may need to install 'portaudio19-dev' (e.g., sudo apt-get install portaudio19-dev).")
            return # Exit if stream cannot be opened

        print("Recording and pitch detection started. Press Ctrl+C to stop.")
        print("-" * 40)

        while True: # Main loop for reading and processing audio
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
                detected_note, fundamental_frequency = detect_pitch(data, RATE)

                if detected_note and fundamental_frequency is not None:
                    print(f"\rDetected pitch: {detected_note} ({fundamental_frequency:.2f} Hz)        ", end='')
                elif fundamental_frequency is not None:
                    print(f"\rDetected frequency: {fundamental_frequency:.2f} Hz (No note)        ", end='')
                else:
                    print("\rReading audio... No clear pitch detected.          ", end='')

            except IOError as e:
                # Handle stream read errors (like overflow)
                if hasattr(e, 'errno') and e.errno == pyaudio.paInputOverflowed:
                    print("\rInput overflowed. Skipping this chunk.            ", end='')
                # Handle cases where stream might be closed or invalid
                elif hasattr(e, 'errno') and e.errno == pyaudio.paBadStreamPtr: # Example, might vary
                    print("\rAudio stream error. Attempting to recover...      ", end='')
                    # Potentially try to re-open stream here, or simply exit
                    time.sleep(0.5) # Brief pause
                else:
                    # Other IOErrors that are not overflow
                    print(f"\rIOError during stream read: {e}                  ", end='')
            except Exception as e:
                # Catch any other unexpected errors in the loop
                print(f"\nAn unexpected error occurred in the processing loop: {e}")
                # Depending on severity, you might want to break or continue
                break

    except KeyboardInterrupt:
        print("\n" + "-" * 40)
        print("Recording stopped by user.")
    except Exception as e:
        # Catch all other exceptions (e.g., during PyAudio init)
        print(f"\nAn critical error occurred: {e}")
    finally:
        print("\n" + "-" * 40)
        print("Cleaning up resources...")
        if stream is not None:
            print("Stopping stream...")
            try:
                if stream.is_active(): # Check if stream is active before stopping
                    stream.stop_stream()
                stream.close()
                print("Stream closed.")
            except Exception as e:
                print(f"Error closing stream: {e}")

        if audio is not None:
            print("Terminating PyAudio...")
            audio.terminate()
            print("PyAudio terminated.")
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    record_and_detect_pitch()
