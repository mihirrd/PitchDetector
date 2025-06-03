# Real-Time Pitch Detection

This project implements real-time pitch detection using FFT (Fast Fourier Transform) analysis to identify the fundamental frequency from an audio input and map it to the corresponding musical scale.

The original analysis and concept can be explored in the `Pitch.ipynb` Jupyter notebook, which uses `.mp3` files for demonstration. This `README` focuses on the `realtime_pitch.py` script for live audio input.

## Setup

1.  **Install Python Dependencies:**
    Make sure you have Python 3 installed. Then, install the necessary libraries using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

2.  **System-Level Dependencies (Linux for PyAudio):**
    PyAudio can have system-level dependencies. On Linux, you might need to install packages like `portaudio19-dev` and `python3-dev` (or the equivalent for your specific Python version) *before* running `pip install`.
    For example, on Debian/Ubuntu-based systems:
    ```bash
    sudo apt-get update && sudo apt-get install portaudio19-dev python3-dev
    ```
    For other operating systems, please refer to the PyAudio documentation for installation prerequisites.

## How to Run

1.  **Connect a Microphone:**
    Ensure you have a microphone connected to your computer and that it's selected as the default input device.

2.  **Run the Script:**
    Execute the real-time pitch detection script from your terminal:
    ```bash
    python realtime_pitch.py
    ```

3.  **Observe Output:**
    The script will start listening to your microphone. As it detects audio, it will print the detected musical pitch (e.g., "C", "A#") and the corresponding fundamental frequency (e.g., "Detected pitch: A (440.12 Hz)") to the console. The output updates in real-time.

    To stop the script, press `Ctrl+C` in the terminal.

## Additional Notes
- The `Sounds/` directory contains various `.mp3` files with different musical scales, which were used for testing and development in the `Pitch.ipynb` notebook.
- The core pitch detection in `realtime_pitch.py` uses a NumPy-based FFT approach. For more advanced or robust pitch detection, libraries like Librosa offer algorithms such as YIN or piptrack.
