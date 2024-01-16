import gradio as gr
from transformers import pipeline
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import time
import webrtcvad


def with_gradio():
    def transcribe(audio):
        sr, y = audio
        print(sr, type(sr), y, type(y), y.shape)
        y = y.astype(np.float32)
        y /= np.max(np.abs(y))

        print(y.shape)
        return asr({"sampling_rate": sr, "raw": y})["text"]

    demo = gr.Interface(
        transcribe,
        gr.Audio(sources=["microphone"]),
        "text",
    )

    demo.launch()


def segments():
    duration = 5
    countdown = 3
    print("Start speaking after the recording begins.")

    for i in range(countdown, 0, -1):
        print(f"Recording will start in {i} seconds...")
        time.sleep(1)

    print('Speak!')
    # Start the recording
    myrecording = sd.rec(int(duration * 44100), samplerate=44100, channels=1)
    sd.wait()

    # Convert the recording to mono, float32 and normalize it
    myrecording = myrecording[:, 0]
    myrecording = myrecording.astype(np.float32)
    myrecording /= np.max(np.abs(myrecording))

    # Display the audio waveform
    plt.figure(figsize=(10, 4))
    plt.plot(myrecording)
    plt.title('Audio Waveform')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

    # Perform ASR on the recording
    result = asr({"sampling_rate": 44100, "raw": myrecording})
    print(result['text'])


def real_time():
    q = []
    sampling_rate = 48000

    # Define the callback function to be called when new audio data is available
    def callback(indata, frames, time, status):
        q.append(indata)

    # Define the size of the window for the sliding window approach
    window_size = sampling_rate * 5  # 5 seconds

    # Initialize audio_data as an empty array
    audio_data = np.array([])
    vad = webrtcvad.Vad(2)
    # Start the audio stream
    with sd.InputStream(callback=callback):
        while True:
            # Get the next chunk of audio data
            if not q:
                continue
            audio_chunk = q.pop(0)[:, 0]
            audio_chunk = np.ascontiguousarray(audio_chunk)

            # Append the new audio data to the existing audio data
            # if vad.is_speech(buf=audio_chunk, sample_rate=sampling_rate):
            audio_data = np.concatenate((audio_data, audio_chunk), axis=0)
            # If the window is full, perform ASR on the window of audio data
            if len(audio_data) >= window_size:
                # Extract the window of audio data
                window_of_audio_data = audio_data[:window_size]

                # Perform ASR on the window of audio data
                result = asr({"sampling_rate": sampling_rate, "raw": window_of_audio_data})
                print(result['text'])

                # Slide the window along the incoming audio data
                audio_data = audio_data[len(window_of_audio_data):]
            # else:
            #     print('No audio is being detected')


if __name__ == '__main__':
    asr = pipeline("automatic-speech-recognition")
    # with_gradio()
    # segments()
    real_time()