import time
import wave
import pyaudio
import requests
import numpy as np
from openwakeword.model import Model
from openwakeword.utils import download_models
import argparse
import subprocess
import os
from chatbots import create_chatbot
import re
import datetime

# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default='tflite',
    required=False
)
# make vad_threshold a commandline arg with default value
parser.add_argument(
    "--vad_threshold",
    help="Voice activity detection threshold",
    type=float,
    default=0.5,
    required=False
)
parser.add_argument(
    "--eliza_path",
    help="The path to eliza to load",
    type=str,
    default="../eliza",
    required=False
)
parser.add_argument(
    "--chatbot_type",
    help="define chatbot type",
    type=str,
    default="",
    required=False
)

args=parser.parse_args()

download_models()

# Get microphone stream
ASR_URL = "http://127.0.0.1:8080/inference"
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = args.chunk_size
audio = pyaudio.PyAudio()
mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models
if args.model_path != "":
    owwModel = Model(wakeword_models=[args.model_path], inference_framework=args.inference_framework, vad_threshold=args.vad_threshold)
else:
    owwModel = Model(inference_framework=args.inference_framework, vad_threshold=args.vad_threshold)
MAX_RECORDING_DURATION = 10
MIN_RECORDING_DURATION = 1.5

n_models = len(owwModel.models.keys())

# Convert transcribed text to audio
PIPER_PATH = '../piper/'
VOICE = 'en_US-lessac-medium.onnx'
def speak(text):
    ps = subprocess.Popen(('echo', text), stdout=subprocess.PIPE)
    ps2 = subprocess.Popen((f'{PIPER_PATH}piper', '--model', os.path.join(PIPER_PATH, 'voices', VOICE), '--output-raw'), stdin=ps.stdout, stdout=subprocess.PIPE)
    subprocess.check_output(('aplay', '-r', '22050', '-f', 'S16_LE', '-t', 'raw', '-'), stdin=ps2.stdout)

def get_time():
    return f'it is {datetime.datetime.now().strftime("%H:%M")}'

skills = {
    'get_time': get_time
}

# Run capture loop continuosly, checking for wakewords
if __name__ == "__main__":
    chatbot = create_chatbot(skills, args)
    if chatbot.init_speech:
        speak(chatbot.init_speech)
    print("Listening for wakewords...")
    try:
        frames = []
        detection_ts = None
        while True:
            # Get audio
            audio_np = np.frombuffer(mic_stream.read(CHUNK), dtype=np.int16)

            # Feed to openWakeWord model
            prediction = owwModel.predict(audio_np, threshold={args.model_path: 0.5}, debounce_time=1)

            if prediction[args.model_path] > 0.5 and detection_ts is None:
                print('wakeword detected')
                subprocess.run(["aplay", "./start_listening.wav"])
                detection_ts = time.perf_counter()
            if detection_ts is not None:
                vad_frames = list(owwModel.vad.prediction_buffer)[-20:]
                vad_max_score = np.max(vad_frames) if len(vad_frames) > 0 else 0
                if time.perf_counter() - detection_ts < MAX_RECORDING_DURATION and (vad_max_score > args.vad_threshold or time.perf_counter() - detection_ts < MIN_RECORDING_DURATION):
                    print('appending to audio')
                    frames.append(audio_np)
                else: 
                    mic_stream.stop_stream()
                    print('writing wav file')
                    with wave.open('./testrecording.wav', 'wb') as waveFile:
                        waveFile.setnchannels(CHANNELS)
                        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
                        waveFile.setframerate(RATE)
                        waveFile.writeframes(b''.join(frames))
                    detection_ts = None
                    frames = []
                    print('transcribing audio')
                    transcript = ''
                    with open('./testrecording.wav', 'rb') as fobj:
                        r = requests.post(ASR_URL, files={'file': fobj})
                        transcript = r.json()['text']
                        transcript = re.sub('\[.*\]', '', transcript).strip()
                        # make a response based on the transcribed user input and speak it
                        print('transcript:', transcript)
                        response = chatbot.respond(transcript)
                        if response is None:
                            speak(chatbot.final())
                            break
                        speak(response)
                        mic_stream.start_stream()
    except KeyboardInterrupt:
        pass

    print('Shutting down')
    mic_stream.close()
    audio.terminate()
