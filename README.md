# Speech Interaction

This program allows you to evaluate different speech-to-text (ASR) models on various speech corpora, measuring both Word Error Rate (WER) and inference latency. It can also visualize the trade-off between accuracy and speed for different ASR models.

---

## Getting Started

### 1. Clone the Repositories

The main repository of this project contains the evaluation scripts. You also need to download the Whisper.cpp repository, which provides the server and tools to run the Whisper models for speech-to-text transcription.
:

```
# Clone this repository
git clone https://gitlab.mi.hdm-stuttgart.de/jo041/speech-interaction.git
cd <your-project-folder>

# Clone Whisper repository
git clone https://github.com/openai/whisper.git
cd whisper

# download a model
sh ./models/download-ggml-model.sh <modelname>

# install cmake
sudo apt-get install cmake

# build the project
cmake -B build
cmake --build build -j --config Release
```

### 2. Set Up a Virtual Environment (Recommended)

It is recommended to create a Python virtual environment to keep dependencies isolated:

```
# Create a virtual environment in the folder 'venv'
python -m venv venv

# Activate the virtual environment
# On Linux / macOS:
source venv/bin/activate

# On Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# Upgrade pip inside the venv
pip install --upgrade pip
```

Once activated, all Python packages will be installed in this environment without affecting your system Python.

### 3. Install Dependencies

With the virtual environment activated, install the required Python packages using pip:

```
cd /path/to/project
pip install -r requirements.txt
```

### 4. Start the Whisper Server

Run the Whisper server pointing to your chosen model:

```
./whisper.cpp/build/bin/whisper-server --model ./whisper.cpp/models/<model_name>
```

### 5. Run the Benchmark

The main script evaluates the ASR model on a subset of your chosen corpus:

```
python eval_stt.py --speech_corpus <dataset_name> --dataset_samples <number_of_samples> --model <model_name> --filename <csv_filename>
```

Parameters:

- --speech_corpus – choose which dataset to test (librispeech, fleurs, peoples_speech, gigaspeech)
- --dataset_samples – number of audio samples to evaluate
- --model – model identifier (used for logging, e.g. tiny, base, large)
- --filename - name of the resulting csv file

### 6. Visualizing Results

After evaluation, you can generate a scatter plot of WER vs. real-time factor:

```
python plot_stt.py --filepath <path_to_csv>
```

This helps to quickly compare model accuracy and speed for real-time applications.

## Authors and acknowledgment

Adib Shaqaiq, Jonas Öhler

## License

Copyright © 2026 MI <br>

This program and the accompanying materials are made available under the
terms of the Eclipse Public License 2.0 which is available at
http://www.eclipse.org/legal/epl-2.0.
