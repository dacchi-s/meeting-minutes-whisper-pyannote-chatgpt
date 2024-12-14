# AI-Powered Meeting Minutes Generator

An automated tool that generates meeting minutes from audio/video recordings using AI technologies: Whisper (speech recognition), Pyannote (speaker identification), and ChatGPT (text generation).

## Description

This project automates the tedious task of creating meeting minutes by leveraging state-of-the-art AI technologies. It combines three powerful AI models to deliver accurate and comprehensive meeting documentation:

- **Whisper**: OpenAI's robust speech recognition model that accurately transcribes audio in multiple languages
- **Pyannote**: Advanced speaker diarization system that identifies and distinguishes between different speakers
- **ChatGPT**: Natural language processing model that generates concise and well-structured meeting minutes

The tool supports various audio and video formats and can handle meetings of different sizes and complexities. It's particularly useful for:
- Regular team meetings and standups
- Academic conferences and lectures
- Business presentations and workshops
- Interview transcriptions
- Any scenario where accurate meeting documentation is required

## Features

- Supports multiple audio/video formats (WAV, MP3, MP4, WMA)
- Automatic speech recognition with OpenAI's Whisper
- Speaker identification and diarization using Pyannote
- Intelligent meeting minutes generation with ChatGPT
- GPU acceleration support (CUDA)
- Configurable speaker detection

## Prerequisites

- Ubuntu (tested on 24.04 WSL)
- NVIDIA GPU with CUDA support (tested on RTX 3070Ti)
- Docker
- OpenAI API Key
- Hugging Face Access Token

## Installation

### 1. Docker Setup

First, remove any old Docker installations:
```bash
for pkg in docker.io docker-doc docker-compose docker-compose-v2 podman-docker containerd runc; do sudo apt remove $pkg; done
```

Install Docker Engine:
```bash
sudo apt update
sudo apt install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

Verify installation:
```bash
sudo docker run hello-world
docker -v
```

### 2. Configure Docker

Enable Docker command without sudo:
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
```

Set Docker to start automatically:
```bash
sudo visudo
# Add the following line at the end:
docker ALL=(ALL)  NOPASSWD: /usr/sbin/service docker start

sudo nano ~/.bashrc
# Add the following at the end:
if [[ $(service docker status | awk '{print $4}') = "not" ]]; then
    sudo service docker start > /dev/null
fi
```

### 3. NVIDIA Docker Setup

Install NVIDIA Docker support:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 4. Create and Configure Docker Container

```bash
docker pull nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
docker run -it --gpus all nvcr.io/nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Inside container:
apt update && apt full-upgrade -y
apt install git wget nano ffmpeg -y
```

### 5. Install Miniconda

```bash
cd ~
mkdir tmp
cd tmp
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts: yes, enter, yes
cd ..
rm -rf tmp
```

### 6. Set Up Environment

Create a new directory and environment file:
```bash
mkdir speech_to_text
cd speech_to_text
nano environment_speech_to_text.yml
```

Add the following content:
```yaml
name: speech-to-text
channels:
  - conda-forge
  - pytorch
  - nvidia
  - defaults
dependencies:
  - python=3.9
  - pip
  - cudatoolkit=11.8
  - tiktoken
  - pip:
    - openai-whisper
    - pyannote.audio
    - pydub==0.25.1
    - torch
    - torchvision
    - torchaudio
    - moviepy
    - openai
    - python-dotenv
```

Create and activate the environment:
```bash
conda env create -f environment_speech_to_text.yml
conda activate speech-to-text
```

### 7. Configuration

Create a `.env` file:
```bash
nano .env
```

Add your API keys:
```
OPENAI_API_KEY=sk-XXXXXXXXXXXXXXXXXXXXXX
HF_TOKEN=hf_ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ
```

## Usage

### Basic Usage

```bash
python speech_to_text.py [audio_file] [options]
```

### Options

```
--model {tiny,base,small,medium,large,large-v2,large-v3}
                      Whisper model to use
--num_speakers NUM_SPEAKERS
                      Number of speakers (if known)
--min_speakers MIN_SPEAKERS
                      Minimum number of speakers
--max_speakers MAX_SPEAKERS
                      Maximum number of speakers
```

### Example

```bash
python speech_to_text.py meeting.mp4 --model large-v3 --min_speakers 1 --max_speakers 2
```

### File Transfer

To copy files between Windows and Docker container:

```bash
# Windows to Docker
docker cp "/mnt/c/Windows_path/audio.wav" container_name:root/speech_to_text

# Docker to Windows
docker cp container_name:root/speech_to_text/minutes.txt "/mnt/c/Windows_path/"
```

## Output

The script generates two output files:
1. `*_output.txt`: Contains the full transcription with speaker identification
2. `*_minutes.txt`: Contains the AI-generated meeting minutes

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
