# VoiceSecure

**The first microphone module designed to prevent automated monitoring of speech while preserving its natural intelligibility for human listeners.**

---

## Paper Title  
**For Human Ears Only: Preventing Automated Monitoring on Voice Data**  
*Usenix Security 2025*

---

## Overview

This artifact accompanies our paper and enables full reproducibility of its core results. It includes:

- Code for applying VoiceSecure modifications  
- Pre-trained models  
- Datasets  
- Evaluation scripts  
- Plotting tools  

**VoiceSecure** is a real-time speech transformation method that obfuscates speaker identity and content while maintaining intelligibility for humans. This repository allows you to:

- Apply VoiceSecure transformations  
- Reproduce speaker recognition and ASR experiments  
- Compute metrics: WER, STOI, and speaker mismatch rate (MMR)  
- Generate all paper figures and tables  
- Train the VoiceSecure model from scratch on your own dataset  

---

## Environmental Setup

### 📦 Download Artifact Bundle

[https://doi.org/10.5281/zenodo.15603263](https://doi.org/10.5281/zenodo.15603263)

After extracting the archives into a directory (e.g., `VoiceSecure_Artifacts/`), your structure should look like:

```
VoiceSecure_Artifacts/
├── Data2/
├── ScriptForApplyingVoiceSecure/
├── ScriptForTrainingModel/
├── ScriptsForCompiledResults/
├── ScriptForComputingMetrics/
├── ScriptForDataSetCreation/
├── Trained_Model/
└── requirements.txt
```

---

## Requirements

- **Python 3.8+**
- **MATLAB** with toolboxes:
  - Audio Toolbox  
  - Signal Processing Toolbox  
  - 5G Toolbox  
- **No GPU required**

---

## Installation

```bash
# Install miniconda
bash Miniconda3-latest-Linux-x86_64.sh

# Activate conda
source ~/miniconda3/bin/activate

# Create and activate environment
conda create --name py38 python=3.8.18
conda activate py38

# Install Python requirements
pip install -r requirements.txt
```

### To run DeepSpeech-based evaluations:

1. Clone: [https://github.com/SeanNaren/deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch)
2. Navigate to the repo and run:

```bash
pip install -r requirements.txt
```

3. Ensure this repo is added to your environment path.

---

## Directory Descriptions

### `Data2/`

- `CommonVoice/`, `LibriSpeech_Dev/`, `VCTK/`, `VoxCeleb_Dev/`:  
  Speaker embeddings, WER scores, and audio (excluding VoxCeleb originals).
- `Benign/`: Results for unmodified speech.
- `Latency_Results/`: Latency profiling results.
- `Noise/`: Noise WAV files used in training/testing.
- `TrainingSet/`: Training data.
- `UserStudy/`: Perceptual evaluation responses.

> ⚠️ Original VoxCeleb audio is excluded due to license restrictions.

---

### `Trained_Model/`

- `Trained_VoiceSecure_Model.pth` – pre-trained model  
- Pre-trained speaker models:  
  `Ivector_Pretrained_model`, `spkrec-xvect-voxceleb`, `spkrec-ecapa-voxceleb`

---

### `ScriptForApplyingVoiceSecure/`

- `Apply_VoiceSecure.py`: Apply transformations  
- `ComputeSpeakerEmbeddings.py`, `Script2ComputeIVectors.py`: Speaker embeddings  
- `ComputeWER.py`: Evaluate ASR models (Whisper, DeepSpeech, Wav2Vec2)

---

### `ScriptForTrainingModel/`

- `Training.py`: Train VoiceSecure from scratch

---

### `ScriptForComputingMetrics/`

- `Script2ComputePerceptualScore.py`: STOI scores  
- `Script2PlotASR_Results.py`: WER plots  
- `Script2PlotSpeaker_Results_MMR.py`: MMR plots  
- `Script2PlotStoi_Results.py`: STOI plots  

---

### `ScriptsForCompiledResults/`

- MATLAB scripts to reproduce paper figures/tables using data from `Data2/`

---

### `ScriptForDataSetCreation/`

- Helpers for file conversion and dataset preparation

---

## Workflow Summary

1. **Apply Modifications** – Process clean audio with VoiceSecure  
2. **Compute Metrics** – Run WER, MMR, and STOI evaluations  
3. **Plot Results** – Generate visuals using compiled data  

We provide pre-computed results for:

- **LibriSpeech**
- **VoxCeleb**
- **CommonVoice**
- **VCTK**

Including:

- Speaker embeddings: x-vector, ECAPA, i-vector  
- WER scores: Whisper, DeepSpeech, Wav2Vec2  
- Modified and original audio (except VoxCeleb)

---

## Notes

- Scripts are pre-configured for **LibriSpeech**.
- Modify paths and settings (lines marked with `# adjust according to the data`) to use other datasets.

---

## Dataset Links

- [LibriSpeech](https://www.openslr.org/12)  
- [VoxCeleb](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)  
- [CommonVoice](https://commonvoice.mozilla.org/en)  
- [VCTK](https://datashare.ed.ac.uk/handle/10283/2950)

---
