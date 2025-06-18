import os
import sys
import time
import numpy as np
import glob as glob

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim

from scipy.signal import resample, hilbert
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.io import savemat
from scipy.io import loadmat

import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

from speechbrain.pretrained import EncoderClassifier



import speech_recognition as sr
import Levenshtein
import jiwer
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.ASR import EncoderDecoderASR
from deepspeech4loss import DeepSpeech4Loss


# PyTorch Actor-Critic Model with LSTM to predict actions based on past state
class ActorCriticModel(nn.Module):
    def __init__(self, window_size, action_space_size, Num_Past_Windows):
        super(ActorCriticModel, self).__init__()
        
        # LSTM for past windows and actions
        self.lstm = nn.LSTM(window_size + Num_Past_Windows * window_size + Num_Past_Windows * action_space_size, 256, batch_first=True)
        
        # Actor: predicts actions
        self.actor = nn.Linear(256, action_space_size)
        
        # Critic: estimates the value of the state
        self.critic = nn.Linear(256, 1)

    def forward(self, current_window, past_windows, past_actions):
        # Flatten the past windows and actions
        past_windows_flat = past_windows.view(-1, Num_Past_Windows * window_size)
        past_actions_flat = past_actions.view(-1, Num_Past_Windows * action_space_size)
        
        # Concatenate current window, past windows, and past actions
        x = torch.cat([current_window, past_windows_flat, past_actions_flat], dim=1)
        x = x.unsqueeze(1)  # Add batch dimension
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Get the output from the last time step
        
        # Predict actions (Actor)
        actions = torch.sigmoid(self.actor(lstm_out))  # Sigmoid to keep actions in range [0, 1]
        
        # Predict value (Critic)
        value = torch.sigmoid(self.critic(lstm_out))
        
        return actions, value



# Parameters
window_size = 1024  # Size of each speech window
action_space_size = 6  # 4 continuous actions
max_action_value = 1  # Max value of action
min_action_value = 0  # Min value of action
n_speeches = 200  # Number of speech examples
Num_Past_Windows = 2

model_path = "D:/Irtaza/VoiceSecure_Artifacts/Trained_Model/Trained_VoiceSecure_Model.pth"

# Create environment and model
actor_critic_model = ActorCriticModel(window_size, action_space_size, Num_Past_Windows)
# Load the saved model parameters
if(1):
    model_save_path = model_path
    actor_critic_model.load_state_dict(torch.load(model_save_path))
    actor_critic_model.eval()
print("########################################################################")
print("########################################################################")
print("--> Functional: Loading Pre-trained VoiceSecure Model")
optimizer = optim.Adam(actor_critic_model.parameters(), lr=0.001)

# dummy values
current_window = np.zeros((window_size,))
past_windows = np.zeros((Num_Past_Windows, window_size))
past_actions = np.zeros((Num_Past_Windows, action_space_size))

current_window_tensor = torch.FloatTensor(current_window).unsqueeze(0)
past_windows_tensor = torch.FloatTensor(past_windows).unsqueeze(0)
past_actions_tensor  = torch.FloatTensor(past_actions).unsqueeze(0)

action, value = actor_critic_model(current_window_tensor, past_windows_tensor, past_actions_tensor)
print("--> Functional: Running VoiceSecure Model")


Fs = 16000
audio = np.random.uniform(low=-1.0, high=1.0, size=2*Fs).astype(np.float32)

def ComputeEmbedding(signal, classifier):
    signal = torch.from_numpy(signal)
    embeddings = classifier.encode_batch(signal)
    embeddings = np.array(embeddings).squeeze()
    return embeddings

ModelUse = "ECAPA"
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="D:/Irtaza/VoiceSecure_Artifacts/Trained_Model/spkrec-ecapa-voxceleb")
embeddings = ComputeEmbedding(audio, classifier)
print("--> Functional: Computing Speaker Embedding from ECAPA is functional -> ", embeddings.shape)

ModelUse = "Xvector"
classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="D:/Irtaza/VoiceSecure_Artifacts/Trained_Model/spkrec-xvect-voxceleb")
embeddings = ComputeEmbedding(audio, classifier)
print("--> Functional: Computing Speaker Embedding from XVector is functional -> ", embeddings.shape)



def get_text_from_wav_using_whisper(wav_file, recognizer):
    with sr.AudioFile(wav_file) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_whisper(audio)
            return text.lower()
        except sr.UnknownValueError:
            return ""

def get_text_from_wav_using_deepspeech(wav_file, asr_model):
    waveform, sample_rate = torchaudio.load(wav_file)
    text = asr_model.predict(waveform.numpy())
    return text[0].lower()

def get_text_from_wav_using_Wav2Vec2(wav_file):
    audio_array, sample_rate = librosa.load(wav_file, sr=16000)  # Make sure sample rate is 16000 for Wav2Vec2

    # Process audio to get input values for the model
    input_values = processor(audio_array, return_tensors="pt", sampling_rate=16000, padding="longest").input_values

    # Make predictions
    with torch.no_grad():
        logits = model(input_values.to("cpu")).logits

    # Get the predicted IDs and decode them to text
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0].lower()  # Return the transcription text

def calculate_accuracy(expected, recognized):
    distance = Levenshtein.distance(expected, recognized)
    if(max(len(expected), len(recognized))==0):
       accuracy = 0
       return accuracy
    accuracy = (distance / max(len(expected), len(recognized))) * 100
    return accuracy

wav_file = "Temp.wav"
ModelUse = "Whisper"
recognizer = sr.Recognizer()
text = get_text_from_wav_using_whisper(wav_file, recognizer)
print("--> Functional: Computing Word Error Rate from Whisper is functional")

ModelUse = "DeepSpeech"
recognizer =DeepSpeech4Loss(pretrained_model='librispeech', device_type = "cpu", device='cpu')
text = get_text_from_wav_using_deepspeech(wav_file, recognizer)
print("--> Functional: Computing Word Error Rate from DeepSpeech is functional")

    
ModelUse = "Wav2Vec2"
# Load the Wav2Vec2 model and processor
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
recognizer = None
text = get_text_from_wav_using_Wav2Vec2(wav_file)
print("--> Functional: Computing Word Error Rate from Wav2Vec2 is functional")


print("Installation Complete")



print("########################################################################")
print("########################################################################")