import torch 
import torch.nn as nn
import torch.optim as optim



# libraries required for modification
import numpy as np
import glob as glob
from scipy.signal import resample, hilbert
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
import time
import sounddevice as sd
import soundfile as sf
import librosa
import matplotlib
import matplotlib.pyplot as plt
from scipy.fft import fft


import speech_recognition as speech_to_text_model
import Levenshtein
from scipy.io import savemat
from scipy.io import loadmat
import os
import sys
import torch
import time
from tqdm import tqdm
import pickle
import jiwer
import torchaudio
from pystoi import stoi
from pydub import AudioSegment
from io import BytesIO
from speechbrain.pretrained import EncoderClassifier
from deepspeech4loss import DeepSpeech4Loss


###############################################################################
###############################################################################

def get_text_from_wav_using_deepspeech(wav_file, asr_model):
    waveform = wav_file
    waveform = waveform.reshape(-1, len(waveform))
    text = asr_model.predict(waveform)
    return text[0].lower()

def get_text_from_wav_using_whisper(waveform, recognizer):
    waveform = (waveform * 32767).astype(np.int16)
    audio_segment = AudioSegment(
        waveform.tobytes(), 
        frame_rate=16000,
        sample_width=waveform.dtype.itemsize, 
        channels=1
    )
    
    # Save to a BytesIO buffer
    buffer = BytesIO()
    audio_segment.export(buffer, format="wav")
    buffer.seek(0)  # Move back to the start of the BytesIO buffer

    # Use the speech_recognition library as usual
    with speech_to_text_model.AudioFile(buffer) as source:
        audio = recognizer.record(source)
        try:
            text = recognizer.recognize_whisper(audio)
            return text.lower()
        except speech_to_text_model.UnknownValueError:
            return ""

def calculate_accuracy(expected, recognized):
    distance = Levenshtein.distance(expected, recognized)
    if(max(len(expected), len(recognized))==0):
       accuracy = 0
       return accuracy
    accuracy = (distance / max(len(expected), len(recognized))) * 100
    return accuracy

def calculate_accuracy_WER(expected, recognized):
    wer = jiwer.wer((expected), (recognized))
    return wer*100



###############################################################################
###############################################################################
def ComputeEmbedding(waveform, classifier):
    #signal, fs =torchaudio.load(filename)
    signal = torch.tensor(np.transpose(waveform))
    embeddings = classifier.encode_batch(signal)
    embeddings = np.array(embeddings).squeeze()
    return embeddings

def cosine_similarity(vec1, vec2):
    # Compute the dot product
    dot_product = np.dot(vec1, vec2)
    
    # Compute the magnitudes (norms) of the vectors
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    # Return the cosine similarity
    return dot_product / (norm_vec1 * norm_vec2)

def rmse(y_true, y_pred):
    # Compute the squared differences
    squared_diff = (y_true - y_pred) ** 2
    
    # Compute the mean of the squared differences
    mean_squared_diff = np.mean(squared_diff)
    
    # Return the square root of the mean squared differences (RMSE)
    return np.sqrt(mean_squared_diff)


###############################################################################
###############################################################################
def windows_to_vector(windows):
    # Reshape the windows array back to a 1D array
    vector = np.reshape(windows, (-1,))
    return vector

def create_non_overlapping_windows(signal, window_size=1024):
    # Ensure the signal length is a multiple of window_size by trimming the extra part
    trimmed_length = (len(signal) // window_size) * window_size
    signal = signal[:trimmed_length]

    # Reshape the signal into windows
    windows = np.reshape(signal, (-1, window_size))
    return windows, signal

def CreatingDataSet():
    DataFolder = "D:/Irtaza/VoiceSecure_Artifacts/Data2/TrainingSet/" # adjust according to the dataset
    
    OriginalFolder="Original"
    ListOfSpeakers = glob.glob(DataFolder + OriginalFolder + "/*") # adjust according to the dataset
    ListOfFiles = glob.glob(DataFolder + OriginalFolder + "/*/*/*.wav") # adjust according to the dataset
    print("Number of Speakers:", len(ListOfSpeakers))
    print("Number of Files:", len(ListOfFiles))
    
    SpeechDataSet = []
    OriginalSpeech = []
    for i in range(len(ListOfFiles)):
        Infile = ListOfFiles[i]
        data, samplerate = sf.read(Infile)
        WindowedData, Speech = create_non_overlapping_windows(data, window_size=1024)
        SpeechDataSet.append(WindowedData)
        OriginalSpeech.append(Speech)
    return SpeechDataSet, OriginalSpeech
    
    
###############################################################################
###############################################################################

def gaussian_vector(size, peak_position, sigma):
    # Create an array of indices
    x = np.arange(size)
    
    # Calculate Gaussian curve values
    gaussian = np.exp(-((x - peak_position) ** 2) / (2 * sigma ** 2))
    
    # Normalize the values to be between 0 and 1
    gaussian /= np.max(gaussian)
    
    return gaussian.reshape(-1, 1)

def CombineFrames(data):
    data = data.reshape(-1, 1)
    WW = 20
    XX = np.arange(0, 2*WW + 1)  # Creating XX as [0, 1, 2, ..., 40]
    spline = UnivariateSpline(XX, data, s=1)
    spline_values = spline(XX).reshape(-1, 1)
    S1 = gaussian_vector(41, 20, 3)
    S2 = 1-S1
    FinalData = S1*spline_values +  S2*data
    FinalData = FinalData.reshape(-1)
    return FinalData

###############################################################################
###############################################################################


def ConvertToAudio(Fx):
    # receive positive side frequency components
    Fx[0] = np.real(Fx[0])
    Fx[-1] = np.real(Fx[-1])
    
    tempfx = np.append(Fx, np.conj(np.flipud(Fx[1:-1])), axis=0)
    NewAudio = np.fft.ifft(tempfx, axis = 0)
    return np.real(NewAudio)


def ShiftFormant(Sig1, Fs, delay):
    Sig1 = np.squeeze(Sig1)
    kernel_size = 5
    kernel = np.ones(kernel_size) / kernel_size
    
    plotting = 0
    DelayFactor = int(round(delay))
    CurrentLength = len(Sig1)
    NewLength = int(Fs / 2)
    Env1 = np.convolve(Sig1, kernel, mode='same')
    Env1[Env1 <= 0] = 0.05
    REnv1 = resample(Env1, NewLength)

    if DelayFactor > 0:
        ShiftedEnv = np.zeros_like(REnv1)
        ShiftedEnv[DelayFactor:] = REnv1[:-DelayFactor]
    else:
        ShiftedEnv = np.zeros_like(REnv1)
        ShiftedEnv[:DelayFactor] = REnv1[-DelayFactor:]

    Env2 = resample(ShiftedEnv, CurrentLength)
    Scaling = Env2 / Env1

    Scaling[Scaling < 0] = 0
    Scaling[Scaling > 5] = 0.1
    return Scaling.reshape(-1,1)


def ProcessPitch(tempfx, PitchFactor):
    N = len(tempfx)
    NewLength = int(np.ceil(PitchFactor*N))
    New_temp = np.zeros((NewLength,1), dtype=np.complex128)
    
    
    for i in range(N):
        new_ind = int(np.ceil(i*PitchFactor))
        New_temp[new_ind] = New_temp[new_ind] + tempfx[i]
    New_temp = New_temp[:len(tempfx)]
    return New_temp
        

def process_segment(temp, Fs, Parameters, PrevScaling):
    temp2 = temp
    tempfx = np.fft.fft(temp, axis = 0)
    tempfx = tempfx[:len(temp) // 2 + 1]
    
    Scaling= np.ones((len(tempfx), 1))
    FinalScaling = PrevScaling
    if np.random.rand(1) <= Parameters[0]:
        #print("Inaudible freuqency")
        absTempfx = np.abs(tempfx)
        if(np.max(absTempfx) !=0):
            Scaling[absTempfx < Parameters[1]] = Parameters[2]
            FinalScaling = PrevScaling + 0.2*(Scaling - PrevScaling)
            tempfx  = tempfx * FinalScaling
            tempfx[:5] *= 0.2*tempfx[:5] 
             
    
    if np.random.rand(1) < Parameters[3]:
        #print("Pitch")
        tempfx = ProcessPitch(tempfx, Parameters[4])
    
    if np.random.rand(1) < Parameters[5]:
        #print("Formant")
        Scaling22 = ShiftFormant(np.abs(tempfx), Fs, Parameters[6])
        tempfx = Scaling22 * tempfx   
    
    # Implement ConvertToAudio function (not provided)
    temp = ConvertToAudio(tempfx)
    
    if np.random.rand(1) < Parameters[7]:
        #print("Flip")
        temp = np.flipud(temp)
    
    temp += Parameters[8] * np.roll(temp, int(Parameters[9]))
    temp = temp/(1+Parameters[8])
    
    return temp, FinalScaling
    
def ModifySpeech(current_window, Scaling, Fs, Parameters):
    temp_data = current_window.reshape(-1, 1)
    P_data, Scaling = process_segment(temp_data, Fs, Parameters, Scaling)
    
    modified_window = P_data.reshape(-1)
    return modified_window, Scaling

def SmoothCombinedFrames(data, WindowSize):
    NumWindows = int(np.floor(len(data)/WindowSize))
    WW = 20
    for i in range(NumWindows-1):
        if(i != 0):
            data[i*WindowSize-WW:i*WindowSize+WW+1] = CombineFrames(data[i*WindowSize-WW:i*WindowSize+WW+1])
    return data

def ComputeReward(OriginalAudio, ModifiedAudio):
    Speaker_Original = ComputeEmbedding(OriginalAudio, Speaker_classifier)
    Speaker_Modified = ComputeEmbedding(ModifiedAudio, Speaker_classifier)
    SS_similarity = cosine_similarity(Speaker_Original, Speaker_Modified)
    Sim_Score = 1-SS_similarity
    
    #recognized_text1 = get_text_from_wav_using_whisper(OriginalAudio, Whisper_recognizer)
    #recognized_text2 = get_text_from_wav_using_whisper(ModifiedAudio, Whisper_recognizer)
    
    recognized_text1 = get_text_from_wav_using_deepspeech(OriginalAudio, deepspeech_recognizer)
    recognized_text2 = get_text_from_wav_using_deepspeech(ModifiedAudio, deepspeech_recognizer)
    
    accuracy1 = calculate_accuracy(recognized_text1, recognized_text2)
    wer1 = calculate_accuracy_WER(recognized_text1, recognized_text2)
    wer1 = wer1/100
    
    stoi_score = stoi(OriginalAudio, ModifiedAudio, 16000)
    
    Rewards = ((stoi_score) + (wer1) + (1*Sim_Score))
    Rewards = (2/3)*(Rewards) - 1
    
    return Rewards, stoi_score, wer1, Sim_Score

# Environment setup (state includes current speech window, past windows, past actions)
class SpeechEnv:
    def __init__(self, speech_dataset, window_size, action_space_size, Num_Past_Windows):
        self.speech_dataset = speech_dataset  # List of speech examples
        self.window_size = window_size
        self.action_space_size = action_space_size
        self.past_windows = np.zeros((Num_Past_Windows, window_size))  # Store 2 previous windows
        self.past_actions = np.zeros((Num_Past_Windows, action_space_size))  # Store 2 previous actions
        self.current_speech = None
        self.Num_Past_Windows = Num_Past_Windows
    
    def reset(self):
        # Randomly select a speech example from the dataset
        self.current_speech = self.speech_dataset[np.random.randint(len(self.speech_dataset))]
        self.n_windows = len(self.current_speech)
        
        # Reset past windows and actions
        self.past_windows = np.zeros((self.Num_Past_Windows, self.window_size))
        self.past_actions = np.zeros((self.Num_Past_Windows, self.action_space_size))
        
        return self.current_speech[0], self.past_windows, self.past_actions, self.n_windows
    
    def step(self, current_window_idx, action):
        # Update past windows and past actions
        self.past_windows = np.roll(self.past_windows, shift=-1, axis=0)
        self.past_actions = np.roll(self.past_actions, shift=-1, axis=0)
        self.past_windows[-1] = self.current_speech[current_window_idx]  # Current window becomes last in past
        self.past_actions[-1] = action  # Current action becomes last in past
        
        # Return next window, along with updated past windows and actions
        if current_window_idx + 1 < self.n_windows:
            return self.current_speech[current_window_idx + 1], self.past_windows, self.past_actions
        else:
            return None  # End of episode

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

SpeechDataset, OriginalSpeech = CreatingDataSet()



Speaker_classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="D:/Irtaza/VoiceSecure_Artifacts/Trained_Model/spkrec-xvect-voxceleb")
#Speaker_classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="D:/Irtaza/VoiceSecure_Artifacts/Trained_Model/spkrec-ecapa-voxceleb")
device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
deepspeech_recognizer =DeepSpeech4Loss(pretrained_model='librispeech', device_type = "gpu", device=device)
Whisper_recognizer = speech_to_text_model.Recognizer()




# Create environment and model
env = SpeechEnv(SpeechDataset, window_size, action_space_size, Num_Past_Windows)
actor_critic_model = ActorCriticModel(window_size, action_space_size, Num_Past_Windows)
optimizer = optim.Adam(actor_critic_model.parameters(), lr=0.001)

# Training loop parameters
n_episodes = 100000
gamma = 0.99  # Discount factor for reward

# Store training statistics
all_rewards = []
all_losses = []


STOI_SCORE = []
WER_SCORE = []
SIM_SCORE = []

for episode in range(n_episodes):
    print("Episode: ", episode)
    # Reset environment with a new random speech example
    current_window, past_windows, past_actions, n_windows = env.reset()
    modified_speech = []
    episode_rewards = []

    # Initialize lists to store states, actions, and rewards
    states = []
    actions = []
    rewards = []
    
    Scaling = np.zeros((int(window_size/2+1),1))
    
    # Starting Parameters
    Parameters = np.array([1.0, 1.0, 0.0, 1.0, 1.08, 1.0, 20, 0.1, 0.5, 250, 0.000001])
    
    for window_idx in range(n_windows):
        # Convert data to tensors
        current_window_tensor = torch.FloatTensor(current_window).unsqueeze(0)  # Add batch dim
        past_windows_tensor = torch.FloatTensor(past_windows).unsqueeze(0)
        past_actions_tensor = torch.FloatTensor(past_actions).unsqueeze(0)

        # Predict action and value
        action, value = actor_critic_model(current_window_tensor, past_windows_tensor, past_actions_tensor)
        action = torch.clamp(action, min=1e-6, max=1 - 1e-6)
        action = action.squeeze().detach().numpy()  # Get the predicted action
        action = np.clip(action, 0, 1)  # Ensure the actions are within the valid range
        
        # actions can be [0 1]
        Parameters[1] = 0.6 + 0.4*action[0] # 0.6 to 1
        Parameters[2] = 0.5*action[1] # 0 to 0.5
        Parameters[4] = 1.03 + 0.07*action[2] # 1.03 to 1.10
        Parameters[6] = 5 + 20 * action[3] # 5 to 25
        Parameters[8] = 0.1 + 0.4*action[4] # 0.1 to 0.5
        Parameters[9] = 100 + 300*action[5] # 100 to 400
        Parameters[6] = round(Parameters[6])
        Parameters[9] = round(Parameters[9])
        
        '''
        if episode % 50 == 0:
            print(f"Parameters[1]: {Parameters[1]}, Parameters[2]: {Parameters[2]}, Parameters[4]: {Parameters[4]}, Parameters[6]: {Parameters[6]}, Parameters[8]: {Parameters[8]}, Parameters[9]: {Parameters[9]}")
            print(f"action[0]: {action[0]}, action[1]: {action[1]}, action[2]: {action[2]}, action[3]: {action[3]}, action[4]: {action[4]}, action[5]: {action[5]}")
            print("================================================================================================")
        '''
        modified_window, Scaling = ModifySpeech(current_window, Scaling, 16000, Parameters)
        modified_speech.append(modified_window)
        
        # Store current state
        states.append((current_window, past_windows, past_actions))
        
        # Step to the next window
        result = env.step(window_idx, action)
        if result is None:
            break  # Episode finished
        current_window, past_windows, past_actions = result

    # Compute total reward after processing all windows
    ModifiedSpeech = np.array(modified_speech)
    
    OriginalAudio = windows_to_vector(env.current_speech)
    ModifiedAudio = windows_to_vector(ModifiedSpeech)
    TempM = ModifiedAudio.copy()
    ModifiedAudio = SmoothCombinedFrames(TempM, 1024)
    
    reward, stoi_score, wer1, SpeakerSound_similarity = ComputeReward(OriginalAudio, ModifiedAudio)
    rewards.append(reward)
    STOI_SCORE.append(stoi_score)
    WER_SCORE.append(wer1)
    SIM_SCORE.append(SpeakerSound_similarity)
    
    # Discount rewards for backpropagation
    discounted_rewards = [reward] * len(states)
    # Convert rewards to tensors
    discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32)
    # Update policy and value functions
    optimizer.zero_grad()
    total_loss = torch.tensor(0.0, requires_grad=True)  # Initialize total_loss as a tensor
    past_action = torch.tensor(np.zeros((1,6)))
    target_action = torch.tensor([[1.0, 0.0, 1.0, 0.25, 0.25, 0.5]])
    target_change = torch.tensor([[0.1, 0.1, 0.15, 0.1, 0.1, 0.1]])
    past_action_mean = np.zeros((10, 6))
    
    for i in range(len(states)):
        # Convert state back to tensors
        state = states[i]
        current_window_tensor = torch.FloatTensor(state[0]).unsqueeze(0)
        past_windows_tensor = torch.FloatTensor(state[1]).unsqueeze(0)
        past_actions_tensor = torch.FloatTensor(state[2]).unsqueeze(0)

        predicted_action, predicted_value = actor_critic_model(current_window_tensor, past_windows_tensor, past_actions_tensor)
        
        if torch.isnan(predicted_action).any() or torch.isnan(predicted_value).any():
            print(f"NaN detected in predicted action or value at episode {episode}, window {window_idx}")
            
        predicted_action = torch.clamp(predicted_action, min=1e-6, max=1 - 1e-6)
        
        
        # Compute advantage
        advantage = discounted_rewards[i]
        action_loss = -2*advantage
        action_moving = 1*torch.sum((torch.abs(past_action - predicted_action) - target_change)**2)
        target_loss = torch.sum((target_action - predicted_action)**2)
        Mean_Action_loss = -1*torch.sum((torch.tensor(np.mean(past_action_mean, axis = 0)) - predicted_action)**2)
        past_action_mean = np.roll(past_action_mean, shift=-1, axis=0)
        past_action_mean[-1,:] = predicted_action.detach().numpy()
        past_action = predicted_action
        
        # Value loss (value function approximation loss)
        value_loss = (predicted_value.squeeze() - discounted_rewards[i]) ** 2
        if episode % 200 == 0:
            print("P__Action:", predicted_action)
            print("Reward:", discounted_rewards[i], " -> P__Value:", predicted_value)
            print("advantage:", advantage, " -> Action Loss:", action_loss, " -> Value Loss:", value_loss)
        total_loss = total_loss + action_loss + value_loss + 2*action_moving + 1.5*target_loss + 2*Mean_Action_loss
        
        
    total_loss = total_loss/len(states)
    total_loss = torch.clamp(total_loss, min = -3000)
    # Backpropagation
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor_critic_model.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()

    # Track rewards and losses for analysis
    all_rewards.append(np.mean(rewards))
    all_losses.append(total_loss.item())

    if episode % 100 == 0:
        print(f"Episode {episode}/{n_episodes}, \
              Average Reward: {np.mean(all_rewards[-10:]):.4f}, \
              STOI: {np.mean(STOI_SCORE[-10:]):.4f}, \
              WER: {np.mean(WER_SCORE[-10:]):.4f}, \
              SIM: {np.mean(SIM_SCORE[-10:]):.4f}")

# Save the model's state dictionary
model_save_path = 'Trained_Model.pth'  # Specify the path where you want to save the model
torch.save(actor_critic_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")