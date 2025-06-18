import torch 
import torch.nn as nn
import torch.optim as optim



# libraries required for modification
import os
import sys
import time
import numpy as np
import glob as glob

from scipy.signal import resample, hilbert
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from scipy.signal import find_peaks
from scipy.io import savemat
from scipy.io import loadmat

import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

import torch
import torchaudio
    
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
    
    temp = ConvertToAudio(tempfx)
    
    if np.random.rand(1) < Parameters[7]:
        #print("Flip")
        temp = np.flipud(temp)
    
    temp += Parameters[8] * np.roll(temp, int(Parameters[9]))
    temp = temp/(1+Parameters[8])
    noise = np.sqrt(Parameters[10]) * np.random.randn(len(temp),1)
    
    return temp, FinalScaling
    
def ModifySpeech(current_window, Scaling, Fs, Parameters):
    temp_data = current_window.reshape(-1, 1)
    P_data, Scaling = process_segment(temp_data, Fs, Parameters, Scaling)
    modified_window = P_data.reshape(-1)
    return modified_window, Scaling


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

#############################################################################
#############################################################################

DataFolder = "D:/Irtaza/VoiceSecure_Artifacts/Data2/LibriSpeech_Dev/" # path to dataset folder
OriginalFolder="Original" # folder containing actual data files
ModificationName = "VoiceSecure_temp" # script will create a directoy in DataFolder, and save VoiceSecured speech
model_path = "D:/Irtaza/VoiceSecure_Artifacts/Trained_Model/Trained_VoiceSecure_Model.pth"
Saving = 1 # bool to save
ListOfSpeakers = glob.glob(DataFolder + OriginalFolder + "/*") # arranged according to the dataset
ListOfFiles = glob.glob(DataFolder + OriginalFolder + "/*/*/*.wav") # arranged according to the dataset
print("Number of Speakers:", len(ListOfSpeakers))
print("Number of Files:", len(ListOfFiles))

#############################################################################
#############################################################################


# Create environment and model
actor_critic_model = ActorCriticModel(window_size, action_space_size, Num_Past_Windows)
# Load the saved model parameters
if(1):
    model_save_path = model_path
    actor_critic_model.load_state_dict(torch.load(model_save_path))
    actor_critic_model.eval()

optimizer = optim.Adam(actor_critic_model.parameters(), lr=0.001)
#####################################################################

WW = 20
TotalNumberOfFiles = len(ListOfFiles)

for num_file in range(len(ListOfFiles)):
    print(num_file, " /", TotalNumberOfFiles)
    Parameters = np.array([1.0, 0.7, 0.0, 1.0, 1.08, 1.0, 10, 0.1, 0.1, 250, 0.000001])
    Scaling = np.zeros((int(window_size/2+1),1))
    Infile = ListOfFiles[num_file]
    OriginalAudio, samplerate = sf.read(Infile)
    modified_speech = OriginalAudio.copy()
    current_window = np.zeros((window_size,))
    past_windows = np.zeros((Num_Past_Windows, window_size))
    past_actions = np.zeros((Num_Past_Windows, action_space_size))
    n_windows = int(np.floor(len(OriginalAudio)/window_size))
    StoringParameters = np.zeros((n_windows-1, 11))
    
    for window_idx in range(n_windows - 1):
        # Convert data to tensors
        current_window = modified_speech[window_idx*window_size:(window_idx+1)*window_size]
        
        #####################################################
        #print("Current_window:", current_window.shape)
        #print("past_window:", past_windows.shape)
        #print("past_action:", past_actions.shape)
        #####################################################
        
        current_window_tensor = torch.FloatTensor(current_window).unsqueeze(0)  # (1,1024)
        past_windows_tensor = torch.FloatTensor(past_windows).unsqueeze(0) # (1,2,1024)
        past_actions_tensor = torch.FloatTensor(past_actions).unsqueeze(0) # (1,2,6)
        
        # Predict action and value
        action, value = actor_critic_model(current_window_tensor, past_windows_tensor, past_actions_tensor)
        action = torch.clamp(action, min=1e-6, max=1 - 1e-6)
        
        
        action = action.squeeze().detach().numpy()  # Get the predicted action
        action = np.clip(action, 0, 1)  # Ensure the actions are within the valid range
        
        
        # Apply modification to speech window
        if(1):
            Parameters[1] = 0.6 + 0.4*action[0] 
            Parameters[2] = 0.5*action[1] 
            Parameters[4] = 1.03 + 0.07*action[2] 
            Parameters[6] = 5 + 20 * action[3] 
            Parameters[8] = 0.1 + 0.4*action[4] 
            Parameters[9] = 100 + 300*action[5] 
            
            Parameters[6] = round(Parameters[6])
            Parameters[9] = round(Parameters[9])
            
        StoringParameters[window_idx, :] = Parameters

        
        modified_window, Scaling = ModifySpeech(current_window, Scaling, 16000, Parameters)
        modified_speech[window_idx*window_size:(window_idx+1)*window_size] = modified_window
        
        if(window_idx != 0):
            modified_speech[window_idx*window_size-WW:window_idx*window_size+WW+1] = CombineFrames(modified_speech[window_idx*window_size-WW:window_idx*window_size+WW+1])
        
        
        past_windows = np.roll(past_windows, shift=-1, axis=0)
        past_windows[-1,:] = current_window
        
        past_actions = np.roll(past_actions, shift=-1, axis=0)
        past_actions[-1,:] = action
        
    
    ModifiedAudio = np.array(modified_speech) # (N, )
    
    outfile = Infile.replace(OriginalFolder, ModificationName)
    
    if Saving:
        directory = os.path.dirname(outfile)
        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)
        sf.write(outfile, ModifiedAudio, 16000)
    
print("Done")
    
        