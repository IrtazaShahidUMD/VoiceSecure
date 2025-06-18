'''
Compute Speech Recognition Accuracy using following models
1. Whisper
2. DeepSpeech
3. Wav2Vec2 Facebook
Just change the parameter: ModelUse
Also define the path and name for manipulated dir

'''


import speech_recognition as sr
import Levenshtein
import glob as glob
import numpy as np
from matplotlib import pyplot as plt
from scipy.io import savemat
import os
import sys
import torch
import time
from tqdm import tqdm
import pickle
import jiwer
import torchaudio
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from speechbrain.inference.ASR import EncoderDecoderASR

from deepspeech4loss import DeepSpeech4Loss


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
        logits = model(input_values.to("cuda")).logits

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

def calculate_accuracy_WER(expected, recognized):
    wer = jiwer.wer((expected), (recognized))
    return wer*100

def loadLabels(text_file_path):
    # Initialize an empty dictionary to store the data
    file_data = {}
    
    # Read the text file
    with open(text_file_path, 'r') as file:
        for line in file:
            # Split the line into filename and sentence
            parts = line.strip().split(' ', 1)
            if len(parts) == 2:
                filename = parts[0]
                sentence = parts[1]
    
                # Store the data in the dictionary
                file_data[filename] = sentence
    
    # Print the dictionary
    return file_data


def RunSpeechRecognition(recognizer, DataFolder, ManipulatedDir, ResultsFile, ModelUse, Labels):
    #python program to check if a directory exists
    path = DataFolder +  ModelUse +"WERScores2/"
    # Check whether the specified path exists or not
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)
        print("The new directory is created!")
    
    ManipulatedFolder = ManipulatedDir
    FileNames = glob.glob(DataFolder+ManipulatedDir+"/*/*/*.wav") # adjusted according to dataset
    FileNames = np.sort(FileNames)
    #print(FileNames)
    print("Number of files: ", len(FileNames))
    
    Accuracy1=[]
    Wer1 = []
    SpeakerNames = []
    for i in range(len(FileNames)):
        wav_file1 = FileNames[i]
        
        # extracting labels according to the labels file (vary across datasets)
        SpeakerNames.append(wav_file1.split(os.sep)[-3])
        labelPointer= wav_file1.split(os.sep)[-1]
        labelPointer = labelPointer[:-4]
        TrueLabel = Labels[labelPointer].lower()
        
        
        if(ModelUse == "Whisper"):
            recognized_text1 = get_text_from_wav_using_whisper(wav_file1, recognizer)
            
        if(ModelUse == "DeepSpeech"):
            recognized_text1 = get_text_from_wav_using_deepspeech(wav_file1, recognizer)
        
        if(ModelUse == "Wav2Vec2"):
            recognized_text1 = get_text_from_wav_using_Wav2Vec2(wav_file1)
        
        accuracy1 = calculate_accuracy(TrueLabel, recognized_text1)
        
        wer1 = calculate_accuracy_WER(TrueLabel, recognized_text1)
        
        Accuracy1.append(accuracy1)
        Wer1.append(wer1)
        print(i, "/", len(FileNames), " => Dist: ", accuracy1, " => Wer: ", wer1)
        if (0):
            print("==========================================================")
            print(i)
            print("Original->", TrueLabel)
            print("Modified->",recognized_text1)
        
            
            print(f"Accuracy for WAV File 1: {accuracy1:.2f}%")
            print("==========================================================")
            #sys.exit(0)
    print("#############################################################")
    print("Modification type:", ManipulatedDir)
    print("Modified Audio -> Overall Distance1:",np.mean(Accuracy1))
    
    print("Modified Audio -> Overall Wer1:",np.mean(Wer1))
    print("#############################################################")
    #BigAccuracy.append(np.mean(Accuracy))
    print("ResultsFile:", ResultsFile)
    savemat(ResultsFile, {'Accuracy1': Accuracy1, 'Wer1':Wer1, 'SpeakerNames':SpeakerNames})



#############################################################################
#############################################################################
# Description
# This script compute word error rate (DeepSpeech, Whisper and Wav2Vec2) for the modifications
# mentioned in ManipulatedDirList and store in a {ModelUse}WERScores2 directory
# Also mention path to the label file for the corresponding datasets

# Specify the path to the text file
text_file_path = "D:/Irtaza/VoiceSecure_Artifacts/Data2/LibriSpeech_Dev/Labels.txt"
Labels = loadLabels(text_file_path)
print(len(Labels))

DataFolder = "D:/Irtaza/VoiceSecure_Artifacts/Data2/LibriSpeech_Dev/"
OriginalFolder='Original'
ManipulatedDirList = ["Original", "VoiceSecure"]

#ModelUse = "DeepSpeech"
ModelUse = "Whisper"
#ModelUse = "Wav2Vec2"

#############################################################################
#############################################################################

for i in range(len(ManipulatedDirList)):
    ManipulatedDir = ManipulatedDirList[i]
    ResultsFile = DataFolder + ModelUse + "WERScores2/" + ManipulatedDir + '.mat'
    
    
    if(ModelUse == "Whisper"):
        recognizer = sr.Recognizer()
    
    if(ModelUse == "DeepSpeech"):
        device = torch.device(
                    'cuda' if torch.cuda.is_available() else 'cpu')
        recognizer =DeepSpeech4Loss(pretrained_model='librispeech', device_type = "gpu", device=device)
    
    if(ModelUse == "Wav2Vec2"):
        # Load the Wav2Vec2 model and processor
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h").to("cuda")
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        recognizer = None
    


    RunSpeechRecognition(recognizer, DataFolder, ManipulatedDir, ResultsFile, ModelUse, Labels)

