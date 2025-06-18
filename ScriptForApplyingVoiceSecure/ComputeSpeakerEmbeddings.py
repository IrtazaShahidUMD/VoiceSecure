import torch
import torchaudio
from speechbrain.pretrained import EncoderClassifier
import glob as glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat
import librosa
import os

def ComputeEmbedding(filename, classifier):
    signal, fs =torchaudio.load(filename)
    embeddings = classifier.encode_batch(signal)
    embeddings = np.array(embeddings).squeeze()
    return embeddings

def RunSpeechRecognition(classifier, DataFolder, ManipulatedDir, ResultsFile, ModelUse):
    #python program to check if a directory exists
    path = DataFolder + ModelUse + "Embeddings2/"
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
    Embeddings = []
    SpeakerNames = []
    for i in range(len(FileNames)):
        wav_file1 = FileNames[i]
        SpeakerNames.append(wav_file1.split(os.sep)[-3])
        
        
        if(ModelUse == "Xvector"):
            embedding = ComputeEmbedding(wav_file1, classifier)
        
        if(ModelUse == "ECAPA"):
            embedding = ComputeEmbedding(wav_file1, classifier)
        
        Embeddings.append(embedding)
        print(i, "/", len(FileNames))
        
    SpeakerNames = np.reshape(np.array(SpeakerNames), [-1,1])
    Embeddings = np.array(Embeddings)
    FileNames = np.array(FileNames)
    
    print("#############################################################")
    print("Computed Speaker Embeddings using model: ",ModelUse )
    print("Speakers:", np.shape(SpeakerNames))
    print("Embeddings:", np.shape(Embeddings))
    print("FileNames:",np.shape(FileNames))
    print("#############################################################")
    print("ResultsFile:", ResultsFile)
    savemat(ResultsFile, {'Embeddings':Embeddings, 'FileNames':FileNames, 'SpeakerNames':SpeakerNames})


#############################################################################
#############################################################################
# Description
# This script compute embeddings (Xvector and ECAPA) for the modifications
# mentioned in ManipulatedDirList and store in a {ModelUse}Embeddings2 directory

DataFolder = "D:/Irtaza/VoiceSecure_Artifacts/Data2/LibriSpeech_Dev/" # path to dataset folder
ManipulatedDirList = ["Original", "VoiceSecure"] 
ModelUse = "Xvector"
#ModelUse = "ECAPA"

#############################################################################
#############################################################################

for i in range(len(ManipulatedDirList)):
    ManipulatedDir = ManipulatedDirList[i]
    ResultsFile = DataFolder + ModelUse + "Embeddings2/" + ManipulatedDir + '.mat'
    print(ResultsFile)
    
    if(ModelUse == "Xvector"):
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-xvect-voxceleb", savedir="D:/Irtaza/VoiceSecure_Artifacts/Trained_Model/spkrec-xvect-voxceleb")
    
    if(ModelUse == "ECAPA"):
        classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="D:/Irtaza/VoiceSecure_Artifacts/Trained_Model/spkrec-ecapa-voxceleb")
        
    
    RunSpeechRecognition(classifier, DataFolder, ManipulatedDir, ResultsFile, ModelUse)

