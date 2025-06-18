%% Description
% This script take computed WER score from {ModelUse}WERScores2 directory
% from the corresponding dataset and compute mean score and store compiled
% results for all modification in CompiledResults directory in DataFolder
%%
clc; clear all; close all;
%% DataPath
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\Data2\";
DataFolder = "LibriSpeech_Dev\"; % Select DataSet
DataFolder = strcat(HomePath, DataFolder);
InputFolder = "Original/";
ModificationList = ["Original"; "VoiceSecure"];

%ModelUse = "Whisper";
ModelUse = "DeepSpeech";
%ModelUse = "Wav2Vec2";
%% Plotting all modifications together WER Whisper

ModificationList2 = ModificationList;
AllScore = [];
WERScore = [];
ResultsFolder = strcat(ModelUse,"WERScores2/");
for i = 1:length(ModificationList)
    Modification = ModificationList(i);
    infile = strcat(DataFolder, ResultsFolder, Modification, ".mat");
    load(infile);
    AllScore = [AllScore Accuracy1'];
    WERScore = [WERScore Wer1'];
end
ModificationType = [ModificationList2];
%%
% figure;
% boxplot((AllScore), ModificationType, 'Symbol',".", "Colors","k");
% grid minor;
% ylabel("Lav Distance"); ylim([0 100])
% title(ModelUse);

figure;
boxplot((WERScore), ModificationType, 'Symbol',".", "Colors","k");
grid minor;
ylabel("Word Error Rate"); ylim([0 100])
title(ModelUse);
%%
CompiledResults = strcat(DataFolder, "CompiledResults\", ModelUse, ".mat");
%save(CompiledResults, "WERScore", "ModificationType");