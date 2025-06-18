%% Description
% This script take compile stoi score from StoiScores2 directory
% from the corresponding dataset and compute mean score and store compiled
% results for all modification in CompiledResults directory in DataFolder
%%
clc; clear all; close all;
%% DataPaths
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\Data2\";
DataFolder = "LibriSpeech_Dev\"; % Select DataSet
InputFolder = "Original\";
ModificationList = ["Original"; "VoiceSecure"];

DataFolder = strcat(HomePath, DataFolder);
ModelUse = "Stoi";
ModificationList2 = ModificationList;
%% Plotting Stoi Score
AllScore = [];
ResultsFolder = strcat(ModelUse,"Scores2\");
for i = 1:length(ModificationList)
    Modification = ModificationList(i);
    infile = strcat(DataFolder, ResultsFolder, Modification, ".mat");
    load(infile);
    AllScore = [AllScore StoiScore]; 
end
ModificationType = [ModificationList2];
%%
figure;
boxplot((AllScore), [ModificationType], 'Symbol',".", "Colors","k");
grid minor;
ylabel("Stoi Score"); ylim([0 1])
title(ModelUse);
CompiledResults = strcat(DataFolder, "CompiledResults\", ModelUse, ".mat");
%save(CompiledResults, "AllScore", "ModificationType");