%% Description
% This script compute perceptual score (STOI) for all modification in 
% ModificationList relative to the InputFolder, and store results in the
% StoiScores2 directory inside DataFolder
%%
clc; clear all; close all;
%%
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\Data2\";
DataFolder = "LibriSpeech_Dev\";
InputFolder = "Original\";
ModificationList = ["Original"; "VoiceSecure"];

DataFolder = strcat(HomePath, DataFolder);
SaveScore = 1;
ModelUse = "Stoi";
%%
SaveFolder = strcat(DataFolder, ModelUse, "Scores2\");
if(SaveScore)
    if ~exist(SaveFolder, 'dir')
        mkdir(SaveFolder);
    end
end
%%


ListOfFiles = dir(strcat(DataFolder, InputFolder, "*/*/*.wav")); % adjusted according to dataset
for i = 1:length(ModificationList)
    Modification = ModificationList(i);
    if(ModelUse == "Stoi")
        [StoiScore] = ComputeStoiScore(ListOfFiles, InputFolder, Modification);
        if(SaveScore)
        savingfile = strcat(SaveFolder, Modification, ".mat");
        %save(savingfile, 'StoiScore');
        end
    end
end
%%
function [Score] = ComputeStoiScore(ListOfFiles, InputFolder, Modification)
    N = length(ListOfFiles);
    Score = zeros(N,1);
    f = waitbar(0, 'Starting');
    for i = 1:N
        waitbar(i/N, f, sprintf('Modification (%s): %d %%',Modification,floor(i/N*100)));
        infile1 = strcat(ListOfFiles(i).folder, "\", ListOfFiles(i).name);
        [data1, Fs] = audioread(infile1);
        infile2 = strrep(infile1, InputFolder, strcat(Modification,"\"));
        [data2, Fs] = audioread(infile2);
        v = min(length(data1) ,length(data2));
        data1 = data1(1:v);
        data2 = data2(1:v);
        Score(i) = stoi(data1, data2, Fs);
    end
    close(f);
end