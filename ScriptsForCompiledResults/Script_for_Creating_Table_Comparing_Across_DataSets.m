%% Description
% This file create tables for comparing MMR and WER across three different
% datasets
%%
clc; clear all; close all;
%%
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\"; 
DataPath = "Data2\";

DataSets_ASV = ["LibriSpeech_Dev"; "VoxCeleb_Dev"; "VCTK"];
DataSets_ASR = ["LibriSpeech_Dev"; "VCTK"; "CommonVoice";];

SpeakerModels = "Xvector";
ASRModels = "DeepSpeech";

UseIndices = [1 2 3 4 5];
Method = ["McAdams"; "VoiceMask";  "VoiceSecure"];
%% Plotting Speaker Results
MMR_Data = [];
for i = 1:length(DataSets_ASV)
    infile = strcat(HomePath, DataPath, DataSets_ASV(i), "\", "CompiledResults\", SpeakerModels, ".mat");
    load(infile);
    ComputedMMR = ComputedMMR(end-2:end);
    MMR_Data = [MMR_Data ComputedMMR];
end
%% Creating table
LibriSpeech = MMR_Data(:,1);
VoxCeleb = MMR_Data(:,2);
VCTK = MMR_Data(:,3);
MMR_Table = table(Method, LibriSpeech, VoxCeleb, VCTK);

%% Plotting WER Results
WER_Data = [];
for i = 1:length(DataSets_ASR)
    infile = strcat(HomePath, DataPath, DataSets_ASR(i), "\", "CompiledResults\", ASRModels, ".mat");
    load(infile);
    ComputedWER = mean(WERScore);
    ComputedWER = ComputedWER(end-2:end);
    if (i > 1) % because results for Dataset 2 and Dataset 3 are in different order
        ComputedWER = circshift(ComputedWER,-1);
    end
    WER_Data = [WER_Data ComputedWER'];
end
%% Creating table
LibriSpeech = WER_Data(:,1);
VCTK = WER_Data(:,2);
CommonVoice = WER_Data(:,3);

WER_Table = table(Method, LibriSpeech, VCTK, CommonVoice);
%% Printing tables
disp(MMR_Table);
disp(WER_Table);