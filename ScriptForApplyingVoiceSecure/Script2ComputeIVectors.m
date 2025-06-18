%% Description
% This script compute speaker embedding using i-vector
%%
clc; clear all; close all;
%%
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\Data2\"; 
DataFolder = "LibriSpeech_Dev\";
ModificationList = ["Original"; "VoiceSecure"];
Model_Path = "D:\Irtaza\VoiceSecure_Artifacts\Trained_Model\Ivector_Pretrained_model.mat";

DataFolder = strcat(HomePath, DataFolder);
ModelUse = "Ivector";
Saving = 1;
ResultsFolder = strcat(ModelUse, "Embeddings2\");
%%
load(Model_Path);
%%
OutputDir = strcat(DataFolder, ResultsFolder);
if(Saving)
    if ~exist(OutputDir, 'dir')
        mkdir(OutputDir);
        disp("Directory Created");
    end
end

%%
for num_modification = 1:length(ModificationList)
    ManipulatedDir = ModificationList(num_modification);
    Run_Embedding_Extraction(DataFolder, ManipulatedDir, OutputDir, Saving, sr);
end

%%
function Run_Embedding_Extraction(DataFolder, ManipulatedDir, OutputDir, Saving, sr)
    ListOfFiles = dir(strcat(DataFolder,ManipulatedDir, "\*\*\*.wav")); % adjusted according to dataset
    
    SpeakerNames = [];
    FileNames = [];
    Embeddings = zeros(length(ListOfFiles), 200);
    NN = length(ListOfFiles);
    for i = 1:length(ListOfFiles)
        disp(strcat(ManipulatedDir, " -> ", num2str(i), "/", num2str(NN)));
        Spk_name = strsplit(ListOfFiles(i).folder, "\");
        Spk_name = string(Spk_name{end-1});
        SpeakerNames = [SpeakerNames; Spk_name];
        infile = strcat(ListOfFiles(i).folder, "\", ListOfFiles(i).name);
        FileNames = [FileNames; infile];
        [data, Fs] = audioread(infile);
        w = ivector(sr,data)';
        Embeddings(i,:) = w;
    end
    
    SpeakerNames = char(SpeakerNames);
    FileNames = char(FileNames);
    if(Saving)
        outfile = strcat(OutputDir, ManipulatedDir, ".mat");
        save(outfile,"Embeddings", "FileNames", "SpeakerNames");
    end
end