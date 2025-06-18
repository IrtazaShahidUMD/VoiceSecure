%% Description
% This file plot compiled comparison results with benign noises
%%
clc; clear all; close all;
%%
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\"; 
DataPath = "Data2\";
%%
DataFolder = strcat(HomePath, DataPath, "Benign/");
WER_infile = strcat(DataFolder, "CompiledResults/", "DeepSpeech.mat");
Speaker_infile = strcat(DataFolder, "CompiledResults/", "Xvector.mat");

load(WER_infile);
load(Speaker_infile);

WER = mean(WERScore);
figure; hold on;
bar(ModificationType, WER);
ylabel("Word Error Rate (%)");
grid; grid minor;
set(gca, "FontSize", 20);
ylim([0 60])


figure; hold on;
bar(ModificationType, ComputedMMR);
ylabel("MisMatch Rate (%)");
grid; grid minor;
set(gca, "FontSize", 20);
ylim([0 40])
