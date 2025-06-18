%% Description
% This file plot speech intelligibility results
% which already computed and stored in compiled results
%%
clc; clear all; close all;
%% Path to DataSets
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\"; 
DataPath = "Data2\";
%%
infile = strcat(HomePath, DataPath, "LibriSpeech_Dev/CompiledResults/Stoi.mat");
load(infile);

Score = 100*mean(AllScore);
ModificationType(2) = "Benign";
figure;
bar("Speech Intelligibility", Score);
grid; grid minor;
ylabel("Percentage (%)");
legend(ModificationType);
legend('Location', 'northwest', 'FontSize',20);
set(gcf, 'Position', [100, 100, 650, 450]);
set(gca, "FontSize", 20);

%outfile = "plots/Revised_STOI_Score.png";
%saveas(gcf, outfile);
