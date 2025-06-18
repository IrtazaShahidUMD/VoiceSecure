%% Description
% This file plot ablation study results for both mismatch rate and word
% error rate.
%%
clc; clear all; close all;
%% Path to DataSets
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\"; 
DataPath = "Data2\";
%%
DataFolder = strcat(HomePath, DataPath, "LibriSpeech_Dev/");
WER_infile = strcat(DataFolder, "CompiledResults/", "Ablation_DeepSpeech.mat");
Speaker_infile = strcat(DataFolder, "CompiledResults/", "Ablation_Xvector.mat");

load(WER_infile);
load(Speaker_infile);
WER = mean(WERScore);
%%
including_fixed_parameter = 1;
ModificationType = ["original";
    "VoiceSecure";
    "frequency removal";
    "inducing echo";
    "temporal flipping";
    "formant shifting";
    "pitch modification"];
if(including_fixed_parameter)
    FixedParameter_WER = load(strcat(DataFolder, "CompiledResults/", "Fixed_Parameter_DeepSpeech.mat"));
    FixedParameter_MMR = load(strcat(DataFolder, "CompiledResults/", "Fixed_Parameter_Xvector.mat"));
    ModificationType = [ModificationType; "fixed parameters"];
    WER = [WER mean(FixedParameter_WER.WERScore)];
    ComputedMMR = [ComputedMMR; FixedParameter_MMR.ComputedEER];
end

%%

ModificationType = ModificationType(3:end);
for i =1:length(ModificationType)
    ModificationType(i) = strrep(ModificationType(i), "VoiceSecure_ablation_", "");
    ModificationType(i) = strrep(ModificationType(i), "_", " ");
end

WER = WER(3:end);
ComputedMMR = ComputedMMR(3:end);

VoiceSecure_WER = 52.77; % from librispeech_dev, x-vector mmr for voicesecure 
VoiceSecure_MMR = 32.42; % from librispeech_dev, deepspeech wer for voicesecure

%%
figure; hold on;
bar(ModificationType, WER);
yline(VoiceSecure_WER,':','VoiceSecure','LineWidth',3, 'FontSize',18);
ylabel("Word Error Rate (%)");
grid; grid minor;
set(gca, "FontSize", 20);
ylim([0 60])

outfile = "Plots/Ablation_WER.png";
%saveas(gcf, outfile);
%%

figure;
bar(ModificationType, ComputedMMR);
yline(VoiceSecure_MMR,':','VoiceSecure','LineWidth',3, 'FontSize',18);
ylabel("MisMatch Rate (%)");
grid; grid minor;
set(gca, "FontSize", 20);
ylim([0 40])
outfile = "Plots/Ablation_MMR.png";
%saveas(gcf, outfile);
