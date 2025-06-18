%% Description
% This file plot adversarial study results for both mismatch rate and word
% error rate.
%%
clc; clear all; close all;
%% Path to DataSets
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\"; 
DataPath = "Data2\";
%%
DataFolder = strcat(HomePath, DataPath);
CompiledResults = "CompiledResults/";

DataSets_ASV = ["LibriSpeech_Dev"; "VoxCeleb_Dev"; "VCTK"];
SpeakerModels = ["Xvector"; "ECAPA"; "Ivector";];
Results = zeros(length(DataSets_ASV), length(SpeakerModels), 3);
xaxis_label = SpeakerModels;
%%
for i = 1:length(DataSets_ASV)
    for j = 1:length(SpeakerModels)
        infile = strcat(DataFolder, DataSets_ASV(i),"/", CompiledResults, "Adversarial_", SpeakerModels(j), ".mat");
        load(infile);
        Final_MMR = ComputedMMR;
        Results(i,j, :) = Final_MMR;
    end
end
%%
legend_labels = ["Original vs. Original";
                 "Original vs. VoiceSecure";
                 "VoiceSecure vs. VoiceSecure";];
for i = 1:length(SpeakerModels)
    temp = squeeze(Results(i,:,:));
    figure;
    set(gcf, 'Position', [100, 100, 650, 450]);
    bar(SpeakerModels, temp);
    grid; grid minor;
    legend(legend_labels);
    ylabel("MisMatch Rate (%)");
    set(gca, "FontSize", 20);
    ylim([0 100]);
    legend('Location', 'northwest', 'FontSize',20);
    outfile = strcat("Plots/","Adversarial_MMR_", DataSets_ASV(i),".png");
    %saveas(gcf, outfile);
end
%%
DataSets_ASR = ["CommonVoice"; "VCTK"];
Files = ["Original"; "VoiceSecure"; "Fine_Tuned"];

DATA = zeros(length(DataSets_ASR), length(Files));
for i = 1:length(DataSets_ASR)
    for j = 1:length(Files)
        infile = strcat(HomePath, DataPath, DataSets_ASR(i), "/", "WhisperWERScores2/", Files(j), ".mat");
        load(infile);
        DATA(i,j) = mean(Wer1);
    end
end
%%
legend_labels = ["Whisper on clean speech";
                "Whisper on VoiceSecure";
                "Finetuned Whisper on VoiceSecure"];

figure;
set(gcf, 'Position', [100, 100, 650, 450]);
bar(DataSets_ASR, DATA');
grid; grid minor;
legend(legend_labels);
ylabel("Word Error Rate (%)");
legend('Location', 'northwest', 'FontSize',20);
set(gca, "FontSize", 20);
ylim([0 100]);
outfile = strcat("Plots/","Adversarial_WER",".png");
%saveas(gcf, outfile);