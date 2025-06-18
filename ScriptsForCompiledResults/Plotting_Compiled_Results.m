%% Description
% This file plot compiled results for dataset over three different speaker
% and three different asr models
%%
clc; clear all; close all;
%%
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\"; 
DataPath = "Data2\";

DataFolder = "LibriSpeech_Dev\";

DataFolder = strcat(HomePath, DataPath, DataFolder);
CompiledResults = "CompiledResults\";

UseIndices = [1 2 3 4 5];
ModificationName = ["Original"; "Noise30"; "McAdams"; "VoiceMask";  "VoiceSecure"];
TypeName = ["Original"; "Benign"; "McAdams"; "VoiceMask";  "VoiceSecure"];
%% Plotting Speaker Results
SpeakerModels = ["Xvector"; "ECAPA"; "Ivector";];
PlotSpeakerResults(DataFolder, SpeakerModels, CompiledResults, UseIndices, ModificationName, TypeName);
%% Plotting WER Results
ASRModels = ["Whisper"; "DeepSpeech"; "Wav2Vec2"];
PlotASRResults(DataFolder, ASRModels, CompiledResults, UseIndices, ModificationName, TypeName);

%% Plotting Intelligibility Results
QualityModels = ["Stoi";];
PlotQualityResults(DataFolder, QualityModels, CompiledResults, UseIndices, ModificationName, TypeName)
%% Helper Functions

function PlotSpeakerResults(DataFolder, SpeakerModels, CompiledResults, UseIndices, ModificationName, TypeName)
Modifications = [];
EER_Results = [];
MMR_Results = [];

for m = 1:length(SpeakerModels)
    infile = strcat(DataFolder, CompiledResults, SpeakerModels(m), ".mat");
    load(infile);
    EER_Results = [EER_Results ComputedEER];
    MMR_Results = [MMR_Results ComputedMMR];
    Modifications = [Modifications ModificationList2];
end

ModificationList = ModificationList2(UseIndices);
EER_Results = EER_Results(UseIndices, :);
MMR_Results = MMR_Results(UseIndices, :);

figure;
set(gcf, 'Position', [100, 100, 650, 450]);
bar(SpeakerModels, EER_Results')
grid; grid minor;
ylabel("Equal Error Rate (%)"); 

set(gca, "FontSize", 16);
legend(TypeName, 'FontSize', 12,  'Location','northwest');
ylim([0 30]);

figure;
set(gcf, 'Position', [100, 100, 650, 450]);
bar(SpeakerModels, MMR_Results')
grid; grid minor;
ylabel("MisMatch Rate (%)"); 

set(gca, "FontSize", 16);
legend(TypeName, 'FontSize', 12,  'Location','northwest');
ylim([0 50]);
end
%%
function PlotASRResults(DataFolder, ASRModels, CompiledResults, UseIndices, ModificationName, TypeName)
Modifications = [];
WER_Results = [];
for m = 1:length(ASRModels)
    infile = strcat(DataFolder, CompiledResults, ASRModels(m), ".mat");
    load(infile);

    WER_Results = [WER_Results mean(WERScore)'];
    Modifications = [Modifications ModificationType];
end
ModificationList = ModificationType(UseIndices);
WER_Results = WER_Results(UseIndices, :);

figure;
set(gcf, 'Position', [100, 100, 650, 450]);
bar(ASRModels, WER_Results')
grid; grid minor;
ylabel("Word Error Rate (%)");
legend(TypeName, 'Location','northwest');
set(gca, "FontSize", 16);
end
%%
function PlotQualityResults(DataFolder, QualityModels, CompiledResults, UseIndices, ModificationName, TypeName)
Modifications = [];
Quality_Results = [];
for m = 1:length(QualityModels)
    infile = strcat(DataFolder, CompiledResults, QualityModels(m), ".mat");
    load(infile);
    Quality_Results = [Quality_Results mean(AllScore)'];
    Modifications = [Modifications ModificationType];
end
ModificationList = ModificationType(UseIndices);
Quality_Results = Quality_Results(UseIndices, :);
QualityModels = ["Speech Intelligibility"];
figure;
set(gcf, 'Position', [100, 100, 650, 450]);
bar(QualityModels, 100*Quality_Results')
grid; grid minor;
ylabel("Percentage (%)");
legend(TypeName,  'Location','southeast');
set(gca, "FontSize", 16);
end