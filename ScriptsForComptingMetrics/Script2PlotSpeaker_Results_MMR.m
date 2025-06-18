%% Description
% This script take computed embeddings from {ModelUse}Embeddings2 directory
% of the corresponding dataset. Compute Speaker Mismatch and Equal Error
% Rate of all modifications in the ModificationList relative to the
% Original speech and store the compiled results for all modification in 
% CompiledResults directory in DataFolder
%%
clc; clear all; close all;
%% DataPath
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\Data2\";
DataFolder = "LibriSpeech_Dev\"; % select dataset
Original = "Original";
ModificationList = ["Original"; "VoiceSecure"];

ModelUse = "Xvector";
%ModelUse = "ECAPA";
%ModelUse = "Ivector";
DataFolder = strcat(HomePath, DataFolder);
ResultsFolder = strcat(ModelUse, "Embeddings2\");
%% Loading Enrolled Data


ModificationList2 = ModificationList;

[OriginalData] = LoadData(DataFolder, ResultsFolder, Original);
ComputedEER = zeros(size(ModificationList));
ComputedMMR = zeros(size(ModificationList));
ComputedAccuracy = zeros(size(ModificationList));
for i = 1:length(ModificationList)
    Modification = ModificationList(i);
    titletxt = strcat(Modification, " ",ModelUse);
    [ModifiedData] = LoadData(DataFolder, ResultsFolder, Modification);
    [~, ~, SimilarityScore] = ComputeMetricScore(OriginalData, ModifiedData);
    [Sim_SameSpeakerScore, Sim_DifferentSpeakerScore] = SeparateSameAndDifferentSpeakers(SimilarityScore,OriginalData.SpeakerNames);
    Sim_DifferentSpeakerScore = randsample(Sim_DifferentSpeakerScore, length(Sim_SameSpeakerScore));
    [SelectedThresholdValue, Results] = PlotEERCurves(Sim_SameSpeakerScore, Sim_DifferentSpeakerScore, titletxt);
    ComputedEER(i) = Results.EER;
    ComputedMMR(i) = Results.MMR;
    ComputedAccuracy(i) = Results.Accuracy;
    disp("================================================================")
    disp(strcat("ModificationType:",Modification));
    disp(strcat("EER:", num2str(Results.EER), ", Selected Threshold:",num2str(SelectedThresholdValue)));
    disp(strcat("Acc:", num2str(Results.Accuracy), ", FAR:", num2str(Results.FAR), ", FRR:", num2str(Results.FRR), ", MMR:", num2str(Results.MMR)            )   );
    disp("================================================================")
end

figure;
bar(ModificationList2,ComputedEER);
grid minor;
ylim([0 25]);
ylabel("Equal Error Rate"); xlabel("Modification Type");
sgtitle(ModelUse);

figure;

bar(ModificationList2,ComputedMMR);
grid minor;
ylim([0 25]);
ylabel("MisMatch Rate"); xlabel("Modification Type");
titletxt = strcat("Speaker Verification (", ModelUse, ")");
sgtitle(ModelUse);



CompiledResults = strcat(DataFolder, "CompiledResults\", ModelUse, ".mat");
%save(CompiledResults, "ComputedEER", "ComputedMMR", "ModificationList2");
%% Helper functions
function [Data] = LoadData(DataFolder, ResultsFolder, Modification)
    infile = strcat(DataFolder, ResultsFolder, Modification, ".mat");
    load(infile);
    SpeakerNames = squeeze(SpeakerNames);
    [SpeakerNames] = ConvertCharacterListToStrings(SpeakerNames);
    [FileNames] = ConvertCharacterListToStrings(FileNames);
    Data.Embeddings = Embeddings;
    Data.SpeakerNames = SpeakerNames;
    Data.FileNames = FileNames;
    disp(strcat(Modification, " -> Embeddings: (",num2str(size(Embeddings)), ") -> ", "SpeakerNames: ", num2str(length(SpeakerNames))));
end
%%
function [SpkNames] = ConvertCharacterListToStrings(SpeakerNames)
    SpkNames = [];
    for i= 1:size(SpeakerNames, 1)
        SpkNames = [SpkNames; convertCharsToStrings(SpeakerNames(i,:))];
    end
end
%%
function [DistanceScore, AngleScore, SimilarityScore] = ComputeMetricScore(OriginalData, ModifiedData)

N = size(OriginalData.Embeddings,1);
SimilarityScore = zeros(N,N);
DistanceScore = 0;
AngleScore = 0;

f = waitbar(0, 'Starting');
for i = 1:N
    waitbar(i/N, f, sprintf('Progress: %d %%', floor(i)));
    for j = 1:N
        if(i == j)
            SimilarityScore(i,j) = nan;
            continue;
        end
        Vec1 = OriginalData.Embeddings(i,:);
        Vec2 = ModifiedData.Embeddings(j,:);
        SimilarityScore(i,j) = dot(Vec1,Vec2)/(norm(Vec1)*norm(Vec2));
    end

end
close(f);
end
%%
function [SameSpeakerScore, DifferentSpeakerScore] = SeparateSameAndDifferentSpeakers(MetricScore,SpeakerNames)
UniqueSpeakers = unique(SpeakerNames);
SameSpeakerScore = [];
DifferentSpeakerScore = [];
for SpeakerNum = 1:length(UniqueSpeakers)
    SameSpeakers = SpeakerNames == UniqueSpeakers(SpeakerNum);
    DiffSpeakers = SpeakerNames ~= UniqueSpeakers(SpeakerNum);
    SS = MetricScore(SameSpeakers,SameSpeakers);
    DS = MetricScore(SameSpeakers,DiffSpeakers);
    SS = SS(:);
    DS = DS(:);
    % because daigonla has nans
    SS = SS(~isnan(SS));
    DS = DS(~isnan(DS));
    SameSpeakerScore = [SameSpeakerScore; SS];
    DifferentSpeakerScore = [DifferentSpeakerScore; DS];
end
end
%%
function [SelectedThresholdValue, Results] = PlotEERCurves(SameSpeakerScore, DifferentSpeakerScore, titletxt)
Threshold = [-1:0.01:1];
AccuracyResult = [];
TPRResult = [];
FARResult = [];
FRRResult = [];
TNRResult = [];
MMRResult = [];

for i = 1:length(Threshold)
    ThresholdValue = Threshold(i);

    SS_Predictions = SameSpeakerScore >= ThresholdValue;
    DS_Predictions = DifferentSpeakerScore >= ThresholdValue;

    TP = sum(SS_Predictions);

    FN = sum(SS_Predictions == 0);

    FP = sum(DS_Predictions);

    TN = sum(DS_Predictions == 0);

    Accuracy = 100*(TP + TN)/(TP + FP + FN + TN);
    TPR = 100*TP/(TP + FN);
    FAR = 100*FP/(FP + TN);
    FRR = 100*FN/(FN + TP);
    TNR = 100*TN/(TN + FP);
    
    mrr = FRR + FAR;
    AccuracyResult = [AccuracyResult Accuracy];
    TPRResult = [TPRResult TPR];
    FARResult = [FARResult FAR];
    FRRResult = [FRRResult FRR];
    TNRResult = [TNRResult TNR];
    MMRResult = [MMRResult mrr];
end

[v, thresh_indx] = min(abs(FARResult - FRRResult));

SelectedThresholdValue = Threshold(thresh_indx);
Results.Accuracy = AccuracyResult(thresh_indx);
Results.FAR = FARResult(thresh_indx);
Results.FRR = FRRResult(thresh_indx);
Results.EER = (Results.FAR + Results.FRR) / 2;
Results.MMR = MMRResult(thresh_indx);
end