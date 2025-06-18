%% Description
% This file compute and plot user study results
% Also compute p-test and report results
%%
clc; clear all; close all;
%% Path to UserStudy Data
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\"; 
DataPath = "Data2\UserStudy\";
%%
Files = dir(strcat(HomePath, DataPath, "*.mat"));
FinalResult = [];
for i = 1:length(Files)
    infile = strcat(HomePath, DataPath, Files(i).name);
    load(infile);
    Result = ComputeScore(Sim, Cla);
    FinalResult = [FinalResult; Result];
end
%%
RR = mean(FinalResult);
ModelNames = ["McAdams"; "VoiceMask"; "VocieSecure"];
SimScore = RR(1:3);
ClaScore = RR(4:end);
figure;
bar(ModelNames, SimScore)
ylabel("Speaker Confidence");
set(gca, "FontSize", 18);
grid on;

figure;
bar(ModelNames, ClaScore)
ylabel("Speech Clarity");
set(gca, "FontSize", 18);
grid on;
%% Computing paired t-test
[h1,p_McAdams_confidence] = ttest(FinalResult(:,1), FinalResult(:,3));
[h1,p_VoiceMask_confidence] = ttest(FinalResult(:,2), FinalResult(:,3));
[h1,p_McAdams_clarity] = ttest(FinalResult(:,4), FinalResult(:,6));
[h1,p_VoiceMask_clarity] = ttest(FinalResult(:,5), FinalResult(:,6));
clc;
disp("============ paired t-test =================");
disp(strcat("Confidence with McAdams: ", num2str(p_McAdams_confidence)));
disp(strcat("Confidence with VoiceMask: ", num2str(p_VoiceMask_confidence)));

disp(strcat("Clarity with McAdams: ", num2str(p_McAdams_clarity)));
disp(strcat("Clarity with VoiceMask: ", num2str(p_VoiceMask_clarity)));

%% Helper Functions
function [Result] = ComputeScore(Sim, Cla)
    Sim_VoiceMask = mean(Sim(1:3:15));
    Sim_VoiceSecure = mean(Sim(2:3:15));
    Sim_McAdams = mean(Sim(3:3:15));
    
    Clar_VoiceMask = mean(Cla(1:3:15));
    Clar_VoiceSecure = mean(Cla(2:3:15));
    Clar_McAdams = mean(Cla(3:3:15));

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % disp(strcat("Similarity Confidence"))
    % disp(strcat("McAdams: ", num2str(Sim_McAdams)))
    % disp(strcat("VoiceMask: ", num2str(Sim_VoiceMask)))
    % disp(strcat("VoiceSecure: ", num2str(Sim_VoiceSecure)))
    % 
    % disp(strcat("Clarity"))
    % disp(strcat("McAdams: ", num2str(Clar_McAdams)))
    % disp(strcat("VoiceMask: ", num2str(Clar_VoiceMask)))
    % disp(strcat("VoiceSecure: ", num2str(Clar_VoiceSecure)))
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Result = [Sim_McAdams Sim_VoiceMask Sim_VoiceSecure Clar_McAdams Clar_VoiceMask Clar_VoiceSecure];
end