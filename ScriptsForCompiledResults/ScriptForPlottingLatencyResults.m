%% Description
% This file creates a cdf plot for the latency results of VoiceSecure
%%
clc; clear all; close all;
%%
HomePath = "D:\Irtaza\VoiceSecure_Artifacts\"; 
DataPath = "Data2\";
%%
DataFolder = strcat(HomePath, DataPath, "Latency_Results/");
Filename = "Test";
NumFiles = 3;
L = [];
for i = 1:NumFiles
    infile = strcat(DataFolder, Filename, num2str(i), ".mat");
    load(infile);
    L = [L; Latency_Calculations];
end
%%
h = cdfplot(L);
set(h, 'LineWidth', 4);
hold on;
xline(150/1000, 'r:', 'LineWidth', 4);
xlabel('Latency (sec)');
ylabel('CDF');
title('');
grid on; grid minor;
xlim([20/1000 170/1000])
legend({'VoiceSecure', 'Acceptable Latency'});
set(gca, 'FontSize', 18);