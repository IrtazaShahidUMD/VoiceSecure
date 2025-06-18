%%
clc; clear all; close all;
%%
HomePath = "D:\Irtaza\Data2\";
DataFolder = "train-clean-360\";
dataset = "train-clean-360\";
%%
ListOfFiles = dir(strcat(HomePath, DataFolder, dataset, "*\*\*.flac"));
N_Before = length(ListOfFiles);
for i = 1:length(ListOfFiles)
    disp(strcat(num2str(i), "/", num2str(N_Before)));
    infile = strcat(ListOfFiles(i).folder, "\", ListOfFiles(i).name);
    outfile = strrep(infile, "flac", "wav");
    [data, Fs] = audioread(infile);
    audiowrite(outfile, data, Fs);
    delete(infile);
end
ListOfFiles = dir(strcat(HomePath, DataFolder, dataset, "*\*\*.wav"));
N_After = length(ListOfFiles);
disp(strcat("Number of Flac Files: ", num2str(N_Before)));
disp(strcat("Number of Wav Files: ", num2str(N_After)));
