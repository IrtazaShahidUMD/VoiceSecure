%%
clc; clear all; close all;
%%
HomePath = "D:\Irtaza\Data2\";
DataFolder = "train-clean-360\";
dataset = "train-clean-360\";
%%
ListOfFiles = dir(strcat(HomePath, DataFolder, dataset, "*\*\*.txt"));


% Name of the output file
outputFile = strcat(HomePath, DataFolder, "Label.txt");
% Open the output file for writing
fidOut = fopen(outputFile, 'w');

% Loop through each text file
for k = 1:length(ListOfFiles)
    % Get the full file name
    
    
    fullFileName = strcat(ListOfFiles(k).folder, "\", ListOfFiles(k).name);
    % Display the name of the file being processed
    fprintf('Processing file %s\n', fullFileName);
    
    % Open the current file for reading
    fidIn = fopen(fullFileName, 'r');
    
    % Read the contents of the file and write to the output file
    while ~feof(fidIn)
        line = fgets(fidIn); % Get a line from the file
        fprintf(fidOut, '%s', line); % Write it to the output file
    end
    
    % Close the current file
    fclose(fidIn);
end

% Close the output file
fclose(fidOut);

fprintf('All files have been combined into %s\n', outputFile);
