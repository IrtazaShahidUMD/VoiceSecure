%% Description
% This script plot the first evaluation figure
%%
clc; clear all; close all;
%%
Models = ["Original"; "Benign"; "McAdams"; "VoiceMask"; "Our System"];
MMR = [2 8.1 20.5 26.5 32.1];
WER = [6 8.2 28 48 52];
STOI = [100 98 84 61 73];
%%
% Define marker styles and colors for each model
markers = {"o", "s", "d", "^", "pentagram"};
colors = lines(length(Models));
markerSizes = 350*ones(1,length(MMR));
markerSizes(end) = 850;
% Color palette and markers
colors = lines(length(Models)); % Color scheme
markers = {'v', 's', 'd', '^', 'Hexagram'}; % Different marker shapes for each model
markers = {'v', 's', 'd', '^', 'o'}; % Different marker shapes for each model
markers = {'o', 'o', 'o', 'o', 'Pentagram'}; % Different marker shapes for each model

colors = [0.4660    0.6740    0.1880;
    0.8500    0.3250    0.0980;
    0         0.4470    0.7410;
    0.4940    0.1840    0.5560;
    0.9290    0.6940    0.1250;
    ];

figure;
hold on;
for i = 1:length(Models)
    scatter(MMR(i), WER(i), markerSizes(i), 'Marker', markers{i}, ...
            'MarkerEdgeColor', colors(i,:), 'MarkerFaceColor', colors(i,:), ...
            'DisplayName', sprintf('%s (STOI: %d%%)', Models(i), STOI(i)));
    text(MMR(i), WER(i) + 2.5, sprintf('%s', Models(i)), ...
        'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
        'FontSize', 18);
end
xlabel('Speaker MisMatch Rate (%)');
ylabel('Word Error Rate (%)');
set(gca, 'FontSize', 18);
grid on;
xlim([-3, max(MMR) + 5]);
ylim([0, max(WER) + 10]);
