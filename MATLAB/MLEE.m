%% Predictive Handover Simulation using Okumura–Hata Model
% Author: Ashutosh Borthakur
% Description:
%   Simulates a mobile user moving through a grid covered by multiple
%   base stations. Computes RSSI values using Okumura–Hata model
%   and records handover events as the user moves.

clear; clc; close all;

%% --- PARAMETERS --------------------------------------------------------
freqMHz   = 900;      % carrier frequency (MHz)
hb        = 30;       % base station antenna height (m)
hm        = 1.5;      % mobile antenna height (m)
areaSize  = 2;        % km × km area
numBS     = 4;        % number of base stations
gridSize  = 14;       % user grid resolution per axis (14x14 = 196 points)

%% --- GENERATE USER POSITIONS ------------------------------------------
x = linspace(0, areaSize, gridSize);
[yGrid, xGrid] = meshgrid(x, x);
positions = [xGrid(:), yGrid(:)];
numPoints = size(positions, 1);

%% --- DEPLOY BASE STATIONS ---------------------------------------------
rng(1);
bsPos = areaSize * rand(numBS, 2); % random BS locations (km)

%% --- DEFINE Okumura–Hata MODEL ---------------------------------------
ahm = @(hm, freqMHz) (1.1*log10(freqMHz) - 0.7)*hm - (1.56*log10(freqMHz) - 0.8);
okumuraHataLoss = @(d_km) 69.55 + 26.16*log10(freqMHz) - ...
    13.82*log10(hb) - ahm(hm, freqMHz) + ...
    (44.9 - 6.55*log10(hb)) .* log10(d_km);

%% --- SIMULATE USER MOTION & SIGNALS -----------------------------------
rssiData = zeros(numPoints, numBS);
connectedBS = zeros(numPoints,1);

for i = 1:numPoints
    userPos = positions(i, :);
    for b = 1:numBS
        d = sqrt(sum((userPos - bsPos(b,:)).^2)); % distance in km
        d = max(d, 0.01); % avoid log(0)
        L = okumuraHataLoss(d);
        txPower_dBm = 43;  % 20 W typical BS transmit power
        rssiData(i,b) = txPower_dBm - L; % received power (dBm)
    end
    [~, connectedBS(i)] = max(rssiData(i,:)); % strongest BS
end

%% --- CREATE HANDOVER LABELS ------------------------------------------
handoverEvent = [0; diff(connectedBS) ~= 0]; % 1 if handover occurred

%% --- CREATE DATASET TABLE --------------------------------------------
varNames = [{'X_km','Y_km','ConnectedBS','Handover'}, ...
    arrayfun(@(i) sprintf('RSSI_BS%d',i), 1:numBS, 'UniformOutput', false)];
dataset = array2table([positions, connectedBS, handoverEvent, rssiData], ...
    'VariableNames', varNames);

%% --- VISUALIZE RESULTS -----------------------------------------------
figure;
scatter(positions(:,1), positions(:,2), 40, connectedBS, 'filled');
hold on;
scatter(bsPos(:,1), bsPos(:,2), 120, 'kp', 'filled');
text(bsPos(:,1)+0.05, bsPos(:,2), ...
    arrayfun(@(i)sprintf('BS%d',i),1:numBS,'UniformOutput',false), 'FontWeight','bold');
xlabel('X (km)'); ylabel('Y (km)');
title('User Path and Connected Base Stations');
colorbar; grid on;

figure;
plot(1:numPoints, connectedBS, 'LineWidth', 1.5);
xlabel('User Step'); ylabel('Connected BS');
title('Handover Events Over Time');
grid on;

%% --- SAVE DATASET -----------------------------------------------------
writetable(dataset, 'handover_dataset.csv');
disp(['Dataset saved as handover_dataset.csv with ', num2str(numPoints), ' samples.']);
