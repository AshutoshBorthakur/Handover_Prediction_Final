function handover_gui
    % ===========================================================
    % GUI for Predictive Handover Simulation and ML Evaluation
    % ===========================================================
    close all; clc;

    %% --- CREATE MAIN UI FIGURE ---------------------------------------
    fig = uifigure('Name', 'Predictive Handover GUI', 'Position', [100 100 1150 650]);

    % Panels
    ctrlPanel   = uipanel(fig, 'Title', 'Controls', 'Position', [20 350 320 280]);
    plotPanel   = uipanel(fig, 'Title', 'Plots', 'Position', [360 50 750 580]);
    outputPanel = uipanel(fig, 'Title', 'Output', 'Position', [20 50 320 280]);

    %% --- CONTROL BUTTONS ---------------------------------------------
    uilabel(ctrlPanel, 'Text', 'Predictive Handover Simulation', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Position', [40 230 250 25]);

    % Select Save Folder
    btnFolder = uibutton(ctrlPanel, 'Text', 'Select Save Folder', ...
        'FontSize', 11, 'Position', [60 220-50 180 35], ...
        'ButtonPushedFcn', @selectSaveFolder);

    % Run Simulation
    btnSim = uibutton(ctrlPanel, 'Text', 'Run Simulation', ...
        'FontSize', 11, 'Position', [60 220-100 180 35], ...
        'ButtonPushedFcn', @runSimulation);

    % Train Models
    btnTrain = uibutton(ctrlPanel, 'Text', 'Train Models', ...
        'FontSize', 11, 'Position', [60 220-150 180 35], ...
        'Enable', 'off', 'ButtonPushedFcn', @trainModels);

    % Status
    uilabel(ctrlPanel, 'Text', 'Status:', ...
        'Position', [20 50 60 25], 'HorizontalAlignment', 'left', 'FontWeight', 'bold');
    statusBox = uilabel(ctrlPanel, 'Text', 'Waiting...', ...
        'Position', [90 50 200 25], 'BackgroundColor', 'w', ...
        'HorizontalAlignment', 'left');

    %% --- PLOT AXES ----------------------------------------------------
    ax = uiaxes(plotPanel, 'Position', [50 220 630 320]);
    title(ax, 'Simulation Output');

    %% --- OUTPUT METRICS ----------------------------------------------
    uilabel(outputPanel, 'Text', 'Model Performance', ...
        'FontSize', 12, 'FontWeight', 'bold', 'Position', [70 240 200 25]);

    metricLabels = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
    metricValues = gobjects(1, numel(metricLabels));

    for i = 1:numel(metricLabels)
        uilabel(outputPanel, 'Text', metricLabels{i}, ...
            'Position', [40 240 - i*40 100 25], 'HorizontalAlignment', 'left');
        metricValues(i) = uilabel(outputPanel, 'Text', '-', ...
            'Position', [160 240 - i*40 100 25], 'FontWeight', 'bold');
    end

    % Shared data structure
    data = struct();
    data.saveFolder = pwd; % default to current working directory

    %% --- CALLBACK: Select Save Folder --------------------------------
    function selectSaveFolder(~,~)
        folder = uigetdir(pwd, 'Select Folder to Save Dataset');
        if ischar(folder) && folder ~= 0
            data.saveFolder = folder;
            statusBox.Text = sprintf('Save Folder:\n%s', folder);
        else
            statusBox.Text = 'Folder selection cancelled.';
        end
    end

    %% --- CALLBACK: Run Simulation ------------------------------------
    function runSimulation(~,~)
        statusBox.Text = 'Running simulation...';
        drawnow;

        % Parameters
        freqMHz = 900; hb = 30; hm = 1.5;
        areaSize = 2; numBS = 4; gridSize = 14;

        % Grid
        x = linspace(0, areaSize, gridSize);
        [yGrid, xGrid] = meshgrid(x, x);
        positions = [xGrid(:), yGrid(:)];
        numPoints = size(positions, 1);

        % Base stations
        rng(1);
        bsPos = areaSize * rand(numBS, 2);

        % Okumura–Hata model
        ahm = @(hm, f) (1.1*log10(f)-0.7)*hm - (1.56*log10(f)-0.8);
        Lhata = @(d) 69.55 + 26.16*log10(freqMHz) - 13.82*log10(hb) ...
            - ahm(hm, freqMHz) + (44.9 - 6.55*log10(hb)) .* log10(d);

        % RSSI simulation
        rssiData = zeros(numPoints, numBS);
        connectedBS = zeros(numPoints,1);
        for i = 1:numPoints
            for b = 1:numBS
                d = sqrt(sum((positions(i,:) - bsPos(b,:)).^2));
                d = max(d, 0.01);
                L = Lhata(d);
                txPower = 43;
                rssiData(i,b) = txPower - L;
            end
            [~, connectedBS(i)] = max(rssiData(i,:));
        end
        handover = [0; diff(connectedBS) ~= 0];

        % Create dataset
        varNames = [{'X_km','Y_km','ConnectedBS','Handover'}, ...
            arrayfun(@(i) sprintf('RSSI_BS%d',i), 1:numBS, 'UniformOutput', false)];
        dataset = array2table([positions, connectedBS, handover, rssiData], ...
            'VariableNames', varNames);

        % --- Save dataset to user-chosen folder ---
        filePath = fullfile(data.saveFolder, 'handover_dataset.csv');
        try
            writetable(dataset, filePath);
            statusBox.Text = sprintf('Simulation complete!\nSaved to:\n%s', filePath);
        catch ME
            warning(ME.message);
            altPath = fullfile(tempdir, 'handover_dataset.csv');
            writetable(dataset, altPath);
            statusBox.Text = sprintf('Saved to temp:\n%s', altPath);
        end

        btnTrain.Enable = 'on';
        data.dataset = dataset;

        % Visualization
        cla(ax);
        scatter(ax, positions(:,1), positions(:,2), 40, connectedBS, 'filled');
        hold(ax, 'on');
        scatter(ax, bsPos(:,1), bsPos(:,2), 120, 'kp', 'filled');
        title(ax, 'User Path and Connected BS');
        xlabel(ax, 'X (km)'); ylabel(ax, 'Y (km)');
        legend(ax, 'User Points', 'Base Stations');
        grid(ax, 'on');
    end

    %% --- CALLBACK: Train Models --------------------------------------
    function trainModels(~,~)
        if ~isfield(data, 'dataset')
            statusBox.Text = 'Run simulation first!';
            return;
        end
        dataset = data.dataset;
        statusBox.Text = 'Training models...'; drawnow;

        % Features and labels
        features = dataset(:, startsWith(dataset.Properties.VariableNames, 'RSSI') | ...
            strcmp(dataset.Properties.VariableNames, 'X_km') | strcmp(dataset.Properties.VariableNames, 'Y_km'));
        X = table2array(features);
        y = dataset.Handover;

        % Split
        cv = cvpartition(y, 'HoldOut', 0.3);
        X_train = X(training(cv), :);
        y_train = y(training(cv));
        X_test = X(test(cv), :);
        y_test = y(test(cv));

        % Logistic Regression
        logModel = fitglm(X_train, y_train, 'Distribution', 'binomial', 'Link', 'logit');
        y_pred_log = round(predict(logModel, X_test));

        % Random Forest
        rfModel = TreeBagger(100, X_train, y_train, ...
            'Method', 'classification', 'OOBPrediction', 'On', ...
            'OOBPredictorImportance', 'On');
        [y_pred_rf, ~] = predict(rfModel, X_test);
        y_pred_rf = str2double(y_pred_rf);

        % Evaluate metrics (using Random Forest)
        conf = confusionmat(y_test, y_pred_rf);
        acc = mean(y_pred_rf == y_test);
        prec = conf(2,2)/sum(conf(:,2));
        rec = conf(2,2)/sum(conf(2,:));
        f1 = 2*(prec*rec)/(prec+rec);

        % Update GUI
        metricValues(1).Text = sprintf('%.2f%%', acc*100);
        metricValues(2).Text = sprintf('%.2f', prec);
        metricValues(3).Text = sprintf('%.2f', rec);
        metricValues(4).Text = sprintf('%.2f', f1);
        statusBox.Text = 'Training complete!';

        % Plots
        figure('Name','Confusion Matrices');
        subplot(1,2,1);
        confusionchart(confusionmat(y_test, y_pred_log), {'No Handover','Handover'});
        title('Logistic Regression');
        subplot(1,2,2);
        confusionchart(conf, {'No Handover','Handover'});
        title('Random Forest');

        figure('Name','Feature Importance');
        bar(rfModel.OOBPermutedPredictorDeltaError);
        set(gca, 'XTickLabel', features.Properties.VariableNames, 'XTickLabelRotation', 45);
        title('Random Forest Feature Importance');
        ylabel('Δ Error'); grid on;
    end
end
