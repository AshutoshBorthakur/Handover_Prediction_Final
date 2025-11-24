function handover_gui
    % ===========================================================
    % Predictive Handover GUI - Fixed & Improved Version
    % Fixes: Removed 'lon/lat' error, proper plotting, no overwrites
    % ===========================================================
    close all; clc;

    %% --- CREATE MAIN UI FIGURE ---------------------------------------
    fig = uifigure('Name', 'Predictive Handover GUI', ...
        'Position', [100 100 1150 650], 'CloseRequestFcn', @onClose);

    % Panels
    ctrlPanel   = uipanel(fig, 'Title', 'Controls', 'Position', [20 350 320 280]);
    plotPanel   = uipanel(fig, 'Title', 'Plots', 'Position', [360 50 750 580]);
    %outputPanel = uipanel(fig, 'Title', 'Output Metrics', 'Position', [20 50 320 280]);

    %% --- CONTROL BUTTONS ---------------------------------------------
    uilabel(ctrlPanel, 'Text', 'Predictive Handover Simulation', ...
        'FontSize', 13, 'FontWeight', 'bold', 'Position', [40 230 240 30]);

    btnFolder = uibutton(ctrlPanel, 'Text', 'Select Save Folder', ...
        'Position', [60 170 180 35], 'ButtonPushedFcn', @selectSaveFolder);

    btnSim = uibutton(ctrlPanel, 'Text', 'Run Simulation', ...
        'Position', [60 120 180 35], 'ButtonPushedFcn', @runSimulation);

    btnTrain = uibutton(ctrlPanel, 'Text', 'Train & Evaluate Model', ...
        'Position', [60 70 180 35], 'Enable', 'off', 'ButtonPushedFcn', @trainModels);

    % Status
    uilabel(ctrlPanel, 'Text', 'Status:', 'Position', [20 20 60 25], ...
        'FontWeight', 'bold', 'HorizontalAlignment', 'left');
    statusBox = uilabel(ctrlPanel, 'Text', 'Ready.', ...
        'Position', [90 10 210 50], 'BackgroundColor', 'white', ...
        'HorizontalAlignment', 'left', 'WordWrap', 'on');

    %% --- PLOT AXES ---------------------------------------------------
    axSim = uiaxes(plotPanel, 'Position', [70 340 650 220]);
    title(axSim, 'Simulation: User Path and Connected BS');
    xlabel(axSim, 'X (km)'); ylabel(axSim, 'Y (km)');

    axModel = uiaxes(plotPanel, 'Position', [70 80 650 220]);
    title(axModel, 'Model Results - No prediction yet');
    xlabel(axModel, 'X (km)'); ylabel(axModel, 'Y (km)');

    %% --- OUTPUT METRICS ----------------------------------------------
    %uilabel(outputPanel, 'Text', 'Python XGBoost Model Performance', ...
     %   'FontSize', 12, 'FontWeight', 'bold', 'Position', [20 230 280 25]);

    %metricLabels = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
    %metricValues = gobjects(1,4);
    %yPos = 180;
    %for i = 1:4
     %   uilabel(outputPanel, 'Text', [metricLabels{i} ':'], ...
     %       'Position', [40 yPos 100 25], 'HorizontalAlignment', 'left');
     %   metricValues(i) = uilabel(outputPanel, 'Text', '-', ...
     %       'Position', [150 yPos 120 25], 'FontWeight', 'bold', 'FontSize', 12);
     %   yPos = yPos - 40;
    %end

    %% --- SHARED DATA -------------------------------------------------
    data = struct();
    data.saveFolder = pwd;
    data.dataset = [];
    data.bsPos = [];
    data.areaSize = 2;

    % Load Python model (handover_model.pkl must be in same folder)
    pyModel = [];
    if exist('handover_model.pkl', 'file')
        try
            pyModel = py.joblib.load('handover_model.pkl');
            statusBox.Text = 'Python XGBoost model loaded successfully.';
        catch ME
            statusBox.Text = 'Error loading model.';
            warning(sprintf("Could not load Python model: %s", ME.message));

        end
    else
        statusBox.Text = 'handover_model.pkl not found!';
    end

    %% --- CALLBACKS ---------------------------------------------------

    function selectSaveFolder(~,~)
        folder = uigetdir(pwd, 'Select Folder to Save Dataset');
        if folder ~= 0
            data.saveFolder = folder;
            statusBox.Text = sprintf('Save folder set:\n%s', folder);
        end
    end

    function runSimulation(~,~)
        statusBox.Text = 'Running simulation...'; drawnow;

        % Parameters
        freqMHz = 900; hb = 30; hm = 1.5;
        areaSize = 2; numBS = 4; gridSize = 14;
        data.areaSize = areaSize;

        % Grid points
        x = linspace(0, areaSize, gridSize);
        [xGrid, yGrid] = meshgrid(x, x);
        positions = [xGrid(:), yGrid(:)];

        % Base stations (fixed seed for reproducibility)
        rng(42);
        bsPos = areaSize * [0.2 0.2; 0.8 0.3; 0.7 0.8; 0.3 0.7];
        data.bsPos = bsPos;

        % Okumura-Hata path loss
        ahm = @(f) (1.1*log10(f)-0.7)*hm - (1.56*log10(f)-0.8);
        Lhata = @(d) 69.55 + 26.16*log10(freqMHz) - 13.82*log10(hb) ...
            - ahm(freqMHz) + (44.9 - 6.55*log10(hb)) .* log10(max(d,0.01));

        % Compute RSSI and best BS
        rssiData = zeros(size(positions,1), numBS);
        connectedBS = zeros(size(positions,1),1);

        for i = 1:size(positions,1)
            for b = 1:numBS
                d = norm(positions(i,:) - bsPos(b,:));
                rssiData(i,b) = 43 - Lhata(d);  % TxPower = 43 dBm
            end
            [~, connectedBS(i)] = max(rssiData(i,:));
        end

        handover = [false; diff(connectedBS) ~= 0];

        % Create dataset
        varNames = [{'X_km','Y_km','ConnectedBS','Handover'}, ...
            arrayfun(@(i) sprintf('RSSI_BS%d',i), 1:numBS, 'UniformOutput', false)];
        dataset = array2table([positions, connectedBS, double(handover), rssiData], ...
            'VariableNames', varNames);

        % Save
        filePath = fullfile(data.saveFolder, 'handover_dataset.csv');
        try
            writetable(dataset, filePath);
            statusBox.Text = sprintf('Simulation done!\nSaved:\n%s', filePath);
        catch
            filePath = fullfile(pwd, 'handover_dataset.csv');
            writetable(dataset, filePath);
            statusBox.Text = sprintf('Saved to current folder:\n%s', filePath);
        end

        data.dataset = dataset;
        btnTrain.Enable = 'on';

        % --- Plot Simulation (Connected BS) ---
        cla(axSim);
        scatter(axSim, positions(:,1), positions(:,2), 50, connectedBS, 'filled');
        hold(axSim, 'on');
        scatter(axSim, bsPos(:,1), bsPos(:,2), 200, 'kp', 'filled', ...
            'MarkerFaceColor', 'yellow', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        hold(axSim, 'off');
        title(axSim, 'Simulation: Connected Base Station per Grid Point');
        xlabel(axSim, 'X (km)'); ylabel(axSim, 'Y (km)');
        colormap(axSim, lines(numBS));
        colorbar(axSim, 'Ticks', 1:numBS, 'TickLabels', arrayfun(@(x)sprintf('BS%d',x),1:numBS,'UniformOutput',false));
        grid(axSim, 'on'); axis(axSim, [0 areaSize 0 areaSize]);
        legend(axSim, 'Grid Points', 'Base Stations', 'Location', 'northwest');

        % --- Placeholder for Model Results ---
        cla(axModel);
        title(axModel, 'Model Results - Click "Train & Evaluate Model"');
        text(axModel, 0.5, 0.5, 'No prediction yet', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontSize', 16, 'Color', [0.6 0.6 0.6]);
        axis(axModel, [0 areaSize 0 areaSize]);
        grid(axModel, 'on');
    end

    function trainModels(~,~)
        if isempty(data.dataset)
            statusBox.Text = 'Run simulation first!';
            return;
        end

        statusBox.Text = 'Evaluating Python XGBoost model...'; drawnow;

        dataset = data.dataset;

        % --- Map to real-world-like features (same as Python training) ---
        lat0 = 15.3920; lon0 = 73.8810;
        lat = lat0 + dataset.Y_km * 0.009;   % rough km → degree
        lon = lon0 + dataset.X_km * 0.009;

        % Extract strongest RSSI (the one from connected BS)
        rssi = zeros(height(dataset),1);
        for i = 1:height(dataset)
            bs = dataset.ConnectedBS(i);
            rssi(i) = dataset{i, 4 + bs};  % columns 5–8 are RSSI_BS1..4
        end

        % Synthetic features (same distribution as real data)
        speed    = 2 + 8*rand(height(dataset),1);      % 2–10 m/s
        altitude = -25 + 15*rand(height(dataset),1);    % -25 to -10 m

        X_py = [lon, lat, rssi, speed, altitude];
        y_true = dataset.Handover;

        if isempty(pyModel)
            statusBox.Text = 'Python model not available!';
            return;
        end

        % --- Predict on ALL points for visualization ---
        py_X_all = py.numpy.array(X_py);
        py_pred_all = pyModel.predict(py_X_all);
        y_pred_all = double(py_pred_all);

        % --- Train/test split for metrics ---
        cv = cvpartition(y_true, 'HoldOut', 0.3);
        idx_train = training(cv);
        idx_test  = test(cv);

        py_X_test = py.numpy.array(X_py(idx_test,:));
        py_pred_test = pyModel.predict(py_X_test);
        y_pred_test = double(py_pred_test);
        y_test = y_true(idx_test);

        % --- Metrics ---
        conf = confusionmat(y_test, y_pred_test);
        tp = conf(2,2); fp = conf(1,2); fn = conf(2,1); tn = conf(1,1);
        acc  = (tp + tn) / sum(conf(:));
        prec = tp / (tp + fp); if isnan(prec), prec = 0; end
        rec  = tp / (tp + fn); if isnan(rec),  rec  = 0; end
        f1   = 2*prec*rec/(prec+rec); if isnan(f1), f1 = 0; end

        % Update GUI
        metricValues(1).Text = sprintf('%.2f%%', acc*100);
        metricValues(2).Text = sprintf('%.3f', prec);
        metricValues(3).Text = sprintf('%.3f', rec);
        metricValues(4).Text = sprintf('%.3f', f1);

        % --- Plot Predicted Handover Regions ---
        cla(axModel);
        scatter(axModel, dataset.X_km, dataset.Y_km, 50, y_pred_all, 'filled');
        hold(axModel, 'on');
        scatter(axModel, data.bsPos(:,1), data.bsPos(:,2), 200, 'kp', 'filled', ...
            'MarkerFaceColor', 'yellow', 'MarkerEdgeColor', 'k', 'LineWidth', 1.5);
        hold(axModel, 'off');

        title(axModel, 'Predicted Handover Regions (Python XGBoost)');
        xlabel(axModel, 'X (km)'); ylabel(axModel, 'Y (km)');
        colormap(axModel, [0.2 0.8 0.2; 0.9 0.2 0.2]);  % Green = No HO, Red = HO
        cbar = colorbar(axModel, 'Ticks', [0.25 0.75], 'TickLabels', {'No Handover', 'Handover'});
        grid(axModel, 'on'); axis(axModel, [0 data.areaSize 0 data.areaSize]);

        % --- Confusion Matrix in separate window ---
        %fig_cm = figure('Name', 'Confusion Matrix - Python XGBoost', 'NumberTitle', 'off');
        %confusionchart(y_test, y_pred_test, 'RowSummary','row-normalized', ...
         %   'ColumnSummary','column-normalized');
        %title('Python XGBoost on Simulated Data');

        statusBox.Text = sprintf('Evaluation complete!');
    end

    function onClose(~,~)
        delete(fig);
    end
end