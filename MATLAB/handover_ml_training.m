%% ML Training for Predictive Handover Events
% Author: Ashutosh Borthakur
% Description:
%   Trains Logistic Regression and Random Forest models
%   to predict handover events using simulated RSSI and location data.

clear; clc; close all;

%% --- LOAD DATASET -----------------------------------------------------
data = readtable('handover_dataset.csv');

% Features: RSSI + Position
featureNames = data.Properties.VariableNames( ...
    startsWith(data.Properties.VariableNames,'RSSI') | ...
    strcmp(data.Properties.VariableNames,'X_km') | ...
    strcmp(data.Properties.VariableNames,'Y_km'));

X = data{:, featureNames};         % feature matrix
y = data.Handover;                 % labels (0 or 1)

%% --- TRAIN/TEST SPLIT -------------------------------------------------
cv = cvpartition(y, 'HoldOut', 0.3);
X_train = X(training(cv), :);
y_train = y(training(cv));
X_test  = X(test(cv), :);
y_test  = y(test(cv));

fprintf('Training samples: %d, Test samples: %d\n', length(y_train), length(y_test));

%% --- MODEL 1: LOGISTIC REGRESSION ------------------------------------
disp('Training Logistic Regression...');
logModel = fitglm(X_train, y_train, 'Distribution', 'binomial', 'Link', 'logit');

% Predict probabilities and classes
y_pred_log = round(predict(logModel, X_test));

% Evaluate
acc_log = mean(y_pred_log == y_test);
confMat_log = confusionmat(y_test, y_pred_log);

% Display results
fprintf('\n--- Logistic Regression Results ---\n');
disp(confMat_log);
fprintf('Accuracy: %.2f%%\n', 100*acc_log);

precision_log = confMat_log(2,2) / sum(confMat_log(:,2));
recall_log    = confMat_log(2,2) / sum(confMat_log(2,:));
f1_log        = 2*(precision_log*recall_log)/(precision_log+recall_log);
fprintf('Precision: %.2f  Recall: %.2f  F1: %.2f\n', precision_log, recall_log, f1_log);

%% --- MODEL 2: RANDOM FOREST (TREEBAGGER) -----------------------------
disp('Training Random Forest...');
numTrees = 100;
rfModel = TreeBagger(numTrees, X_train, y_train, ...
    'Method', 'classification', ...
    'OOBPrediction', 'On', ...
    'OOBPredictorImportance', 'On', ... % <-- Added this line
    'NumPredictorsToSample', 'all');

% Predict on test data
[y_pred_rf, scores_rf] = predict(rfModel, X_test);
y_pred_rf = str2double(y_pred_rf); % convert cell to double

% Evaluate
acc_rf = mean(y_pred_rf == y_test);
confMat_rf = confusionmat(y_test, y_pred_rf);

fprintf('\n--- Random Forest Results ---\n');
disp(confMat_rf);
fprintf('Accuracy: %.2f%%\n', 100*acc_rf);

precision_rf = confMat_rf(2,2) / sum(confMat_rf(:,2));
recall_rf    = confMat_rf(2,2) / sum(confMat_rf(2,:));
f1_rf        = 2*(precision_rf*recall_rf)/(precision_rf+recall_rf);
fprintf('Precision: %.2f  Recall: %.2f  F1: %.2f\n', precision_rf, recall_rf, f1_rf);

%% --- VISUALIZATION ---------------------------------------------------
% Confusion matrices
figure;
subplot(1,2,1);
confusionchart(confMat_log, {'No Handover','Handover'});
title('Logistic Regression Confusion Matrix');

subplot(1,2,2);
confusionchart(confMat_rf, {'No Handover','Handover'});
title('Random Forest Confusion Matrix');

% Feature importance for Random Forest
figure;
bar(rfModel.OOBPermutedPredictorDeltaError);
set(gca, 'XTickLabel', featureNames, 'XTickLabelRotation', 45);
title('Random Forest Feature Importance');
ylabel('Predictor Importance (Δ Error)');
grid on;

