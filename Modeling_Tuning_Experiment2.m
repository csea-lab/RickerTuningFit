%% Modeling tuning in Aversive Generalization Learning (Experiment 2)
% This is the execution code used to fitting two models in the
% detection accuracy data from an Aversive Generalization Learning paradigm
% Further information about the experimental design and methods can be find in: .......

% THE MODELS:
% Ricker Wavelet: The Ricker wavelet is a Difference-of-Gaussian function
% that calculates the amplitude of a tuning function using one free
% parameter (the standad deviation). Aside of the standard deviation, the
% Ricker function needs a k-vector [k1:kn] representing the number of conditions to
% be fitted.

% Morlet Wavelet: The Morlet Wavelet is a band-pass function typically used
% to obtain the oscillatory information from an electrophysiological
% signal. This wavelet can be used to calculate the amplitude of a tuning
% function. The input values for this wavele are the center frequency and
% the standard deviation. This function also needs a k-vector [k1:kn] representing the 
% number of conditions to be fitted.

%% RICKER on the Detection Accuracy data

% 1. LOAD THE BEHAVIORAL DATA:
load('Data_experiment2.mat') % Download data from OSF (https://osf.io/stzb6/)

% Determine the number of bootstraped samples
ndraws = 5000; 
% Inputs for the function 'nlinfit'
k_vector = -0.875:0.125:0.875;
initial_params = 0.1;
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');

% Initialize all iterative variables
BetaRicker_detection = nan(1,5000);
FitRicker_detection = nan(5000,15);
ResidualsRicker_detection = nan(5000,1);
ResidualsNull_detection = nan(5000,1);

% 2. FIT THE MODEL TO THE DATA:
% Fit the Ricker model in each bootstrapped sample and compute the
% bootstrapped resisual distributions for the Ricker wavelet and a Null model.
for draw = 1:ndraws
    
    % Sample with replacement the data
    Data4fit_dect = (rangecorrect(squeeze(mean(Data_experiment2(randi(9, 1,9), :), 1))));

    % Fitting Ricker using 'nlinfit'
    [BetaRicker_detection(draw), ~, ~, ~, ~] = nlinfit(k_vector,Data4fit_dect',@Ricker, initial_params, options);
    % Use each sample best fitting parameter to calculate the model residual (MSE)
    FitRicker_detection(draw,:) = Ricker(BetaRicker_detection(draw),k_vector);
    ResidualsRicker_detection(draw,:) = mean((FitRicker_detection(draw,:)-Data4fit_dect).^2);
    ResidualsNull_detection(draw,:) = mean(Data4fit_dect.^2);

end

% 3. EVALUATE THE FITTED MODEL:
% Use the bootstrapped residual distributions from the Ricker model and Null
% model to compute the Bayesian Bootstrapping (measure of goodness of fit

BF_Ricker_detection = bootstrap2BF_z(ResidualsRicker_detection,ResidualsNull_detection, 0);
% Transform BF10 to BF01 since we are evaluating whether the resisual the
% of Ricker model is smaller than the residual of a Null model
BF_Ricker_detection = 1./BF_Ricker_detection;
LogBFRicker_detection = log10(BF_Ricker_detection);

%% MORLET on the Detection Accuracy data

% Determine the number of bootstraped samples
ndraws = 5000; 
% Inputs for the function 'nlinfit'
k_vector = -7:7;
initial_params = [0.4 1];
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');

% Initialize all iterative variables
BetaMorlet_detection = nan(5000,2);
FitMorlet_detection = nan(5000,15);
ResidualsMorlet_detection = nan(5000,1);
ResidualsNull_detectionM = nan(5000,1);

% 1. FIT THE MODEL TO THE DATA:
% Fit the Morlet model in each bootstrapped sample and compute the
% bootstrapped resisual distributions for the Ricker wavelet and a Null model.
for draw = 1:ndraws

    % Sample with replacement the data
    Data4fit_dectM = (rangecorrect(squeeze(mean(Data_experiment2(randi(9, 1,9), :), 1))));

    % Fitting Morlet using 'nlinfit'
    [BetaMorlet_detection(draw,:), ~, ~, ~, ~] = nlinfit(k_vector,Data4fit_dectM',@TimeDomMorletWavelet, initial_params , options);
    % Use each sample best fitting parameter to calculate the model residual (MSE)
    FitMorlet_detection(draw,:) = TimeDomMorletWavelet(BetaMorlet_detection(draw,:), k_vector);
    ResidualsMorlet_detection(draw,:) = mean((FitMorlet_detection(draw,:)-Data4fit_dectM).^2);
    ResidualsNull_detectionM(draw,:) = mean(Data4fit_dectM.^2);
    
end

% 2. EVALUATE THE FITTED MODEL:
% Use the bootstrapped residual distributions from the Morlet model and Null
% model to compute the Bayesian Bootstrapping (measure of goodness of fit

BF_Morlet_detection = bootstrap2BF_z(ResidualsMorlet_detection,ResidualsNull_detectionM, 0);
% Transform BF10 to BF01 since we are evaluating whether the resisual the of Morlet
% model is smaller than the residual of a Null model
BF_Morlet_detection = 1./BF_Morlet_detection;
LogBF_Morlet_detection = log10(BF_Morlet_detection);

% Logarithmic Transitive Bayes Factors between Ricker BFs and Morlet BFs
LogTransitive_detection = log10(BF_Ricker_detection/BF_Morlet_detection);

%% Plotting Models prediction and Data

% 1. LOAD THE DATA:
load('Data_experiment2.mat')

data_detection = rangecorrect(squeeze(mean(Data_experiment2,1)));
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');
betaR = nlinfit(-0.875:0.125:0.875,data_detection',@Ricker, .1, options);
betaM = nlinfit((-7:7), data_detection', @TimeDomMorletWavelet, [0.4 1] , options);

figure,
plot(data_detection, 'LineWidth',3), hold on, 
plot(Ricker(betaR, -0.875:0.125:0.875)','LineWidth',3),
plot(TimeDomMorletWavelet(betaM, -7:7),'LineWidth',3),
legend('Original data','Ricker model','Morlet wavelet'),title('detection accuracy'), ylabel('accuracy'), xlabel('conditions')
xticks(1:15), xticklabels({'GS1' 'GS2' 'GS3' 'GS4' 'GS5' 'GS6' 'GS7' 'CS+' 'GS9' 'GS10' 'GS11' 'GS12' 'GS13' 'GS14' 'GS15'}), 
box off;h = gcf; set(h, 'PaperPositionMode','auto'); set(h, 'PaperOrientation', 'landscape');set(h, 'Position',[100 100 900 400]); 
ax = gca; ax.FontSize = 24;