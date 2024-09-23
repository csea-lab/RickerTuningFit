%% Modeling tuning in Aversive Generalization Learning (Experiment 1)
% This is the execution code used to fitting two models in the
% electrophysiological data (ssVEP and alpha-band) from an Aversive
% Generalization Learning paradigm. Further information about the
% experimental design and methods can be find in: .......

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

%% RICKER on the ssVEP data

% The ssVEP amplitude was computed by using a Discrete Fourier
% Transformation (DFT) in each artifact-free trial. The dimensions of the
% following data represents: [electrodes,subjects,conditions].
% Conditions = Phase x Condition (3 x 4) = 12

% 1. LOAD THE SSVEP DATA:
load('Data_experiment1_ssvep.mat') % Download data from OSF (https://osf.io/stzb6/)

% Determine the number of bootstraped samples
ndraws = 5000; 
% Inputs for the function 'nlinfit'
k_vector = 0.25:0.25:1;
initial_params = 0.1;
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');
% Initialize all iterative variables
BetaRicker_ssvep_acq = nan(5000,129);
FitRicker_ssvep_acq = nan(5000,129,4);
ResidualsRicker_ssvep_acq = nan(5000,129);
ResidualsNull_ssvep_acq = nan(5000,129);
BetaRicker_ssvep_ext = nan(5000,129);
FitRicker_ssvep_ext = nan(5000,129,4);
ResidualsRicker_ssvep_ext = nan(5000,129);
ResidualsNull_ssvep_ext = nan(5000,129);

% 2. FIT THE MODEL TO THE DATA:
% Fit the Ricker model in each bootstrapped sample and compute the
% bootstrapped resisual distributions for the Ricker wavelet and a Null model.
for draw = 1:ndraws
    
    % Sample with replacement the data from the Acquisition phase and run a range correction
    data4fit_ssvep_acq = rangecorrect(squeeze(mean(Data_experiment1_ssvep(:, randi(31, 1,31), 5:8), 2)));
    % Sample with replacement the data from the Extinction phase and run a range correction
    data4fit_ssvep_ext = rangecorrect(squeeze(mean(Data_experiment1_ssvep(:, randi(31, 1,31), 9:12), 2)));

    for elec = 1:size(Data_experiment1_ssvep,1) % Loop over the electrodes
    % Fitting Ricker in the Acquisition phase:

        % Fitting Ricker using 'nlinfit'
        [BetaRicker_ssvep_acq(draw,elec), ~, ~, ~, ~] = nlinfit(k_vector,data4fit_ssvep_acq(elec,:)',@Ricker, initial_params, options);
        % Use each sample best fitting parameter to calculate the model residual (MSE)
        FitRicker_ssvep_acq(draw,elec,:) = Ricker(BetaRicker_ssvep_acq(draw,elec),k_vector); 
        ResidualsRicker_ssvep_acq(draw,elec,:) = mean((squeeze(FitRicker_ssvep_acq(draw,elec,:))'-data4fit_ssvep_acq(elec,:)).^2);
        ResidualsNull_ssvep_acq(draw,elec,:) = mean((data4fit_ssvep_acq(elec,:)).^2);

    % Fitting Ricker in the Extinction phase:
    
        % Fitting Ricker using 'nlinfit'
        [BetaRicker_ssvep_ext(draw,elec), ~, ~, ~, ~] = nlinfit(k_vector,data4fit_ssvep_ext(elec,:)',@Ricker, initial_params, options);
        % Use each sample best fitting parameter to calculate the model residual (MSE)
        FitRicker_ssvep_ext(draw,elec,:) = Ricker(BetaRicker_ssvep_ext(draw,elec),k_vector);
        ResidualsRicker_ssvep_ext(draw,elec,:) = mean((squeeze(FitRicker_ssvep_ext(draw,elec,:))'-data4fit_ssvep_ext(elec,:)).^2);
        ResidualsNull_ssvep_ext(draw,elec,:) = mean((data4fit_ssvep_ext(elec,:)).^2);

    end

end

% 3. EVALUATE THE FITTED MODEL:
% Use the bootstrapped residual distributions from the Ricker model and Null
% model to compute the Bayesian Bootstrapping (measure of goodness of fit)

% Initialize all iterative variables
BF_Ricker_ssvep_acq = nan(1,129);
BF_Ricker_ssvep_ext = nan(1,129);

for chan = 1:size(Data_experiment1_ssvep,1) % Loop over the electrodes

    BF_Ricker_ssvep_acq(chan) = bootstrap2BF_z(ResidualsRicker_ssvep_acq(:,chan),ResidualsNull_ssvep_acq(:,chan), 0);
    BF_Ricker_ssvep_ext(chan) = bootstrap2BF_z(ResidualsRicker_ssvep_ext(:,chan),ResidualsNull_ssvep_ext(:,chan), 0);

end

% Transform BF10 to BF01 since we are evaluating whether the resisual the of Ricker
% model is smaller than the residual of a Null model
BF_Ricker_ssvep_acq = 1./BF_Ricker_ssvep_acq;
LogBF_Ricker_ssvep_acq = log10(BF_Ricker_ssvep_acq);
BF_Ricker_ssvep_ext = 1./BF_Ricker_ssvep_ext;
LogBF_Ricker_ssvep_ext =log10(BF_Ricker_ssvep_ext);

%% RICKER on the Alpha-band data

% Alpha power for each phase was obtained by convoluting a Morlet wavelet
% (DO NOT CONFUSE WITH THE MORLET WAVELET APPLIED TO THE RESPONSES) against the
% artifact-free EEG trials. The data was later avergaged across the
% frequency bands from 9.33 to 11.33 Hz, and accross the time window:
% 500-1300 ms. 
% For each phase the dimension of the data is: [electrodes,subjects,conditions].

% 1. LOAD THE ALPHA DATA:
load('Data_experiment1_alpha.mat') % Download data from OSF (https://osf.io/stzb6/)

% Determine the number of bootstraped samples
ndraws = 5000; 
% Inputs for the function 'nlinfit'
k_vector = 0.25:0.25:1;
initial_params = 0.1;
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');
% Initialize all iterative variables
BetaRicker_alpha_acq = nan(5000,129);
FitRicker_alpha_acq = nan(5000,129,4);
ResidualsRicker_alpha_acq = nan(5000,129);
ResidualsNull_alpha_acq = nan(5000,129);
BetaRicker_alpha_ext = nan(5000,129);
FitRicker_alpha_ext = nan(5000,129,4);
ResidualsRicker_alpha_ext = nan(5000,129);
ResidualsNull_alpha_ext = nan(5000,129);


% 2. FIT THE MODEL TO THE DATA:
% Fit the Ricker model in each bootstrapped sample and compute the
% bootstrapped resisual distributions for the Ricker wavelet and a Null model.
for draw = 1:ndraws

    % Sample with replacement the data from the Acquisition phase and run a range correction
    data4fit_alpha_acq = (rangecorrect(squeeze(mean(Data_alpha_acq(:, randi(31, 1,31), :), 2))))*-1; % sign inverted to run the model
    % Sample with replacement the data from the Extinction phase and run a range correction
    data4fit_alpha_ext = (rangecorrect(squeeze(mean(Data_alpha_ext(:, randi(31, 1,31), :), 2))))*-1; % sign inverted to run the model
    
    for elec = 1:size(Data_alpha_acq,1) % Loop over the electrodes
    % Fitting Ricker in the Acquisition phase:

        % Fitting Ricker using 'nlinfit'
        [BetaRicker_alpha_acq(draw,elec), ~, ~, ~, ~] = nlinfit(k_vector,data4fit_alpha_acq(elec,:)',@Ricker, initial_params, options);
        % Use each sample best fitting parameter to calculate the model residual (MSE)
        FitRicker_alpha_acq(draw,elec,:) = Ricker(BetaRicker_alpha_acq(draw,elec),k_vector);
        ResidualsRicker_alpha_acq(draw,elec,:) = mean((squeeze(FitRicker_alpha_acq(draw,elec,:))'-data4fit_alpha_acq(elec,:)).^2);
        ResidualsNull_alpha_acq(draw,elec,:) = mean((data4fit_alpha_acq(elec,:)).^2);

    % Fitting Ricker in the Acquisition phase:

        % Fitting Ricker using 'nlinfit'
        [BetaRicker_alpha_ext(draw,elec), ~, ~, ~, ~] = nlinfit(k_vector,data4fit_alpha_ext(elec,:)',@Ricker, initial_params, options);
        % Use each sample best fitting parameter to calculate the model residual (MSE)
        FitRicker_alpha_ext(draw,elec,:) = Ricker(BetaRicker_alpha_ext(draw,elec),k_vector);
        ResidualsRicker_alpha_ext(draw,elec,:) = mean((squeeze(FitRicker_alpha_ext(draw,elec,:))'-data4fit_alpha_ext(elec,:)).^2);
        ResidualsNull_alpha_ext(draw,elec,:) = mean((data4fit_alpha_ext(elec,:)).^2);

    end

end

% 3. EVALUATE THE FITTED MODEL:
% Use the bootstrapped residual distributions from the Ricker model and Null
% model to compute the Bayesian Bootstrapping (measure of goodness of fit)

% Initialize all iterative variables
BF_Ricker_alpha_acq = nan(1,129);
BF_Ricker_alpha_ext = nan(1,129);
for chan = 1:size(Data_alpha_acq,1) % Loop over the electrodes
    BF_Ricker_alpha_acq(chan) = bootstrap2BF_z(ResidualsRicker_alpha_acq(:,chan),ResidualsNull_alpha_acq(:,chan), 0);
    BF_Ricker_alpha_ext(chan) = bootstrap2BF_z(ResidualsRicker_alpha_ext(:,chan),ResidualsNull_alpha_ext(:,chan), 0);

end

% Transform BF10 to BF01 since we are evaluating whether the resisual the of Ricker
% model is smaller than the residual of a Null model
BF_Ricker_alpha_acq = 1./BF_Ricker_alpha_acq;
LogBF_Ricker_alpha_acq = log10(BF_Ricker_alpha_acq);
BF_Ricker_alpha_ext = 1./BF_Ricker_alpha_ext;
LogBF_Ricker_alpha_ext = log10(BF_Ricker_alpha_ext);

%% MORLET on the ssVEP data

% The fitting of the Morlet wavelet is done in the same ssVEP data loaded
% above. In this section, only sensor 74 is used to fit the Morlet in
% acquisition, and sensor 75 was used to fit the model in extinction (see
% manuscript for more information).

% Determine the number of bootstraped samples
ndraws = 5000; 
% Inputs for the function 'nlinfit'
k_vector = 1:4;
initial_params = [0.12 4];
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');
% Initialize all iterative variables
BetaMorlet_ssvep_acq = nan(5000,2);
FitMorlet_ssvep_acq = nan(5000,4);
ResidualsMorlet_ssvep_acq = nan(5000,1);
ResidualsNull_ssvep_acqM = nan(5000,1);
BetaMorlet_ssvep_ext = nan(5000,2);
FitMorlet_ssvep_ext = nan(5000,4);
ResidualsMorlet_ssvep_ext = nan(5000,1);
ResidualsNull_ssvep_extM = nan(5000,1);

% 1. FIT THE MODEL TO THE DATA:
% Fit the Morlet model in each bootstrapped sample and compute the
% bootstrapped resisual distributions for the Morlet wavelet and a Null model.
for draw = 1:ndraws
    
    % Sample with replacement the data from the Acquisition phase and run a
    % range correction for electrode 74 
    data4fit_ssvep_acqM = rangecorrect(squeeze(mean(Data_experiment1_ssvep(74, randi(31, 1,31), 5:8), 2)));
    % Sample with replacement the data from the Extinction phase and run a
    % range correction for electrode 75
    data4fit_ssvep_extM = rangecorrect(squeeze(mean(Data_experiment1_ssvep(75, randi(31, 1,31), 9:12), 2)));
    
    % Fitting Morlet in the Acquisition phase:

        % Fitting Morlet using 'nlinfit'
        [BetaMorlet_ssvep_acq(draw,:), ~, ~, ~, ~] = nlinfit(k_vector,data4fit_ssvep_acqM',@TimeDomMorletWavelet, initial_params, options);
        % Use each sample best fitting parameter to calculate the model residual (MSE)
        FitMorlet_ssvep_acq(draw,:) = TimeDomMorletWavelet(BetaMorlet_ssvep_acq(draw,:),k_vector); 
        ResidualsMorlet_ssvep_acq(draw,:) = mean((squeeze(FitMorlet_ssvep_acq(draw,:))'-data4fit_ssvep_acqM).^2);
        ResidualsNull_ssvep_acqM(draw,:) = mean(data4fit_ssvep_acqM.^2);

    % Fitting Morlet in the Extinction phase:
    
        % Fitting Morlet using 'nlinfit'
        [BetaMorlet_ssvep_ext(draw,:), ~, ~, ~, ~] = nlinfit(k_vector,data4fit_ssvep_extM',@TimeDomMorletWavelet, initial_params, options);
        % Use each sample best fitting parameter to calculate the model residual (MSE)
        FitMorlet_ssvep_ext(draw,:) = TimeDomMorletWavelet(BetaMorlet_ssvep_ext(draw,:),k_vector);
        ResidualsMorlet_ssvep_ext(draw,:) = mean((squeeze(FitMorlet_ssvep_ext(draw,:))'-data4fit_ssvep_extM).^2);
        ResidualsNull_ssvep_extM(draw,:) = mean(data4fit_ssvep_extM.^2);

end

% 2. EVALUATE THE FITTED MODEL:
% Use the bootstrapped residual distributions from the Morlet model and Null
% model to compute the Bayesian Bootstrapping (measure of goodness of fit)

BF_Morlet_ssvep_acq = bootstrap2BF_z(ResidualsMorlet_ssvep_acq,ResidualsNull_ssvep_acqM, 0);
BF_Morlet_ssvep_ext = bootstrap2BF_z(ResidualsMorlet_ssvep_ext,ResidualsNull_ssvep_extM, 0);


% Transform BF10 to BF01 since we are evaluating whether the resisual the of Morlet
% model is smaller than the residual of a Null model
BF_Morlet_ssvep_acq = 1./BF_Morlet_ssvep_acq;
LogBF_Morlet_ssvep_acq = log10(BF_Morlet_ssvep_acq);
BF_Morlet_ssvep_ext = 1./BF_Morlet_ssvep_ext;
LogBF_Morlet_ssvep_ext =log10(BF_Morlet_ssvep_ext);

%% MORLET on the Alpha-band data

% The fitting of the Morlet wavelet is done in the same Alpha-band data loaded
% above. In this section, the information from a right parietal bundle of electrodes 
% were use to fit the model (see manuscript for more information).

% Select the right parietal bundle of electrode to fit the models in
% acquisition and extinction
Data_alpha_acq_rparietal = Data_alpha_acq([77 78 84 85 90 91],:,:);
Data_alpha_ext_rparietal = Data_alpha_ext([77 78 84 85 90 91],:,:);

% Determine the number of bootstraped samples
ndraws = 5000; 
% Inputs for the function 'nlinfit'
k_vector = 1:4;
initial_params = [0.12 4];
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');
% Initialize all iterative variables
BetaMorlet_alpha_acq = nan(5000,6,2);
FitMorlet_alpha_acq = nan(5000,6,4);
ResidualsMorlet_alpha_acq = nan(5000,6);
ResidualsNull_alpha_acqM = nan(5000,6);
BetaMorlet_alpha_ext = nan(5000,6,2);
FitMorlet_alpha_ext = nan(5000,6,4);
ResidualsMorlet_alpha_ext = nan(5000,6);
ResidualsNull_alpha_extM = nan(5000,6);

% 1. FIT THE MODEL TO THE DATA:
% Fit the Morlet model in each bootstrapped sample and compute the
% bootstrapped resisual distributions for the Morlet wavelet and a Null model.
for draw = 1:ndraws
    
    % Sample with replacement the data from the Acquisition phase 
    data4fit_alpha_acqM = rangecorrect(squeeze(mean(Data_alpha_acq_rparietal(:,randi(31, 1,31), :), 2)))*-1; % sign inverted to run the model
    % Sample with replacement the data from the Extinction phase 
    data4fit_alpha_extM = rangecorrect(squeeze(mean(Data_alpha_acq_rparietal(:,randi(31, 1,31), :), 2)))*-1; % sign inverted to run the model

    for elec = 1:size(Data_alpha_acq_rparietal,1) % Loop over the electrodes
    % Fitting Morlet in the Acquisition phase:

        % Fitting Morlet using 'nlinfit'
        [BetaMorlet_alpha_acq(draw,elec,:), ~, ~, ~, ~] = nlinfit(k_vector,data4fit_alpha_acqM(elec,:)',@TimeDomMorletWavelet, initial_params, options);
        % Use each sample best fitting parameter to calculate the model residual (MSE)
        FitMorlet_alpha_acq(draw,elec,:) = TimeDomMorletWavelet(BetaMorlet_alpha_acq(draw,elec,:),k_vector); 
        ResidualsMorlet_alpha_acq(draw,elec,:) = mean((squeeze(FitMorlet_alpha_acq(draw,elec,:))'-data4fit_alpha_acqM(elec,:)).^2);
        ResidualsNull_alpha_acqM(draw,elec,:) = mean((data4fit_alpha_acqM(elec,:)).^2);

    % Fitting Morlet in the Extinction phase:
    
        % Fitting Morlet using 'nlinfit'
        [BetaMorlet_alpha_ext(draw,elec,:), ~, ~, ~, ~] = nlinfit(k_vector,data4fit_alpha_extM(elec,:)',@TimeDomMorletWavelet, initial_params, options);
        % Use each sample best fitting parameter to calculate the model residual (MSE)
        FitMorlet_alpha_ext(draw,elec,:) = TimeDomMorletWavelet(BetaMorlet_alpha_ext(draw,elec,:),k_vector);
        ResidualsMorlet_alpha_ext(draw,elec,:) = mean((squeeze(FitMorlet_alpha_ext(draw,elec,:))'-data4fit_alpha_extM(elec,:)).^2);
        ResidualsNull_alpha_extM(draw,elec,:) = mean((data4fit_alpha_extM(elec,:)).^2);
    end

end

% 2. EVALUATE THE FITTED MODEL:
% Use the bootstrapped residual distributions from the Morlet model and Null
% model to compute the Bayesian Bootstrapping (measure of goodness of fit)

% Initialize all iterative variables
BF_Morlet_alpha_acq = nan(1,6);
BF_Morlet_alpha_ext = nan(1,6);

for chan = 1:size(Data_alpha_acq_rparietal,1) % Loop over the electrodes
    BF_Morlet_alpha_acq(chan) = bootstrap2BF_z(ResidualsMorlet_alpha_acq(:,chan),ResidualsNull_alpha_acqM(:,chan), 0);
    BF_Morlet_alpha_ext(chan) = bootstrap2BF_z(ResidualsMorlet_alpha_ext(:,chan),ResidualsNull_alpha_extM(:,chan), 0);
end


% Transform BF10 to BF01 since we are evaluating whether the resisual the of Morlet
% model is smaller than the residual of a Null model
BF_Morlet_alpha_acq = 1./BF_Morlet_alpha_acq;
LogBF_Morlet_alpha_acq = log10(BF_Morlet_alpha_acq);
BF_Morlet_alpha_ext = 1./BF_Morlet_alpha_ext;
LogBF_Morlet_alpha_ext =log10(BF_Morlet_alpha_ext);

%% Transitive Bayes Factors between Ricker BFs an Morlet BFs

% To contrast the goodness of fit between the Ricker wavelet and the Morlet
% wavelet, compute the logarithmic transitive BF between the Bayes Factors
% of both models.

% Transitive Bayes Factors in the ssVEP data at electrode 74 in Ecquisition
% and 75 in Extinction
LogTransitive_ssvep_acq = log10(BF_Ricker_ssvep_acq(:,74)/BF_Morlet_ssvep_acq);
LogTransitive_ssvep_ext = log10(BF_Ricker_ssvep_ext(:,75)/BF_Morlet_ssvep_ext);

% Transitive Bayes Factors in the alpha-band data across the right parietal
% group of electrodes
LogTransitive_alpha_acq = log10(mean(BF_Ricker_alpha_acq(:,[77 78 84 85 90 91]))/mean(BF_Morlet_alpha_acq));
LogTransitive_alpha_ext = log10(mean(BF_Ricker_alpha_ext(:,[77 78 84 85 90 91]))/mean(BF_Morlet_alpha_ext));


%% Plotting Models prediction and Data

% Plot ssVEP in the acquisition phase
% 1. LOAD THE SSVEP DATA:
load('Data_experiment1_ssvep.mat') 

data_ssvep_acq_s74 = rangecorrect(squeeze(mean(Data_experiment1_ssvep(74, :, 5:8), 2)));
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');
betaR_ssvep_acq = nlinfit(0.25:0.25:1,data_ssvep_acq_s74,@Ricker, 0.1, options);
betaM_ssvep_acq = nlinfit(1:4,data_ssvep_acq_s74,@TimeDomMorletWavelet, [0.12 4], options);

figure,
plot(data_ssvep_acq_s74, 'LineWidth',3), hold on, 
plot(Ricker(betaR_ssvep_acq, 0.25:0.25:1)','LineWidth',3),
plot(TimeDomMorletWavelet(betaM_ssvep_acq, 1:4),'LineWidth',3),
legend('Original data','Ricker model','Morlet wavelet'), title('ssVEP (acquisition)'), ylabel('amplitude'), xlabel('conditions')
xticks([1 2 3 4]), xticklabels({'CS+' 'GS1' 'GS2' 'GS3'}),box off
ax = gca; ax.FontSize = 24;

% Plot ssVEP in the extinction phase

data_ssvep_ext_s75 = rangecorrect(squeeze(mean(Data_experiment1_ssvep(75, :, 9:12), 2)));
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');
betaR_ssvep_ext = nlinfit(0.25:0.25:1,data_ssvep_ext_s75,@Ricker, 0.1, options);
betaM_ssvep_ext = nlinfit(1:4,data_ssvep_ext_s75,@TimeDomMorletWavelet, [0.12 4], options);

figure,
plot(data_ssvep_ext_s75, 'LineWidth',3), hold on, 
plot(Ricker(betaR_ssvep_ext, 0.25:0.25:1)','LineWidth',3),
plot(TimeDomMorletWavelet(betaM_ssvep_ext, 1:4),'LineWidth',3),
legend('Original data','Ricker model','Morlet wavelet'), title('ssVEP (extinction)'), ylabel('amplitude'), xlabel('conditions')
xticks([1 2 3 4]), xticklabels({'CS+' 'GS1' 'GS2' 'GS3'}), box off
ax = gca; ax.FontSize = 24;

% Plot alpha-band in the acquisition phase
% 1. LOAD THE ALPHA DATA:
load('Data_experiment1_alpha.mat')
Data_alpha_acq_rparietal = squeeze(mean(Data_alpha_acq([77 78 84 85 90 91],:,:),1));

d_alpha_acq = rangecorrect(squeeze(mean(Data_alpha_acq_rparietal(:,:), 1)))*-1;
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');
betaR_alpha_acq = nlinfit(0.25:0.25:1,d_alpha_acq',@Ricker, 0.1, options);
betaM_alpha_acq = nlinfit(1:4,d_alpha_acq',@TimeDomMorletWavelet, [0.12 4], options);

figure,
plot(d_alpha_acq, 'LineWidth',3), hold on, 
plot(Ricker(betaR_alpha_acq, 0.25:0.25:1)','LineWidth',3),
plot(TimeDomMorletWavelet(betaM_alpha_acq, 1:4),'LineWidth',3),
legend('Original data','Ricker model','Morlet wavelet'), title('alpha (acquisition)'), ylabel('amplitude'), xlabel('conditions')
xticks([1 2 3 4]), xticklabels({'CS+' 'GS1' 'GS2' 'GS3'}),box off
ax = gca; ax.FontSize = 24;

% Plot alpha-band in the extinction phase
load('Data_experiment1_alpha.mat')
Data_alpha_ext_rparietal = squeeze(mean(Data_alpha_ext([77 78 84 85 90 91],:,:),1));

d_alpha_ext = rangecorrect(squeeze(mean(Data_alpha_ext_rparietal(:,:), 1)))*-1;
options = statset('MaxIter', 100000, 'TolFun', 0.0001,  'Robust', 'off');
betaR_alpha_ext = nlinfit(0.25:0.25:1,d_alpha_ext',@Ricker, 0.1, options);
betaM_alpha_ext = nlinfit(1:4,d_alpha_ext',@TimeDomMorletWavelet, [0.12 4], options);

figure,
plot(d_alpha_ext, 'LineWidth',3), hold on, 
plot(Ricker(betaR_alpha_ext, 0.25:0.25:1)','LineWidth',3),
plot(TimeDomMorletWavelet(betaM_alpha_ext, 1:4),'LineWidth',3),
legend('Original data','Ricker model','Morlet wavelet'), title('alpha (extinction)'), ylabel('amplitude'), xlabel('conditions')
xticks([1 2 3 4]), xticklabels({'CS+' 'GS1' 'GS2' 'GS3'}),box off
ax = gca; ax.FontSize = 24;