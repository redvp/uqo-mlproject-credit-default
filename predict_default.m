%% Projet
clc; clear; close all

%% Prétraitement des données

% Chargement des données
credit_default = readtable("default_clients.xls");

% L'ID n'est pas une caractéristique à prendre en compte
credit_default = removevars(credit_default, "ID");

% Suppression des données qui ne correspondent pas à la description
credit_default = credit_default(~(credit_default.EDUCATION == 5 ...
    | credit_default.EDUCATION == 6 ...
    | credit_default.EDUCATION == 0 ...
    | credit_default.MARRIAGE == 0),:);

% Attributs catégoriques
marriage = categorical(credit_default.MARRIAGE);
marriage = table(marriage);
marriage = onehotencode(marriage);
marriage = renamevars(marriage, {'1','2','3'},...
    {'MARRIED','SINGLE','COMPLICATED'});

% Stockage des attributs sans correction à appliquer
sex = table(credit_default.SEX, 'VariableNames',{'SEX'});
education = table(credit_default.EDUCATION, 'VariableNames',{'EDUCATION'});
default = table(credit_default.defaultPaymentNextMonth,...
    'VariableNames',{'defaultPayment'});

% On les retire de la table
credit_default = removevars(credit_default, {'SEX','EDUCATION',...
    'MARRIAGE', 'defaultPaymentNextMonth'});

% Correction
data_matrix = table2array(credit_default);
data_matrix = data_matrix + normrnd(0,0.005,size(data_matrix));

% Normalisation, division par valeur max
credit_default_normalized = normalize([array2table(data_matrix,...
    'VariableNames',credit_default.Properties.VariableNames) education], 'range');

% Reconstitution du jeu de données complet
credit_default = [credit_default_normalized sex marriage default];

%% Split entre données d'entrainement et données de test

% Sélection des données de la classe défaut
default_tbl = credit_default(credit_default.defaultPayment == 1,:);
% Sélection des données de la classe non-défaut
non_default_tbl = credit_default(credit_default.defaultPayment == 0,:);

% Split données d'entrainement et de test pour la classe défaut
cvp_default = cvpartition(default_tbl.defaultPayment,...
    'Holdout', 0.5);
default_train_ind = training(cvp_default);
default_test_ind = test(cvp_default);
default_train = default_tbl(default_train_ind,:);
default_test = default_tbl(default_test_ind,:);
% Split données d'entrainement et de test pour la classe non-défaut
cvp_non_default = cvpartition(non_default_tbl.defaultPayment,...
    'Holdout', height(non_default_tbl) - height(default_train));
non_default_train_ind = training(cvp_non_default);
non_default_test_ind = test(cvp_non_default);
non_default_train = non_default_tbl(non_default_train_ind,:);
non_default_test = non_default_tbl(non_default_test_ind,:);

% Données d'entrainement
training_data = vertcat(non_default_train, default_train);
% Données de test
testing_data = vertcat(non_default_test, default_test);

%% Classification en defaut/non-defaut

% Validation croisée avec KFold sur les données d'entrainement
cvp_kfold = cvpartition(training_data.defaultPayment,...
    'KFold', 3, 'Stratify', true);

% Valeurs actuelles à prédire sur les données de test
y_test = testing_data.defaultPayment;

% Matrices de confusion
figure(1);

% Réseau de neurones
rnn_model = fitcnet(training_data, 'defaultPayment','CVPartition', cvp_kfold);
rnn_mean_val_error = kfoldLoss(rnn_model);
[~,best_index] = min(kfoldLoss(rnn_model, 'Mode', 'individual'));
best_rnn_mdl = rnn_model.Trained{best_index,1};
[rnn_predictions,rnn_scores] = predict(best_rnn_mdl, testing_data);
rnn_test_error = sum(abs(y_test - rnn_predictions)/height(testing_data));
[rnnX,rnnY,rnnT,rnnAUC] = perfcurve(y_test,max(rnn_scores, [], 2),1);
subplot(231);
confusionchart(y_test,rnn_predictions);
title('Reseau de neurones');

% Classificateur de Bayes
bayes_model = fitcnb(training_data, 'defaultPayment','CVPartition', cvp_kfold);
bayes_mean_val_error = kfoldLoss(bayes_model);
[~,best_index] = min(kfoldLoss(bayes_model, 'Mode', 'individual'));
best_bayes_mdl = bayes_model.Trained{best_index,1};
[bayes_predictions,bayes_scores,bayes_cost] = predict(best_bayes_mdl,testing_data);
bayes_test_error = sum(abs(y_test - bayes_predictions)/height(testing_data));
[bayesX,bayesY,bayesT,bayesAUC] = perfcurve(y_test,max(bayes_scores, [], 2),1);
subplot(232);
confusionchart(y_test,bayes_predictions);
title('Bayes');

% KPPV
kppv_model = fitcknn(training_data, 'defaultPayment',...
    'NumNeighbors',3,'CVPartition', cvp_kfold);
kppv_mean_val_error = kfoldLoss(kppv_model);
[~,best_index] = min(kfoldLoss(kppv_model, 'Mode', 'individual'));
best_kppv_mdl = kppv_model.Trained{best_index,1};
[kppv_predictions,kppv_scores,kppv_cost] = predict(best_kppv_mdl, testing_data);
kppv_test_error = sum(abs(y_test - kppv_predictions)/height(testing_data));
[kppvX,kppvY,kppvT,kppvAUC] = perfcurve(y_test,max(kppv_scores, [], 2),1);
subplot(233);
confusionchart(y_test,kppv_predictions);
title('KPPV');

% Arbre de décision
dt_model = fitctree(training_data, 'defaultPayment','CVPartition', cvp_kfold);
dt_mean_val_error = kfoldLoss(dt_model);
[~,best_index] = min(kfoldLoss(dt_model, 'Mode', 'individual'));
best_dt_mdl = dt_model.Trained{best_index,1};
[dt_predictions,dt_scores,dt_cost] = predict(best_dt_mdl, testing_data);
dt_test_error = sum(abs(y_test - dt_predictions)/height(testing_data));
[dtX,dtY,dtT,dtAUC] = perfcurve(y_test,max(dt_scores, [], 2),1);
subplot(234);
confusionchart(y_test,dt_predictions);
title('Arbre de decision');

% Régression logistique
reglog_model = fitclinear(training_data, 'defaultPayment',...
    'CVPartition', cvp_kfold, 'Learner','logistic');
reglog_mean_val_error = kfoldLoss(reglog_model);
[~,best_index] = min(kfoldLoss(reglog_model, 'Mode', 'individual'));
best_reglog_mdl = reglog_model.Trained{best_index,1};
[reglog_predictions,reglog_scores] = predict(best_reglog_mdl, testing_data);
reglog_test_error = sum(abs(y_test - reglog_predictions)/height(testing_data));
[reglogX,reglogY,reglogT,reglogAUC] = perfcurve(y_test,max(reglog_scores, [], 2),1);
subplot(235);
confusionchart(y_test,reglog_predictions);
title('Regression logistique');

figure(2);
subplot(231)
plot(kppvX,kppvY)
xlabel('False positive rate') 
ylabel('True positive rate')
title(sprintf('KPPV - AUC de %.2f', kppvAUC))
subplot(232)
plot(dtX,dtY)
xlabel('False positive rate') 
ylabel('True positive rate')
title(sprintf('DT - AUC de %.2f', dtAUC))
subplot(233)
plot(bayesX,bayesY)
xlabel('False positive rate') 
ylabel('True positive rate')
title(sprintf('Bayes - AUC de %.2f', bayesAUC))
subplot(234)
plot(rnnX,rnnY)
xlabel('False positive rate') 
ylabel('True positive rate')
title(sprintf('RNN - AUC de %.2f', rnnAUC))
subplot(235)
plot(reglogX,reglogY)
xlabel('False positive rate') 
ylabel('True positive rate')
title(sprintf('RegLog - AUC de %.2f', reglogAUC))

figure(3);
bar([kppv_mean_val_error,dt_mean_val_error,bayes_mean_val_error,...
    reglog_mean_val_error,rnn_mean_val_error].*100);
set(gca, 'XTickLabel',{'KKPV','Arbre de décision','Bayes',...
    'Régression logistique','Reseau de neurone'});
title('Erreurs moyennes (%)');
