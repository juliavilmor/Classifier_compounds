import pandas as pd
import numpy as np
import torch
from sklearn import svm, metrics, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.model_selection import KFold, RepeatedStratifiedKFold
from sklearn.metrics import auc, accuracy_score, recall_score, PrecisionRecallDisplay, precision_recall_curve, average_precision_score
from sklearn.metrics import roc_curve, roc_auc_score, RocCurveDisplay
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from mycolorpy import colorlist as mcp
from statistics import mean
from collections import Counter
from itertools import cycle
import time
import os

# LOAD DATA
def sample_data_from_pickle(pickle_file, n_samples):
    """
    It samples a number of data from a pickle file and save it to a new pickle file.

    Args:
        pickle_file (str): pickle file with the original data set
        n_samples (int): number of samples to be extracted from the original data set

    Returns:
        df: data frame with the sampled data
    """
    
    df = pd.read_pickle(pickle_file)
    print('Shape of dataset: ', df.shape)
    print('Number of active compounds: ', df[df['Label'] == 1].shape)
    print('Number of inactive compounds (decoys): ', df[df['Label'] == 0].shape)

    df = df.sample(n=n_samples)
    outname = os.path.basename(pickle_file).replace('.pkl', '_%s.pkl'%n_samples)
    df.to_pickle(outname)
    return df

def load_data_to_model(pickle_file):
    """
    It loads a pickle file with the data set to be used for the model.
    It prints the number of active and inactive compounds.

    Args:
        pickle_file (str): pickle file with the data set

    Returns:
        df: data frame with the data set
    """
    df = pd.read_pickle(pickle_file)
    print('Original dataset:')
    print(' - Number of active compounds: ', len(df[df['Label'] == 1]))
    print(' - Number of inactive compounds (decoys): ', len(df[df['Label'] == 0]))
    return df


# UNDERSAMPLING TO BALANCE THE DATA
def undersampling(y, x, percentage):
    """
    It undersamples the data set to balance the number of active and inactive compounds.
    It prints the number of active and inactive compounds before and after undersampling.
    Different percentages of undersampling can be used.

    Args:
        y (list): list of labels
        x (list): list of embeddings
        percentage (float): percentage of undersampling

    Returns:
        X_res, Y_res (list, list): list of embeddings and list of labels after undersampling
    """
    
    print(' - Original dataset shape %s' % Counter(y))
    rus = RandomUnderSampler(random_state=42, sampling_strategy=percentage) # random state is the seed
    X_res, Y_res = rus.fit_resample(x, y)
    print(' - Resampled dataset shape %s' % Counter(Y_res))
    return X_res, Y_res

# X = df['coembed'].tolist()
# Y = df['Label'].tolist()
# X_res, Y_res = undersampling(Y, X, percentage=0.05)

# SPLIT DATA INTO TRAINING AND TEST SET
#embeds_to_model = X_res
#label_to_model = Y_res

def split_data(embeds_to_model, label_to_model, test_size=0.2):
    """
    It splits the data set into training and test set.
    It prints the size of the training and test set.
    Different test sizes can be used.

    Args:
        embeds_to_model (list): list of embeddings (X_res)
        label_to_model (list): list of labels (Y_res)

    Returns:
        _type_: _description_
    """
    (
        static_train_x,
        static_test_x,
        static_train_y,
        static_test_y,
    ) = train_test_split(embeds_to_model, label_to_model, test_size=test_size, random_state=SEED)

    splits = [static_train_x, static_test_x, static_train_y, static_test_y]

    print("Splitting data into training and test set with test size %s"%test_size)
    print(" - Training data size:", len(static_train_x))
    print(" - Test data size:", len(static_test_x))
    return splits


# SET MODEL AND MODEL PARAMETERS
### Decision Tree
# model_DT = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=0)

### Random Forest
# param = {
#     "n_estimators": 100,  # number of trees to grows
#     "criterion": "entropy",  # cost function to be optimized for a split
#     #"class_weight": "balanced", # the classes will be weighted inversely proportional to how frequently they appear in the data
#     "n_jobs": 10 # number of cores to distribute the process
# }
# model_RF = RandomForestClassifier(**param)

### Gradient-boosted trees
# model_GB = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)

### Histogram-Based gradient boosting
# model_HGB = HistGradientBoostingClassifier(max_iter=100)
    
### SVM (Support Vector Machine)
# model_SVM = svm.SVC()

#models = [{"label": "Model_DT", "model": model_DT}, {"label": "Model_RF", "model": model_RF},
#          {"label": "Model_GB", "model": model_GB}, {"label": "Model_HGB", "model": model_HGB},
#          {"label": "Model_SVM", "model": model_SVM}]
#models = [{"label": "Model_RF", "model": model_RF}]


# HELPER FUNCTIONS TO CALCULATE MODEL PERFORMANCE

def model_performance(ml_model, test_x, test_y, verbose=True):
    """
    Helper function to calculate model performance

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    test_x: list
        Molecular fingerprints for test set.
    test_y: list
        Associated activity labels for test set.
    verbose: bool
        Print performance measure (default = True)

    Returns
    -------
    tuple:
        Accuracy, sensitivity, specificity, auc, precision, and f1 on test set.
    """

    # Prediction probability on test set
    test_prob = ml_model.predict_proba(test_x)[:, 1] # the greater label

    # Prediction class on test set
    test_pred = ml_model.predict(test_x)

    # Performance of model on test set
    accuracy = accuracy_score(test_y, test_pred)
    sens = recall_score(test_y, test_pred)
    spec = recall_score(test_y, test_pred, pos_label=0)
    auc = roc_auc_score(test_y, test_prob)
    precision = metrics.precision_score(test_y, test_pred)
    f1 = metrics.f1_score(test_y, test_pred)
    tn, fp, fn, tp = confusion_matrix(test_y, test_pred).ravel()
    conf_mat = {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}

    if verbose:
        # Print performance results
        print(f"Accuracy: {accuracy:.2}")
        print(f"Sensitivity: {sens:.2f}")
        print(f"Specificity: {spec:.2f}")
        print(f"AUC: {auc:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"F1: {f1:.2f}")
        print(f"Confusion matrix: {conf_mat}")

    return accuracy, sens, spec, auc, precision, f1, conf_mat

def model_training_and_validation(ml_model, name, splits, verbose=True):
    """
    Fit a machine learning model on a random train-test split of the data
    and return the performance measures.

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    name: str
        Name of machine learning algorithm: RF, SVM, ANN
    splits: list
        List of descriptor and label data: train_x, test_x, train_y, test_y.
    verbose: bool
        Print performance info (default = True)

    Returns
    -------
    tuple:
        Accuracy, sensitivity, specificity, auc on test set.

    """
    train_x, test_x, train_y, test_y = splits

    # Fit the model
    ml_model.fit(train_x, train_y)

    # Calculate model performance results
    accuracy, sens, spec, auc, precision, f1 = model_performance(ml_model, test_x, test_y, verbose)

    return accuracy, sens, spec, auc, precision, f1

#models = [{"label": "Model_RF", "model": model_RF}]
#plot_roc_curves_for_models(models, static_test_x, static_test_y, save_png=True)


# CROSS-VALIDATION

def crossvalidation(ml_model, df, n_folds=5, verbose=False, plot_roc_xf=True, plot_pr_xf=True):
    """
    Machine learning model training and validation in a cross-validation loop.

    Parameters
    ----------
    ml_model: sklearn model object
        The machine learning model to train.
    df: pd.DataFrame
        Data set with SMILES and their associated activity labels.
    n_folds: int, optional
        Number of folds for cross-validation.
    verbose: bool, optional
        Performance measures are printed.

    Returns
    -------
    None

    """
    t0 = time.time()
    # Shuffle the indices for the k-fold cross-validation
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

    # Results for each of the cross-validation folds
    acc_per_fold = []
    sens_per_fold = []
    spec_per_fold = []
    auc_per_fold = []
    precision_per_fold = []
    f1_per_fold = []
    
    # Results for plot of precision-recall curve and ROC curve
    precisions = []
    recalls = []
    average_precisions = []
    fprs = []
    tprs = []
    pr_thresholds = []
    roc_thresholds = []    

    mean_fpr = np.linspace(0, 1, 100)
    fig, ax = plt.subplots(figsize=(6, 6))
    # Loop over the folds
    for train_index, test_index in kf.split(df):
        nfold = len(acc_per_fold) + 1
        # clone model -- we want a fresh copy per fold!
        fold_model = clone(ml_model)
        
        # Training
        # Convert the embedding and the label to a list
        train_x = df.iloc[train_index].coembed.tolist()
        train_y = df.iloc[train_index].Label.tolist()
        
        # Print the number of active and inactive compounds in the training set
        if verbose:
            print(' - Fold: ', nfold)
            print('       Number of active compounds in the training set: ', train_y.count(1))
            print('       Number of inactive compounds in the training set: ', train_y.count(0))
        
        # Fit the model
        fold_model.fit(train_x, train_y)

        # Testing
        # Convert the fingerprint and the label to a list
        test_x = df.iloc[test_index].coembed.tolist()
        test_y = df.iloc[test_index].Label.tolist()

        # Performance for each fold
        accuracy, sens, spec, auc, precision, f1, conf_mat = model_performance(fold_model, test_x, test_y, verbose)
        
        # Save results
        acc_per_fold.append(accuracy)
        sens_per_fold.append(sens)
        spec_per_fold.append(spec)
        auc_per_fold.append(auc)
        precision_per_fold.append(precision)
        f1_per_fold.append(f1)
        
        # Get precision and recall curve metrics
        y_real = []
        y_proba = []
        precision, recall, pr_thresh = precision_recall_curve(test_y, fold_model.predict_proba(test_x)[:, 1])
        average_precision = average_precision_score(test_y, fold_model.predict_proba(test_x)[:, 1])
        if plot_pr_xf:
            display = PrecisionRecallDisplay(
                precision=precision, 
                recall=recall,  
                estimator_name='XGB Fold %s'%nfold)
            display.plot(ax=ax, lw=1, alpha=0.7)
            y_real.append(test_y)
            y_proba.append(fold_model.predict_proba(test_x)[:, 1])        
        
        # Get ROC curve metrics
        fpr, tpr, roc_thresh = roc_curve(test_y, fold_model.predict_proba(test_x)[:, 1], drop_intermediate=False)
        #print(len(roc_thresh), len(fpr), len(tpr)) # if the length is different for each fold, we need to recalculate the plot metrics (below)
        #print(len(np.unique(fold_model.predict_proba(train_x)[:, 1]))) # the length is this number + 1
        if plot_roc_xf:
            display = RocCurveDisplay(fpr=fpr, 
                                        tpr=tpr, 
                                        roc_auc=auc, 
                                        estimator_name='XGB Fold %s'%nfold,
                                        )
            display.plot(ax=ax, lw=0.9, alpha=0.7)
            interp_tpr = np.interp(mean_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
        
        
    # Print statistics of results
    print(
        f"Mean accuracy: {np.mean(acc_per_fold):.2f} \t"
        f"and std : {np.std(acc_per_fold):.2f} \n"
        f"Mean sensitivity: {np.mean(sens_per_fold):.2f} \t"
        f"and std : {np.std(sens_per_fold):.2f} \n"
        f"Mean specificity: {np.mean(spec_per_fold):.2f} \t"
        f"and std : {np.std(spec_per_fold):.2f} \n"
        f"Mean AUC: {np.mean(auc_per_fold):.2f} \t"
        f"and std : {np.std(auc_per_fold):.2f} \n"
        f"Mean precision: {np.mean(precision_per_fold):.2f} \t"
        f"and std : {np.std(precision_per_fold):.2f} \n"
        f"Mean f1: {np.mean(f1_per_fold):.2f} \t"
        f"and std : {np.std(f1_per_fold):.2f} \n"
        f"Time taken : {time.time() - t0:.2f}s\n"
    )

    statistics_list = [np.mean(acc_per_fold), np.std(acc_per_fold), 
                        np.mean(sens_per_fold), np.std(sens_per_fold),
                        np.mean(spec_per_fold), np.std(spec_per_fold),
                        np.mean(auc_per_fold), np.std(auc_per_fold),
                        np.mean(precision_per_fold), np.std(precision_per_fold),
                        np.mean(f1_per_fold), np.std(f1_per_fold),
                        time.time() - t0]
    
    
    # # Recalculate the plot metrics to have the same threshold values for each fold
    # unique_pr_thresholds = set(pr_thresholds[0]).intersection(*pr_thresholds[1:])
    # unique_roc_thresholds = set(roc_thresholds[0]).intersection(*roc_thresholds[1:])
    # #print(len(unique_pr_thresholds), len(unique_roc_thresholds)) # new length of the thresholds
    
    # new_fprs = []
    # new_tprs = []
    # for i in range(len(fprs)): # for each fold
    #     roc_fprs = []
    #     roc_tprs = []
    #     for j, th in enumerate(unique_roc_thresholds): # get the value in the same position of the threshold
    #         roc_fprs.append(fprs[i][j])
    #         roc_tprs.append(tprs[i][j])
    #     roc_fprs = np.array(roc_fprs)
    #     roc_tprs = np.array(roc_tprs)
    #     new_fprs.append(roc_fprs)
    #     new_tprs.append(roc_tprs)

    # new_precisions = []
    # new_recalls = []
    # for i in range(len(precisions)): # for each fold
    #     pr_precisions = []
    #     pr_recalls = []
    #     for j, th in enumerate(unique_pr_thresholds): # get the value in the same position of the threshold
    #         pr_precisions.append(precisions[i][j])
    #         pr_recalls.append(recalls[i][j])
    #     pr_precisions = np.array(pr_precisions)
    #     pr_recalls = np.array(pr_recalls)
    #     new_precisions.append(pr_precisions)
    #     new_recalls.append(pr_recalls)
    
    # Calculate the mean and std of the plot metrics
    if plot_pr_xf:
        y_real = np.concatenate(y_real)
        y_proba = np.concatenate(y_proba)
        mean_precision, mean_recall, _ = precision_recall_curve(y_real, y_proba)
        ax.plot(mean_recall, mean_precision, color='b', lw=1.5, alpha=.9, label='Mean PR (AUC = %0.2f)'%(average_precision_score(y_real, y_proba)))
        ax.legend(loc="best")
        plt.savefig('PR_curve_xf.png')

    if plot_roc_xf:
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        ax.plot(mean_fpr, mean_tpr, color='b', lw=1.5, alpha=.9, label='Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (np.mean(auc_per_fold), np.std(auc_per_fold)))
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
        ax.legend(loc="best")
        plt.savefig('ROC_curve_xf.png')
        
    #plot_metrics = [mean_recall, mean_precision, mean_fpr, mean_tpr]
    plot_metrics = []
    return statistics_list, plot_metrics
    
# EXECUTION OF THE MODEL

if __name__ == '__main__':
    
    # Sample data
    #sample_data_from_pickle('data_embed.pkl', n_samples=100000)
    
    # Load data
    df = load_data_to_model(pickle_file='ESM2650M-fingerprints/data_embed_sample100000.pkl')
    SEED = 1234
    
    # Try different percentages of under-sampling
    # and run the model to get performances
    
    percentages = [0.4]
    X = df['coembed'].tolist()
    Y = df['Label'].tolist()
    
    statistics = dict()
    plots_metrics = dict()
    
    for i, perc in enumerate(percentages):
        print('-------------')
        print('Undersampling percentage: ', perc)
        X_res, Y_res = undersampling(Y, X, percentage=perc)
        df_res = pd.DataFrame({'coembed': X_res, 'Label': Y_res})
        
        # Split data into training and test set
        #splits = split_data(embeds_to_model=X_res, label_to_model=Y_res)

        # Define model and model parameters
        #model_DT = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=0)
        
        param = {
        "n_estimators": 100,  # number of trees to grows
        "criterion": "entropy",  # cost function to be optimized for a split
        "class_weight": "balanced", # the classes will be weighted inversely proportional to how frequently they appear in the data
        "n_jobs": 15 # number of cores to distribute the process
        }
        model_RF = RandomForestClassifier(**param)
        
        # model_XGB = XGBClassifier(n_estimators=3000, device='cuda')
        
        # Cross-validation
        N_FOLDS = 5        
        statistics_list, plot_metrics = crossvalidation(model_RF, df_res, n_folds=N_FOLDS, verbose=False, plot_roc_xf=False, plot_pr_xf=True)
        statistics[perc] = statistics_list
        plots_metrics[perc] = plot_metrics
    exit()
    # Get df of all model statistics
    df_statistics = pd.DataFrame.from_dict(statistics)
    index_names =['acc_mean', 'acc_std', 'sens_mean', 'sens_std', 
                    'spec_mean', 'spec_std', 'auc_mean', 'auc_std', 
                    'precision_mean', 'precision_std', 'f1_mean', 'f1_std', 'time']
    df_statistics.index = index_names
    df_statistics.to_csv('ESM2650M-fingerprints/statistics_XGB.csv')

    # Plot Precision-Recall curve
    plot_colors = mcp.gen_color(cmap="viridis",n=6)
    colors = cycle(plot_colors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    # f_scores = np.linspace(0.2, 0.8, num=4)
    # lines, labels = [], []
    # for f_score in f_scores:
    #     x = np.linspace(0.01, 1)
    #     y = f_score * x / (2 * x - f_score)
    #     (l,) = plt.plot(x[y >= 0], y[y >= 0], color="gray", alpha=0.2)
    #     plt.annotate("f1={0:0.1f}".format(f_score), xy=(0.9, y[45] + 0.02))
    
    for i, color in zip(range(len(percentages)), colors):
        display = PrecisionRecallDisplay(
            precision=plots_metrics[percentages[i]][0], 
            recall=plots_metrics[percentages[i]][1],  
            estimator_name='XGB_%s'%percentages[i])
        display.plot(ax=ax, color=color, label='XGB_%s'%percentages[i])
    
    handles, labels = display.ax_.get_legend_handles_labels()
    #handles.extend([l])
    #labels.extend("f1-score")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles, labels, loc="best")
    ax.set_title('Precision-Recall curve')        
    plt.savefig('PR_curve_XGB.png')
    
    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, color in zip(range(len(percentages)), colors):
        #ax.plot(plots_metrics[percentages[i]][1], plots_metrics[percentages[i]][0], color=color, lw=2, label='RF_%s'%percentages[i]) # it is the same as display (below)
        display = RocCurveDisplay(
            fpr=plots_metrics[percentages[i]][2], 
            tpr=plots_metrics[percentages[i]][3], 
            roc_auc=auc(plots_metrics[percentages[i]][0], plots_metrics[percentages[i]][1]), 
            estimator_name='XGB_%s'%percentages[i])
        display.plot(ax=ax, color=color, label='XGB_%s'%percentages[i])
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best")
    ax.set_title('ROC curve')
    plt.savefig('ROC_curve_XGB.png')
        