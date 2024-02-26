from classifier_model import *

# Fix Model to XGBoost
model_name = 'XGB'
# Fix Protein Embedding to ESM650M
# Fix Molecular Embedding to ChemBERT2
# Fix percentage of undersampling to 0.5
percentage = 0.5

SEED = 1234

# Load data
df = load_data_to_model(pickle_file='ESM2650M-ChemBERT2/data_embed_100000.pkl')

# Tune different hyperparameters

# PARAM TEST 1 --> max_depth and min_child_weight
param_test1 = pd.DataFrame(columns=['max_depth', 'min_child_weight', 'accuracy', 'f1'])
max_depths = [6, 7, 8]
min_child_weights = [2, 3, 4]
        
# PARAM TEST 2 --> gamma
param_test2 = pd.DataFrame(columns=['gamma', 'accuracy', 'f1'])
gammas = [i/10.0 for i in range(0,5)]
    
# PARAM TEST 3 --> subsample and colsample_bytree
param_test3 = pd.DataFrame(columns=['subsample', 'colsample_bytree', 'accuracy', 'f1'])
subsamples = [0.6, 0.7, 0.8, 0.9]
colsample_bytrees = [0.6, 0.7, 0.8, 0.9]
        
# PARAM TEST 4 --> reg_alpha
param_test4 = pd.DataFrame(columns=['reg_alpha', 'accuracy', 'f1'])
reg_alphas = [1e-5, 1e-2, 0.1, 1, 100]
    
# PARAM TEST 5 --> learning_rate
param_test5 = pd.DataFrame(columns=['learning_rate', 'accuracy', 'f1'])
learning_rates = [i/1000.0 for i in range(5,20,2)]


def tune_one_hyperparameter(hyperparameter_name, hyperparameter_values, df, percentage=0.5):
    """_summary_

    Args:
        hyperparameter_name (_type_): _description_
        hyperparameter_values (_type_): _description_
        df (_type_): _description_
        percentage (float, optional): _description_. Defaults to 0.5.
    """
    
    param_test = pd.DataFrame(columns=['%s'%hyperparameter_name, 'accuracy', 'f1'])
    
    for i, value in enumerate(hyperparameter_values):
        print('------------------------------------------------------')
        print('Hyperparameters: %s = %s' % (hyperparameter_name, value))
    
        # Load model
        if hyperparameter_name == 'max_depth':
            model_class = XGBClassifier(n_estimators=1000, device='cuda:0',
                                        learning_rate=0.1, max_depth=value,
                                        min_child_weight=2, gamma=0, 
                                        subsample=0.8, colsample_bytree=0.8,
                                        objective='binary:logistic', nthread=4, 
                                        scale_pos_weight=1, seed=SEED,
                                        reg_alpha=1e-5)
        elif hyperparameter_name == 'min_child_weight':
            model_class = XGBClassifier(n_estimators=1000, device='cuda:0',
                                        learning_rate=0.01, max_depth=7,
                                        min_child_weight=value, gamma=0, 
                                        subsample=0.8, colsample_bytree=0.8,
                                        objective='binary:logistic', nthread=4, 
                                        scale_pos_weight=1, seed=SEED,
                                        reg_alpha=1e-5)
        elif hyperparameter_name == 'gamma':
            model_class = XGBClassifier(n_estimators=1000, device='cuda:0',
                                        learning_rate=0.01, max_depth=7,
                                        min_child_weight=2, gamma=value, 
                                        subsample=0.8, colsample_bytree=0.8,
                                        objective='binary:logistic', nthread=4, 
                                        scale_pos_weight=1, seed=SEED,
                                        reg_alpha=1e-5)
        elif hyperparameter_name == 'subsample':
            model_class = XGBClassifier(n_estimators=1000, device='cuda:0',
                                        learning_rate=0.01, max_depth=7,
                                        min_child_weight=2, gamma=0, 
                                        subsample=value, colsample_bytree=0.8,
                                        objective='binary:logistic', nthread=4, 
                                        scale_pos_weight=1, seed=SEED,
                                        reg_alpha=1e-5)
        elif hyperparameter_name == 'colsample_bytree':
            model_class = XGBClassifier(n_estimators=1000, device='cuda:0',
                                        learning_rate=0.01, max_depth=7,
                                        min_child_weight=2, gamma=0, 
                                        subsample=0.8, colsample_bytree=value,
                                        objective='binary:logistic', nthread=4, 
                                        scale_pos_weight=1, seed=SEED,
                                        reg_alpha=1e-5)
        elif hyperparameter_name == 'reg_alpha':
            model_class = XGBClassifier(n_estimators=1000, device='cuda:0',
                                        learning_rate=0.01, max_depth=7,
                                        min_child_weight=2, gamma=0, 
                                        subsample=0.8, colsample_bytree=0.8,
                                        objective='binary:logistic', nthread=4, 
                                        scale_pos_weight=1, seed=SEED,
                                        reg_alpha=value)
        elif hyperparameter_name == 'learning_rate':
            model_class = XGBClassifier(n_estimators=1000, device='cuda:0',
                                        learning_rate=value, max_depth=7,
                                        min_child_weight=2, gamma=0, 
                                        subsample=0.8, colsample_bytree=0.8,
                                        objective='binary:logistic', nthread=4, 
                                        scale_pos_weight=1, seed=SEED,
                                        reg_alpha=1e-5)
        else:
            raise ValueError('Hyperparameter name not valid')
        
        
        model_class = XGBClassifier(n_estimators=5000, device='cuda:0',
                                    learning_rate=learning_rate, max_depth=7,
                                    min_child_weight=2, gamma=0, 
                                    subsample=0.8, colsample_bytree=0.8,
                                    objective='binary:logistic', nthread=4, 
                                    scale_pos_weight=1, seed=SEED,
                                    reg_alpha=1e-5)
        # Undersampling
        X = df['coembed'].tolist()
        Y = df['Label'].tolist()
        X_res, Y_res = undersampling(Y, X, percentage=percentage)
        df_res = pd.DataFrame({'coembed': X_res, 'Label': Y_res})

        # Split data into training and test set
        static_train_x, static_test_x, static_train_y, static_test_y = split_data(embeds_to_model=X_res, label_to_model=Y_res, test_size=0.15, seed=SEED)
        df_res_train = pd.DataFrame({'coembed': static_train_x, 'Label': static_train_y})

        # Cross-validation
        N_FOLDS = 5
        statistics_list, plot_metrics = crossvalidation(model_class, df_res_train, n_folds=N_FOLDS, seed=SEED, verbose=False, plot_roc_xf=False, plot_pr_xf=False)
        print('Accuracy: {} \t f1: {}'.format(statistics_list[0], statistics_list[10]))
        
        # store parameters and statistics
        param_test = pd.concat([param_test, pd.DataFrame({'%s'%hyperparameter_name: [value], 'accuracy': [statistics_list[0]], 'f1': [statistics_list[10]]})], ignore_index=True)
        
    # choose the best hyperparameters
    print(param_test)
    print('Best hyperparameters: ')
    print(param_test[param_test['f1'] == param_test['f1'].max()])
    
def tune_two_hyperparameters(hyperparameter_name1, hyperparameter_values1, hyperparameter_name2, hyperparameter_values2, df, percentage=0.5):
    """_summary_

    Args:
        hyperparameter_name1 (_type_): _description_
        hyperparameter_values1 (_type_): _description_
        hyperparameter_name2 (_type_): _description_
        hyperparameter_values2 (_type_): _description_
        df (_type_): _description_
        percentage (float, optional): _description_. Defaults to 0.5.
    """
    param_test = pd.DataFrame(columns=['%s'%hyperparameter_name1, '%s'%hyperparameter_name2, 'accuracy', 'f1'])
    
    for i, value1 in enumerate(hyperparameter_values1):
        for j, value2 in enumerate(hyperparameter_values2):
            print('------------------------------------------------------')
            print('Hyperparameters: %s = %s, %s = %s' % (hyperparameter_name1, value1, hyperparameter_name2, value2))
        
            # Load model
            if hyperparameter_name1 == 'max_depth' and hyperparameter_name2 == 'min_child_weight':
                model_class = XGBClassifier(n_estimators=5000, device='cuda:0',
                                            learning_rate=0.01, max_depth=value1,
                                            min_child_weight=value2, gamma=0, 
                                            subsample=0.8, colsample_bytree=0.8,
                                            objective='binary:logistic', nthread=4, 
                                            scale_pos_weight=1, seed=SEED,
                                            reg_alpha=1e-5)
            elif hyperparameter_name1 == 'subsample' and hyperparameter_name2 == 'colsample_bytree':
                model_class = XGBClassifier(n_estimators=5000, device='cuda:0',
                                            learning_rate=0.01, max_depth=7,
                                            min_child_weight=2, gamma=0, 
                                            subsample=value1, colsample_bytree=value2,
                                            objective='binary:logistic', nthread=4, 
                                            scale_pos_weight=1, seed=SEED,
                                            reg_alpha=1e-5)
            else:
                raise ValueError('Hyperparameter name  combination not valid')
            
            # Undersampling
            X = df['coembed'].tolist()
            Y = df['Label'].tolist()
            X_res, Y_res = undersampling(Y, X, percentage=percentage)
            df_res = pd.DataFrame({'coembed': X_res, 'Label': Y_res})
            
            # Split data into training and test set
            static_train_x, static_test_x, static_train_y, static_test_y = split_data(embeds_to_model=X_res, label_to_model=Y_res, test_size=0.15, seed=SEED)
            df_res_train = pd.DataFrame({'coembed': static_train_x, 'Label': static_train_y})

            # Cross-validation
            N_FOLDS = 5
            statistics_list, plot_metrics = crossvalidation(model_class, df_res_train, n_folds=N_FOLDS, seed=SEED, verbose=False, plot_roc_xf=False, plot_pr_xf=False)
            print('Accuracy: {} \t f1: {}'.format(statistics_list[0], statistics_list[10]))
            
            # store parameters and statistics
            param_test = pd.concat([param_test, pd.DataFrame({'%s'%hyperparameter_name1: [value1], '%s'%hyperparameter_name2: [value2], 'accuracy': [statistics_list[0]], 'f1': [statistics_list[10]]})], ignore_index=True)
            
        # choose the best hyperparameters
        print(param_test)
        print('Best hyperparameters: ')
        print(param_test[param_test['f1'] == param_test['f1'].max()])
            
def comparison_between_models(df, list_of_models, list_model_names, percentage=0.5):
    """_summary_

    Args:
        df (_type_): _description_
        list_of_models (_type_): _description_
        list_model_names (_type_): _description_
        percentage (float, optional): _description_. Defaults to 0.5.
    """
    
    # Iterate over the different models
    statistics = dict()
    plots_metrics = dict()
    
    for i, model in enumerate(list_of_models):
        print('--------------------')
        print('Model: ', list_model_names[i])
        
        
        # Undersampling
        X = df['coembed'].tolist()
        Y = df['Label'].tolist()
        X_res, Y_res = undersampling(Y, X, percentage=percentage)
        df_res = pd.DataFrame({'coembed': X_res, 'Label': Y_res})
        
        # Cross-validation
        N_FOLDS = 5
        statistics_list, plot_metrics = crossvalidation(model, df_res, n_folds=N_FOLDS, seed=SEED, verbose=False, plot_roc_xf=True, plot_pr_xf=True, aux=list_model_names[i])
        statistics[list_model_names[i]] = statistics_list
        plots_metrics[list_model_names[i]] = plot_metrics
        
    # Get df of all model statistics
    df_statistics = pd.DataFrame.from_dict(statistics)
    index_names =['acc_mean', 'acc_std', 'sens_mean', 'sens_std',
                    'spec_mean', 'spec_std', 'auc_mean', 'auc_std',
                    'precision_mean', 'precision_std', 'f1_mean', 'f1_std', 'time']
    df_statistics.index = index_names
    df_statistics.to_csv('tests/statistics_hyp_XGB.csv')

    # Plot Precision-Recall curve
    plot_colors = mcp.gen_color(cmap="RdYlGn",n=4)
    colors = cycle(plot_colors)

    fig, ax = plt.subplots(figsize=(10, 8))
    for i, color in zip(range(len(list_model_names)), colors):
        display = PrecisionRecallDisplay(
            precision=plots_metrics[list_model_names[i]][1],
            recall=plots_metrics[list_model_names[i]][0],
            estimator_name='%s_%s'%(model_name, list_model_names[i]))
        display.plot(ax=ax, color=color, label='%s_%s'%(model_name, list_model_names[i]))

    handles, labels = display.ax_.get_legend_handles_labels()
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(handles, labels, loc="best")
    ax.set_title('Precision-Recall curve')
    plt.savefig('tests/PR_curve_hyp_XGB.png')

    # Plot ROC curve
    fig, ax = plt.subplots(figsize=(10, 8))
    for i, color in zip(range(len(list_model_names)), colors):
        display = RocCurveDisplay(
            fpr=plots_metrics[list_model_names[i]][2],
            tpr=plots_metrics[list_model_names[i]][3],
            roc_auc=auc(plots_metrics[list_model_names[i]][0], plots_metrics[list_model_names[i]][1]),
            estimator_name='%s_%s'%(model_name, list_model_names[i]))
        display.plot(ax=ax, color=color, label='%s_%s'%(model_name, list_model_names[i]))
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.legend(loc="best")
    ax.set_title('ROC curve')
    plt.savefig('tests/ROC_curve_hyp_XGB.png')


if __name__ == '__main__':
    
    # Tune hyperparameters here, the order is important!
    #tune_two_hyperparameters(hyperparameter_name1='max_depth', hyperparameter_values1=[6, 7, 8], hyperparameter_name2='min_child_weight', hyperparameter_values2=[2, 3, 4], df=df, percentage=percentage)
    #tune_one_hyperparameter(hyperparameter_name='gamma', hyperparameter_values=[i/10.0 for i in range(0,5)], df=df, percentage=percentage)
    #tune_two_hyperparameters(hyperparameter_name1='subsample', hyperparameter_values1=[0.6, 0.7, 0.8, 0.9], hyperparameter_name2='colsample_bytree', hyperparameter_values2=[0.6, 0.7, 0.8, 0.9], df=df, percentage=percentage)
    #tune_one_hyperparameter(hyperparameter_name='reg_alpha', hyperparameter_values=[1e-5, 1e-2, 0.1, 1, 100], df=df, percentage=percentage)
    #tune_one_hyperparameter(hyperparameter_name='learning_rate', hyperparameter_values=[i/1000.0 for i in range(5,20,2)], df=df, percentage=percentage)
    
    # All performance + plots
    model_default = XGBClassifier(n_estimators=5000, device='cuda:0', seed=SEED)
    model_hyp = XGBClassifier(n_estimators=5000, device='cuda:0',
                            learning_rate=0.015, max_depth=7,
                            min_child_weight=2, gamma=0, 
                            subsample=0.8, colsample_bytree=0.8,
                            objective='binary:logistic', nthread=4, 
                            scale_pos_weight=1, seed=SEED,
                            reg_alpha=1e-5)
    comparison_between_models(df, [model_default, model_hyp], ['default', 'hyp'], percentage=percentage)

