from classifier_model import *

# Load data
df = load_data_to_model(pickle_file='ESM2650M-fingerprints/data_embed_100000.pkl')
SEED = 1234

# Try different percentages of under-sampling
# and run the model to get performances

percentages = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
X = df['coembed'].tolist()
Y = df['Label'].tolist()

statistics = dict()
plots_metrics = dict()

# Indicate the model to use here:
model_name = 'RF'

for i, perc in enumerate(percentages):
    print('-------------')
    print('Model: ', model_name)
    print('Undersampling percentage: ', perc)
    X_res, Y_res = undersampling(Y, X, percentage=perc)
    df_res = pd.DataFrame({'coembed': X_res, 'Label': Y_res})

    # Split data into training and test set
    # static_train_x, static_test_x, static_train_y, static_test_y = split_data(embeds_to_model=X_res, label_to_model=Y_res, test_size=0.15, seed=SEED)
    # df_res_train = pd.DataFrame({'coembed': static_train_x, 'Label': static_train_y})
    
    # Define model and model parameters
    if model_name == 'DT':
        model_class = DecisionTreeClassifier(criterion='entropy', splitter='best', random_state=0)
        
    elif model_name == 'RF':
        param = {
        "n_estimators": 100,  # number of trees to grow
        "criterion": "entropy",  # cost function to be optimized for a split
        "class_weight": "balanced", # the classes will be weighted inversely proportional to how frequently they appear in the data
        "n_jobs": 12 # number of cores to distribute the process
        }
        model_class = RandomForestClassifier(**param)
        
    elif model_name == 'XGB':
        model_class = XGBClassifier(n_estimators=5000, device='cuda:0', 
                                    scale_pos_weight=50, max_depth=4, 
                                    min_child_weight=6, learning_rate=0.01, 
                                    gamma=0, reg_alpha=0.005, objective= 'binary:logistic')
        
    else:
        raise ValueError('Model not found. The only available models are DT, RF or XGB.')

    # Cross-validation
    N_FOLDS = 5
    statistics_list, plot_metrics = crossvalidation(model_class, df_res, n_folds=N_FOLDS, seed=SEED, verbose=False, plot_roc_xf=False, plot_pr_xf=True)
    statistics[perc] = statistics_list
    plots_metrics[perc] = plot_metrics
    
# Get df of all model statistics
df_statistics = pd.DataFrame.from_dict(statistics)
index_names =['acc_mean', 'acc_std', 'sens_mean', 'sens_std',
                'spec_mean', 'spec_std', 'auc_mean', 'auc_std',
                'precision_mean', 'precision_std', 'f1_mean', 'f1_std', 'time']
df_statistics.index = index_names
df_statistics.to_csv('ESM2650M-fingerprints/statistics_%s.csv'%model_name)

# Plot Precision-Recall curve
plot_colors = mcp.gen_color(cmap="viridis",n=6)
colors = cycle(plot_colors)

fig, ax = plt.subplots(figsize=(10, 8))
for i, color in zip(range(len(percentages)), colors):
    display = PrecisionRecallDisplay(
        precision=plots_metrics[percentages[i]][1],
        recall=plots_metrics[percentages[i]][0],
        estimator_name='%s_%s'%(model_name, percentages[i]))
    display.plot(ax=ax, color=color, label='%s_%s'%(model_name, percentages[i]))

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(handles, labels, loc="best")
ax.set_title('Precision-Recall curve')
plt.savefig('ESM2650M-fingerprints/PR_curve_%s.png'%model_name)

# Plot ROC curve
fig, ax = plt.subplots(figsize=(10, 8))
for i, color in zip(range(len(percentages)), colors):
    display = RocCurveDisplay(
        fpr=plots_metrics[percentages[i]][2],
        tpr=plots_metrics[percentages[i]][3],
        roc_auc=auc(plots_metrics[percentages[i]][0], plots_metrics[percentages[i]][1]),
        estimator_name='%s_%s'%(model_name, percentages[i]))
    display.plot(ax=ax, color=color, label='%s_%s'%(model_name, percentages[i]))
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(loc="best")
ax.set_title('ROC curve')
plt.savefig('ESM2650M-fingerprints/ROC_curve_%s.png'%model_name)