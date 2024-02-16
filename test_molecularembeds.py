from classifier_model import *

# Fix Model to XGBoost
model_name = 'XGB'
# Fix Protein Embedding to ESM650M

# Fix percentage of undersampling to 0.5
percentage = 0.5
# And test different Molecular Embeddings

# Select data
SEED = 1234
files = ['ESM2650M-MolFormer/data_embed_100000.pkl', 'ESM2650M-SELFormer/data_embed_100000.pkl',
            'ESM2650M-ChemBERT2/data_embed_100000.pkl', 'ESM2650M-fingerprints/data_embed_100000.pkl']
names = ['MolFormer', 'SELFormer', 'ChemBERT2', 'Fingerprints']

# Load model
model_class = XGBClassifier(n_estimators=5000, device='cuda:0')

# Iterate over the different molecular embeddings
statistics = dict()
plots_metrics = dict()

for i, pfile in enumerate(files):
    print('--------------------')
    print('Model: ', model_name)
    print('Molecular Embedding: ', names[i])
    
    # Load data
    df = load_data_to_model(pickle_file=pfile)
    
    # Undersampling
    X = df['coembed'].tolist()
    Y = df['Label'].tolist()
    X_res, Y_res = undersampling(Y, X, percentage=percentage)
    df_res = pd.DataFrame({'coembed': X_res, 'Label': Y_res})
    
    # Cross-validation
    N_FOLDS = 5
    statistics_list, plot_metrics = crossvalidation(model_class, df_res, n_folds=N_FOLDS, seed=SEED, verbose=False, plot_roc_xf=True, plot_pr_xf=True, aux=names[i])
    statistics[names[i]] = statistics_list
    plots_metrics[names[i]] = plot_metrics

# Get df of all model statistics
df_statistics = pd.DataFrame.from_dict(statistics)
index_names =['acc_mean', 'acc_std', 'sens_mean', 'sens_std',
                'spec_mean', 'spec_std', 'auc_mean', 'auc_std',
                'precision_mean', 'precision_std', 'f1_mean', 'f1_std', 'time']
df_statistics.index = index_names
df_statistics.to_csv('tests/statistics_molecularembeds.csv')

# Plot Precision-Recall curve
plot_colors = mcp.gen_color(cmap="RdYlGn",n=4)
colors = cycle(plot_colors)

fig, ax = plt.subplots(figsize=(10, 8))
for i, color in zip(range(len(names)), colors):
    display = PrecisionRecallDisplay(
        precision=plots_metrics[names[i]][1],
        recall=plots_metrics[names[i]][0],
        estimator_name='%s_%s'%(model_name, names[i]))
    display.plot(ax=ax, color=color, label='%s_%s'%(model_name, names[i]))

handles, labels = display.ax_.get_legend_handles_labels()
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(handles, labels, loc="best")
ax.set_title('Precision-Recall curve')
plt.savefig('tests/PR_curve_molecular_embeds.png')

# Plot ROC curve
fig, ax = plt.subplots(figsize=(10, 8))
for i, color in zip(range(len(names)), colors):
    display = RocCurveDisplay(
        fpr=plots_metrics[names[i]][2],
        tpr=plots_metrics[names[i]][3],
        roc_auc=auc(plots_metrics[names[i]][0], plots_metrics[names[i]][1]),
        estimator_name='%s_%s'%(model_name, names[i]))
    display.plot(ax=ax, color=color, label='%s_%s'%(model_name, names[i]))
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(loc="best")
ax.set_title('ROC curve')
plt.savefig('tests/ROC_curve_molecularembeds.png')



