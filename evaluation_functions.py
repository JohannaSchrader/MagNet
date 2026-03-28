import os
import warnings
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from tensorflow.keras.utils import register_keras_serializable
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import seaborn as sns
from matplotlib.lines import Line2D
from openTSNE import TSNE as OpenTSNE
import warnings
from scipy.special import softmax
os.makedirs("figures", exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "dejavuserif",
    "axes.titlesize": 15,
    "axes.labelsize": 15,
})

DISPLAY_NAMES = {
    'OURS': 'MᴀɢNᴇᴛ', 
    'TABNET': 'TabNet'
}
DATASET_NAME = 'miRNA' 
MODELS = ['OURS', 'MLP', 'TABNET', 'HEAD', '1D', 'XGB', 'RF', 'LR']
TARGET_COMB = 0  

@register_keras_serializable()
class MahalanobisOutput(layers.Layer):
    def __init__(self, num_classes, embedding_dim, **kwargs):
        super(MahalanobisOutput, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

    def build(self, input_shape):
        self.centers = self.add_weight(
            name='prototypes',
            shape=(self.num_classes, self.embedding_dim),
            initializer='glorot_uniform',
            trainable=True
        )
        
        self.inverse_sigma = self.add_weight(
            name='inverse_sigma',
            shape=(self.num_classes, self.embedding_dim),
            initializer='ones',
            trainable=True
        )
        super(MahalanobisOutput, self).build(input_shape)

    def call(self, inputs):
        # inputs: (Batch, Dim)
        x = tf.expand_dims(inputs, 1)          # (Batch, 1, Dim)
        mu = tf.expand_dims(self.centers, 0)   # (1, Classes, Dim)
        sigma = tf.expand_dims(self.inverse_sigma, 0) # (1, Classes, Dim)
        
        squared_diff = tf.square(x - mu)
        sigma_clean = tf.math.softplus(sigma) + 1e-5
        weighted_dist = tf.multiply(squared_diff, sigma_clean)
        mahalanobis_sq = tf.reduce_sum(weighted_dist, axis=2)
        scaler = tf.math.rsqrt(tf.cast(self.embedding_dim, tf.float32))
        
        return tf.nn.softmax(-mahalanobis_sq * scaler)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "embedding_dim": self.embedding_dim,
        })
        return config
    
@register_keras_serializable()
class MCGaussianNoise(layers.GaussianNoise):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
    def get_config(self):
        return super().get_config()

@register_keras_serializable()
class MCDropout(layers.Dropout):
    def call(self, inputs, training=None):
        return super().call(inputs, training=True)
    def get_config(self):
        return super().get_config()

def calculate_entropy(probs):
    """Calculates Shannon entropy across the class probabilities."""
    return -np.sum(probs * np.log(probs + 1e-9), axis=1)

def calculate_confidence(probs):
    """Calculates the Max Softmax Probability (Confidence) for each sample."""
    return np.max(probs, axis=1)


def plot_sex_age_stage_grade():
    status = pd.read_csv('data/status_file.csv')
    status['Accession'] = status['Accession'].astype(str).str.strip()

    bg_df = pd.read_csv('data/backgroundcorrected_idx.csv')
    accessions = bg_df.iloc[:, 0].values

    DATASET_NAME = 'miRNA'
    tasks_map = {2: '14-Class (Tissue)', 0: '19-Class (Disease)'}

    all_fold_scores = []
    subgroup_counts = {}

    for comb, task_name in tasks_map.items():
        pred_file = os.path.join("result", "baselines", f"{DATASET_NAME}_nestedcv_predictions_{comb}_OURS.csv")
        df = pd.read_csv(pred_file)

        df['Real_ID'] = df['id'].apply(lambda row_idx: str(accessions[int(row_idx)]).strip())
        
        df = df.merge(status[['Accession', 'sex', 'stage', 'age', 'glioma']], left_on='Real_ID', right_on='Accession', how='left')

        df['Sex'] = df['sex']
        df['Stage_Raw'] = df['stage'].astype(str).str.replace('.0', '', regex=False)
        df['Glioma_Raw'] = df['glioma'].astype(str).str.replace('.0', '', regex=False)
        df['Age_Raw'] = pd.to_numeric(df['age'], errors='coerce')

        roman_numerals = {'0': 'Stage 0', '1': 'Stage I', '2': 'Stage II', '3': 'Stage III', '4': 'Stage IV'}
        df['Stage'] = df['Stage_Raw'].map(roman_numerals).fillna('Unknown/Other')
        glioma_numerals = {'2': 'Grade II', '3': 'Grade III', '4': 'Grade IV'}
        df['Glioma Grade'] = df['Glioma_Raw'].map(glioma_numerals).fillna('Unknown/Other')
        df['Age Group'] = pd.cut(df['Age_Raw'], bins=[0, 40, 55, 70, 120], labels=['<40', '40-55', '56-70', '>70'])

        if comb == 2:
            for col in ['Sex', 'Age Group', 'Stage', 'Glioma Grade']:
                counts = df[col].value_counts()
                for subgroup, count in counts.items():
                    subgroup_counts[subgroup] = count

        def extract_fold_scores(group_col, valid_groups, demographic_category):
            data = df[df[group_col].isin(valid_groups)]
            for fold in data['outer_fold'].unique():
                fold_data = data[data['outer_fold'] == fold]
                for group in valid_groups:
                    subset = fold_data[fold_data[group_col] == group]
                    if len(subset) >= 5: 
                        true_classes_present = np.unique(subset['y_true'])
                        score = f1_score(
                            subset['y_true'], 
                            subset['y_pred'], 
                            labels=true_classes_present, 
                            average='macro',
                            zero_division=0
                        )
                        all_fold_scores.append({
                            'Task': task_name,
                            'Category': demographic_category,
                            'Subgroup': group,
                            'F1': score
                        })
                        
        extract_fold_scores('Sex', ['Male', 'Female'], 'Sex')
        extract_fold_scores('Age Group', ['<40', '40-55', '56-70', '>70'], 'Age')
        extract_fold_scores('Stage', ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV'], 'Stage')
        extract_fold_scores('Glioma Grade', ['Grade II', 'Grade III', 'Grade IV'], 'Glioma')

    master_df = pd.DataFrame(all_fold_scores)

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6), sharey=True)

    task_palette = {'14-Class (Tissue)': '#1f77b4', '19-Class (Disease)': '#ff7f0e'}

    sns.barplot(
        data=master_df[master_df['Category'] == 'Sex'], 
        x='Subgroup', y='F1', hue='Task', 
        ax=axes[0], palette=task_palette, errorbar='sd', capsize=0.05, edgecolor=".2", alpha=0.9
    )
    axes[0].set_title('A. Biological Sex', fontsize=20)

    sns.lineplot(
        data=master_df[master_df['Category'] == 'Age'], 
        x='Subgroup', y='F1', hue='Task', 
        ax=axes[1], palette=task_palette, marker='o', errorbar='sd', err_style='band', linewidth=2.5, markersize=8
    )
    axes[1].set_title('B. Age Group', fontsize=20)

    sns.lineplot(
        data=master_df[master_df['Category'] == 'Stage'], 
        x='Subgroup', y='F1', hue='Task', 
        ax=axes[2], palette=task_palette, marker='o', errorbar='sd', err_style='band', linewidth=2.5, markersize=8
    )
    axes[2].set_title('C. Solid Tumor Stage', fontsize=20)

    sns.lineplot(
        data=master_df[master_df['Category'] == 'Glioma'], 
        x='Subgroup', y='F1', hue='Task', 
        ax=axes[3], palette=task_palette, marker='o', errorbar='sd', err_style='band', linewidth=2.5, markersize=8
    )
    axes[3].set_title('D. Glioma Grade', fontsize=20)


    y_bottom = 0.0 
    axes[0].set_ylim(y_bottom, 1.05)
    axes[0].set_ylabel('Macro F1 Score (Mean ± SD)', fontsize=20)

    for i, ax in enumerate(axes):
        ax.set_xlabel('')
        ax.tick_params(axis='both', which='major', labelsize=15)
        if ax.get_legend() is not None:
            ax.get_legend().remove()
            
        if i > 0:
            ax.set_ylabel('')
            
        for tick_idx, tick_label in enumerate(ax.get_xticklabels()):
            subgroup_name = tick_label.get_text()
            count = subgroup_counts.get(subgroup_name, 0)
            
            if i == 0:
                txt = ax.text(tick_idx, y_bottom + 0.02, f'n={count}', 
                            ha='center', va='bottom', fontsize=13, color='white', weight='bold')
                txt.set_path_effects([pe.withStroke(linewidth=2, foreground='.2')])
            else:
                ax.text(tick_idx, y_bottom + 0.02, f'n={count}', 
                        ha='center', va='bottom', fontsize=12, color='dimgray', weight='bold')

    sns.despine(left=True)

    axes[3].legend(
        title="Diagnostic Task", 
        loc='lower right', 
        bbox_to_anchor=(0.98, 0.12), 
        framealpha=0.95,
        fontsize=12,            
        title_fontsize=14
    )

    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig('figures/subpop.pdf', bbox_inches='tight')
    plt.show()
    print("--- Original Performance (All Classes) ---")
    sex_stats_original = master_df[master_df['Category'] == 'Sex'].groupby(['Task', 'Subgroup'])['F1'].agg(['mean', 'std']).reset_index()
    
    for _, row in sex_stats_original.iterrows():
        print(f"Task: {row['Task']} | Sex: {row['Subgroup']} | F1: {row['mean']:.3f} ± {row['std']:.3f}")


def plot_confusion_by_stage():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
 
    status = pd.read_csv('data/status_file.csv')
    status['Accession'] = status['Accession'].astype(str).str.strip()

    bg_df = pd.read_csv('data/backgroundcorrected_idx.csv')
    accessions = bg_df.iloc[:, 0].values

    DATASET_NAME = 'miRNA'
    TARGET_COMB = 0  
    pred_file = os.path.join("result", "baselines", f"{DATASET_NAME}_nestedcv_predictions_{TARGET_COMB}_OURS.csv")
    df = pd.read_csv(pred_file)
 
    df['Real_ID'] = df['id'].apply(lambda row_idx: str(accessions[int(row_idx)]).strip())
    df = df.merge(status[['Accession', 'stage']], left_on='Real_ID', right_on='Accession', how='left')

    df['Stage_Raw'] = df['stage'].astype(str).str.replace('.0', '', regex=False)
    roman_numerals = {'0': 'Stage 0', '1': 'Stage I', '2': 'Stage II', '3': 'Stage III', '4': 'Stage IV'}
    df['Stage'] = df['Stage_Raw'].map(roman_numerals).fillna('Unknown/Other')
 
    CLASS_LABELS = [
        'B_BC', 'BC', 'BL', 'BT', 'CC', 'EC', 'GC', 'B_GL', 'GL', 
        'HC', 'LK', 'B_OV', 'OV', 'PC', 'B_PR', 'PR', 'B_SA', 'SA', 'healthy'
    ]

    df['y_true'] = df['y_true'].astype(int)
    df['y_pred'] = df['y_pred'].astype(int)
    class_indices = list(range(len(CLASS_LABELS)))
    stages = ['Stage 0', 'Stage I', 'Stage II', 'Stage III', 'Stage IV']
 
    sns.set_theme(style="white", context="paper", font_scale=1.1)
 
    fig, axes = plt.subplots(1, 6, figsize=(32, 7), 
                            gridspec_kw={'width_ratios': [1, 1, 1, 1, 1, 0.05]},
                            layout='constrained')

    for i, stage in enumerate(stages):
        df_stage = df[df['Stage'] == stage]
        cm = confusion_matrix(df_stage['y_true'], df_stage['y_pred'], labels=class_indices, normalize='true')
        
        sns.heatmap(cm, annot=False, cmap='Blues', ax=axes[i], 
                    xticklabels=CLASS_LABELS, 
                    yticklabels=CLASS_LABELS,
                    vmin=0, vmax=1, linewidths=.5, linecolor='lightgray', square=True,
                    cbar=(i == 4), 
                    cbar_ax=axes[5] if i == 4 else None) 
         
        axes[i].set_title(f'{stage}\n(n={len(df_stage)})', pad=10, fontsize=20)
        axes[i].set_xlabel('Predicted Class', fontsize=20)
        axes[i].set_ylabel('True Class', fontsize=20)
         
        axes[i].tick_params(axis='x', rotation=90, labelsize=15)
        axes[i].tick_params(axis='y', rotation=0, labelsize=15)

    plt.suptitle('Diagnostic Confusion by Solid Tumor Stage (19-Class Task)', y=1.1, fontsize=30)
 

    os.makedirs("figures", exist_ok=True)
    plt.savefig('figures/stage_cm.pdf', bbox_inches='tight')
    plt.show()

def plot_confusion_by_sex(): 
    status = pd.read_csv('data/status_file.csv')
    status['Accession'] = status['Accession'].astype(str).str.strip()

    bg_df = pd.read_csv('data/backgroundcorrected_idx.csv')
    accessions = bg_df.iloc[:, 0].values

    DATASET_NAME = 'miRNA'
    TARGET_COMB = 0  
    pred_file = os.path.join("result", "baselines", f"{DATASET_NAME}_nestedcv_predictions_{TARGET_COMB}_OURS.csv")
    df = pd.read_csv(pred_file)
 
    df['Real_ID'] = df['id'].apply(lambda row_idx: str(accessions[int(row_idx)]).strip())
    df = df.merge(status[['Accession', 'sex']], left_on='Real_ID', right_on='Accession', how='left')
    df['Sex'] = df['sex']
 
    CLASS_LABELS = [
        'B_BC', 'BC', 'BL', 'BT', 'CC', 'EC', 'GC', 'B_GL', 'GL', 
        'HC', 'LK', 'B_OV', 'OV', 'PC', 'B_PR', 'PR', 'B_SA', 'SA', 'healthy'
    ]
 
    df['y_true'] = df['y_true'].astype(int)
    df['y_pred'] = df['y_pred'].astype(int)

    df_male = df[df['Sex'] == 'Male']
    df_female = df[df['Sex'] == 'Female']
 
    class_indices = list(range(len(CLASS_LABELS)))

    cm_male = confusion_matrix(df_male['y_true'], df_male['y_pred'], labels=class_indices, normalize='true')
    cm_female = confusion_matrix(df_female['y_true'], df_female['y_pred'], labels=class_indices, normalize='true')
 
    sns.set_theme(style="white", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
 
    sns.heatmap(cm_male, annot=False, cmap='BuPu', ax=axes[0], 
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, vmin=0, vmax=1,
                linewidths=.5, linecolor='lightgray', square=True)
    axes[0].set_title(f'A. Male Cohort (n={len(df_male)})', pad=15, weight='bold')
    axes[0].set_xlabel('Predicted Class')
    axes[0].set_ylabel('True Class')
 
    sns.heatmap(cm_female, annot=False, cmap='RdPu', ax=axes[1], 
                xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS, vmin=0, vmax=1,
                linewidths=.5, linecolor='lightgray', square=True)
    axes[1].set_title(f'B. Female Cohort (n={len(df_female)})', pad=15, weight='bold')
    axes[1].set_xlabel('Predicted Class')
    axes[1].set_ylabel('') 

    plt.suptitle('Normalized Diagnostic Confusion by Biological Sex (19-Class Task)', y=1.02, fontsize=18, weight='bold')
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig('figures/confusion_matrix_labeled.png', dpi=300, bbox_inches='tight')
    plt.show()




def plot_sex_ablation_panel(): 
    status = pd.read_csv('data/status_file.csv')
    status['Accession'] = status['Accession'].astype(str).str.strip()

    bg_df = pd.read_csv('data/backgroundcorrected_idx.csv')
    accessions = bg_df.iloc[:, 0].values

    DATASET_NAME = 'miRNA'
    tasks_map = {2: '14-Class (Tissue)', 0: '19-Class (Disease)'}
    task_palette = {'14-Class (Tissue)': '#1f77b4', '19-Class (Disease)': '#ff7f0e'}
 
    pred_file_19 = os.path.join("result", "baselines", f"{DATASET_NAME}_nestedcv_predictions_0_OURS.csv")
    df_19 = pd.read_csv(pred_file_19)
    df_19['Real_ID'] = df_19['id'].apply(lambda row_idx: str(accessions[int(row_idx)]).strip())
    df_19 = df_19.merge(status[['Accession', 'sex']], left_on='Real_ID', right_on='Accession', how='left')
    
    CLASS_LABELS_19 = [
        'B_BC', 'BC', 'BL', 'BT', 'CC', 'EC', 'GC', 'B_GL', 'GL', 
        'HC', 'LK', 'B_OV', 'OV', 'PC', 'B_PR', 'PR', 'B_SA', 'SA', 'healthy'
    ]
    
    df_male_19 = df_19[df_19['sex'] == 'Male']
    df_female_19 = df_19[df_19['sex'] == 'Female']
    
    cm_male = confusion_matrix(df_male_19['y_true'].astype(int), df_male_19['y_pred'].astype(int), 
                               labels=range(19), normalize='true')
    cm_female = confusion_matrix(df_female_19['y_true'].astype(int), df_female_19['y_pred'].astype(int), 
                                 labels=range(19), normalize='true')
 
    ablated_f1_scores = []
    
    for comb, task_name in tasks_map.items():
        pred_file = os.path.join("result", "baselines", f"{DATASET_NAME}_nestedcv_predictions_{comb}_OURS.csv")
        df = pd.read_csv(pred_file)
        df['Real_ID'] = df['id'].apply(lambda row_idx: str(accessions[int(row_idx)]).strip())
        df = df.merge(status[['Accession', 'sex']], left_on='Real_ID', right_on='Accession', how='left')
         
        if comb == 0:
            # 19-Class: B_BC, BC, B_OV, OV, B_PR, PR
            excluded_indices = [0, 1, 11, 12, 14, 15] 
            total_classes = 19
        else: 
            excluded_indices = [0, 9, 11] 
            total_classes = 14
            
        shared_class_indices = [idx for idx in range(total_classes) if idx not in excluded_indices]
        
        for fold in df['outer_fold'].unique():
            fold_data = df[df['outer_fold'] == fold]
            for sex in ['Male', 'Female']:
                subset = fold_data[fold_data['sex'] == sex]
                if len(subset) >= 5:
                    true_classes_present = np.unique(subset['y_true'])
                    valid_labels = [c for c in shared_class_indices if c in true_classes_present]
                    
                    score = f1_score(
                        subset['y_true'], subset['y_pred'], 
                        labels=valid_labels, average='macro', zero_division=0
                    )
                    ablated_f1_scores.append({
                        'Sex': sex,
                        'F1': score,
                        'Task': task_name
                    })

    master_f1_df = pd.DataFrame(ablated_f1_scores)
 
    sns.set_theme(style="white", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(1, 3, figsize=(26, 8), gridspec_kw={'width_ratios': [1, 1, 0.7]})
 
    sns.heatmap(cm_male, annot=False, cmap='Blues', ax=axes[0], 
                xticklabels=CLASS_LABELS_19, yticklabels=CLASS_LABELS_19, vmin=0, vmax=1,
                linewidths=.5, linecolor='lightgray', square=True)
    axes[0].set_title(f'A. Male Cohort (n={len(df_male_19)})', pad=15, fontsize=20)
    axes[0].set_xlabel('Predicted Class', fontsize=20)
    axes[0].set_ylabel('True Class', fontsize=20)
    axes[0].tick_params(axis='both', which='major', labelsize=18)
    axes[0].set_xticklabels(CLASS_LABELS_19, rotation=45, ha='right')
 
    sns.heatmap(cm_female, annot=False, cmap='Blues', ax=axes[1], 
                xticklabels=CLASS_LABELS_19, yticklabels=CLASS_LABELS_19, vmin=0, vmax=1,
                linewidths=.5, linecolor='lightgray', square=True)
    axes[1].set_title(f'B. Female Cohort (n={len(df_female_19)})', pad=15, fontsize=20)
    axes[1].set_xlabel('Predicted Class', fontsize=20)
    axes[1].set_ylabel('') 
    axes[1].tick_params(axis='both', which='major', labelsize=18)
    axes[1].set_xticklabels(CLASS_LABELS_19, rotation=45, ha='right')
 
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    sns.barplot(
        data=master_f1_df, 
        x='Sex', y='F1', hue='Task',
        ax=axes[2], palette=task_palette, errorbar='sd', capsize=0.05, edgecolor=".2", alpha=0.9
    )
    axes[2].set_title('C. Performance on Shared Classes', pad=15, fontsize=20)
    axes[2].set_ylim(0.0, 1.05)
    axes[2].set_ylabel('Macro F1 Score (Mean ± SD)', fontsize=20)
    axes[2].set_xlabel('')
    axes[2].grid(True, axis='y', linestyle='-', alpha=0.6)
    axes[2].set_axisbelow(True)

    sns.despine(ax=axes[2], left=True, top=True, right=True)
    axes[2].yaxis.grid(True) 
    axes[2].xaxis.grid(False)
    axes[2].tick_params(axis='x', labelsize=20)
    axes[2].tick_params(axis='y', labelsize=18)
     
    y_bottom = 0.0
    for tick_idx, tick_label in enumerate(axes[2].get_xticklabels()):
        sex_name = tick_label.get_text()
        count = len(df_male_19) if sex_name == 'Male' else len(df_female_19)
        txt = axes[2].text(tick_idx, y_bottom + 0.02, f'n={count}', 
                           ha='center', va='bottom', fontsize=20, color='white', weight='bold')
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground='.2')])

    axes[2].legend(
        title="Diagnostic Task", 
        loc='lower center', 
        bbox_to_anchor=(0.8, 0.12), 
        framealpha=0.95,
        fontsize=14,
        title_fontsize=16,
        edgecolor='lightgray'
    )

    plt.suptitle('Diagnostic Confusion and Sex-Specific Performance Ablation', y=1.05, fontsize=30)
    plt.tight_layout()

    os.makedirs("figures", exist_ok=True)
    plt.savefig('figures/sex_panel.pdf', bbox_inches='tight')
    plt.show()
    print("\n--- Ablated Performance (Shared Classes Only) ---") 
    sex_stats_ablated = master_f1_df.groupby(['Task', 'Sex'])['F1'].agg(['mean', 'std']).reset_index()
     
    for _, row in sex_stats_ablated.iterrows():
        print(f"Task: {row['Task']} | Sex: {row['Sex']} | F1: {row['mean']:.3f} ± {row['std']:.3f}")




def plot_clevelant(): 
    MODELS = ['OURS', 'MLP', 'TABNET', 'HEAD', '1D', 'XGB', 'RF', 'LR']
    muted_palette = sns.color_palette("muted", n_colors=len(MODELS))
    model_colors = dict(zip(MODELS, muted_palette))

    tasks_map = {2: '14-Class (Tissue)', 0: '19-Class (Disease)'}
    comb_values = [2, 0]

    plot_data = []
 
    for comb in comb_values:
        task_label = tasks_map[comb]
        for baseline in MODELS:
            filename = f"{DATASET_NAME}_nestedcv_predictions_{comb}_{baseline}.csv"
            path = os.path.join("result", "baselines", filename)
            if os.path.exists(path):
                df = pd.read_csv(path)
                for f_idx in df['outer_fold'].unique():
                    fold_df = df[df['outer_fold'] == f_idx]
                    if not fold_df.empty:
                        score = f1_score(fold_df['y_true'], fold_df['y_pred'], average='macro')
                        plot_data.append({'Task': task_label, 'Model': baseline, 'F1': score})

    df_results = pd.DataFrame(plot_data)
    summary = df_results.groupby(['Task', 'Model'])['F1'].agg(['mean', 'std']).reset_index()
 
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
 
    fig, ax = plt.subplots(figsize=(15, 4.5)) 

    y_major_ticks = np.arange(len(comb_values)) 
    dodge_width = 0.35 
    offsets = np.linspace(dodge_width, -dodge_width, len(MODELS))

    for i, comb in enumerate(comb_values):
        task = tasks_map[comb]
        for j, model in enumerate(MODELS):
            row = summary[(summary['Task'] == task) & (summary['Model'] == model)]
            if not row.empty:
                m_val = row['mean'].values[0]
                s_val = row['std'].values[0]
                
                is_ours = (model == 'OURS')
                y_pos = y_major_ticks[i] + offsets[j]
                display_label = DISPLAY_NAMES.get(model, model) if model == 'OURS' or model == 'TABNET' else model 
                ax.errorbar(
                    x=m_val, y=y_pos, xerr=s_val, fmt='o',          
                    color=model_colors[model], 
                    label=display_label if i == 0 else "",
                    alpha=1.0, #if is_ours else 0.65, 
                    capsize=4, #if is_ours else 3, 
                    elinewidth=1.2, #if is_ours else 1.2,
                    markersize=8, #if is_ours else 6,
                    markeredgecolor='white' if is_ours else 'white',
                    markeredgewidth=1.2, #if is_ours else 0.5,
                    zorder=10 if is_ours else 3   
                )
 
    ax.set_yticks(y_major_ticks)
    ax.set_yticklabels([tasks_map[c] for c in comb_values], fontsize=13)
 
    ax.tick_params(axis='y', length=0, pad=10) 

    ax.set_xlabel('Macro F1 Score', fontsize=10, labelpad=10)
    ax.set_title('Model Performances Across tasks (Mean ± SD)', pad=15, fontsize=15)
 
    ax.set_xlim(0.6, 1.0)
 
    for y in y_major_ticks:
        ax.axhline(y, color='gray', linestyle='--', alpha=0.2, zorder=0)
 
    sns.despine(left=True, bottom=False)
    ax.spines['bottom'].set_color('#dddddd')
 
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles, labels, 
        loc='center left', 
        bbox_to_anchor=(1.02, 0.5),  
        frameon=False, 
        fontsize=10, 
        title="Models"
        # title_fontproperties={'weight':'bold'}
    )

    plt.tight_layout()
    plt.savefig("figures/performance_cleveland.pdf", bbox_inches='tight')
    plt.show()




def plot_ood(): 
    DATA_NAME = "miRNA"  
    COMB = 0              
    SCALES = ["noise_1.0", "noise_3.0", "noise_5.0", "covariate_shift"]
    SCALE_LABELS = ['Additive Noise\n($\sigma=1.0$)', 'Additive Noise\n($\sigma=3.0$)', 
                    'Additive Noise\n($\sigma=5.0$)', 'Covariate Shift\n(x5)']
    BASE_DIR = "result/baselines" 
     
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
 
    def calculate_shannon_entropy(probs):
        epsilon = 1e-12
        probs = np.vstack(probs).astype(np.float32) if probs.dtype == object else probs
        return -np.sum(probs * np.log(probs + epsilon), axis=1)

    def calculate_confidence(probs):
        probs = np.vstack(probs).astype(np.float32) if probs.dtype == object else probs
        return np.max(probs, axis=1)
 
    results_list = []
 
    for model in MODELS:
        file_path = f"{BASE_DIR}/{DATA_NAME}_OOD_probs_comb{COMB}_{model}.npz"
        if not os.path.exists(file_path):
            print(f"  [Warning] Skipping {model}: File not found")
            continue
            
        data = np.load(file_path, allow_pickle=True)
        
        for fold in range(1, 6):
            ent_id_key = f"fold_{fold}_real_entropy"
            prob_id_key = f"fold_{fold}_real_probs"
            
            if prob_id_key not in data: continue
             
            probs_id = data[prob_id_key]
            mean_conf_id = np.mean(calculate_confidence(probs_id))
            
            ent_id = data[ent_id_key] if (ent_id_key in data and data[ent_id_key] is not None) else calculate_shannon_entropy(probs_id)
                 
            for scale in SCALES:
                ent_ood_key = f"fold_{fold}_{scale}_entropy"
                prob_ood_key = f"fold_{fold}_{scale}_probs"
                
                if prob_ood_key in data and data[prob_ood_key].size > 0:
                    probs_ood = data[prob_ood_key]
                    mean_conf_ood = np.mean(calculate_confidence(probs_ood))
                     
                    gap = mean_conf_id - mean_conf_ood
                     
                    ent_ood = data[ent_ood_key] if (ent_ood_key in data and data[ent_ood_key] is not None) else calculate_shannon_entropy(probs_ood)
                    y_true = np.concatenate([np.zeros(len(ent_id)), np.ones(len(ent_ood))])
                    y_scores = np.concatenate([ent_id, ent_ood])
                    auroc = roc_auc_score(y_true, y_scores)
                    
                    results_list.append({
                        "Model": model,
                        "Fold": fold,
                        "Scale": scale,
                        "AUROC": auroc,
                        "Separation_Gap": gap
                    })

    df_results = pd.DataFrame(results_list)
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))  
    palette = sns.color_palette("muted", n_colors=len(MODELS))
 
    sns.barplot(
        data=df_results, x='Scale', y='AUROC', hue='Model',
        palette=palette, capsize=0.05, errwidth=1.5, ax=ax1
    )
    ax1.axhline(y=0.50, color='black', linestyle='--', linewidth=1.5, alpha=0.7)  
    ax1.axhline(y=0.0, color='black', linestyle='-', linewidth=1.2)  
    ax1.set_title("A. OOD Detection Performance", pad=15, fontsize=14)
    ax1.set_ylabel("AUROC (Higher is Better)")
    ax1.set_xlabel("") 
    ax1.set_ylim(-0.2, 1.05) 
    ax1.set_xticklabels(SCALE_LABELS)
    ax1.get_legend().remove() 
 
    sns.barplot(
        data=df_results, x='Scale', y='Separation_Gap', hue='Model',
        palette=palette, capsize=0.05, errwidth=1.5, ax=ax2
    )
    ax2.axhline(y=0.0, color='black', linestyle='-', linewidth=1.2)  
    ax2.set_title("B. Confidence Separation Gap", pad=15, fontsize=14)
    ax2.set_ylabel("Separation Gap (Higher is Safer)")
    ax2.set_xlabel("")  
    ax2.set_ylim(-0.1, 0.525) 
    ax2.set_xticklabels(SCALE_LABELS)
 
    handles, labels = ax2.get_legend_handles_labels() 
    new_labels = [DISPLAY_NAMES.get(label, label) for label in labels]
    ax2.legend(handles, new_labels, loc='upper right', title="Models", frameon=True, fontsize=10, title_fontsize=11)

    plt.tight_layout()
    plt.savefig(f"figures/ood.pdf", bbox_inches='tight')
    plt.show()






def plot_interpret(): 
    DATA_NAME = "miRNA"
    COMB = 0  
    REPRESENTATIVE_FOLD = 1 
    RAW_DATA_PATH = "data/sorted.csv"
    BASE_DIR = "result/baselines"
    MODEL_DIR = "result/saved_models"

    CLASS_NAMES_19 = ['B_BC', 'BC', 'BL', 'BT', 'CC', 'EC', 'GC', 'B_GL', 'GL', 
                    'HC', 'LK', 'B_OV', 'OV', 'PC', 'B_PR', 'PR', 'B_SA', 'SA', 'healthy']

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
 
    def get_fold_data_for_manifold(raw_csv_path, npz_path, fold_num=1):
        df = pd.read_csv(raw_csv_path)
        X = df.drop(columns=['label']).values
        y = df['label'].values 

        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(outer_cv.split(X, y))
        train_idx, test_idx = splits[fold_num - 1] 
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        npz_data = np.load(npz_path, allow_pickle=True)
        selected_features = npz_data[f"fold_{fold_num}_features"]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled[:, selected_features], y_train, X_test_scaled[:, selected_features], y_test

    def load_aggregated_predictions(data_name, comb):
        """FIXED: Loads the single CSV that contains all 5 folds."""
        path = os.path.join(BASE_DIR, f"{data_name}_nestedcv_predictions_{comb}_OURS.csv")
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df['y_true'].values, df['y_pred'].values
        else:
            raise FileNotFoundError(f"Could not find the predictions file: {path}")
 
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))
 
    npz_path = f"{BASE_DIR}/{DATA_NAME}_OOD_probs_comb{COMB}_OURS.npz"
    X_train, y_train, X_test, y_test = get_fold_data_for_manifold(RAW_DATA_PATH, npz_path, REPRESENTATIVE_FOLD)
    num_real_train = len(X_train)

    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    model_path = f"{MODEL_DIR}/{DATA_NAME}_comb{COMB}_fold{REPRESENTATIVE_FOLD}_OURS.keras"
    extractor_model = tf.keras.models.load_model(model_path, compile=False)
    extractor = tf.keras.Model(inputs=extractor_model.input, outputs=extractor_model.get_layer('embedding').output)

    sph_train_smote = tf.math.l2_normalize(extractor.predict(X_train_smote, verbose=0), axis=1).numpy()
    sph_test = tf.math.l2_normalize(extractor.predict(X_test, verbose=0), axis=1).numpy()
    patient_data = np.vstack([sph_train_smote, sph_test])

    tsne = OpenTSNE(n_components=2, metric="cosine", perplexity=45, initialization="pca", early_exaggeration=12.0, random_state=42, n_jobs=-1)
    patient_embedding = tsne.fit(patient_data)

    tsne_train_smote = np.array(patient_embedding[:len(X_train_smote)])
    tsne_test = np.array(patient_embedding[len(X_train_smote):])

    num_classes = len(CLASS_NAMES_19)
    cmap = plt.get_cmap('tab20', num_classes)
    ax1.grid(True, linestyle='--', alpha=0.3)

    for i, cls_name in enumerate(CLASS_NAMES_19):
        color = cmap(i)
        class_mask_smote = (y_train_smote == i)
        mask_real, mask_fake = class_mask_smote.copy(), class_mask_smote.copy()
        mask_real[num_real_train:] = False  
        mask_fake[:num_real_train] = False  
        
        ax1.scatter(tsne_train_smote[mask_fake, 0], tsne_train_smote[mask_fake, 1], color=color, alpha=0.1, s=20, marker='o', edgecolor='none', zorder=0)
        ax1.scatter(tsne_train_smote[mask_real, 0], tsne_train_smote[mask_real, 1], color=color, alpha=0.75, s=50, marker='o', edgecolor='white', linewidth=0.5, zorder=1)
        
        test_mask = (y_test == i)
        ax1.scatter(tsne_test[test_mask, 0], tsne_test[test_mask, 1], color=color, alpha=0.9, s=40, marker='o', edgecolor='black', linewidth=0.5, zorder=2)
        
    label_coords = np.zeros((num_classes, 2))
    for i in range(num_classes):
        class_points = tsne_train_smote[y_train_smote == i]
        if len(class_points) > 0:
            label_coords[i] = np.median(class_points, axis=0)
            ax1.text(label_coords[i, 0], label_coords[i, 1], CLASS_NAMES_19[i], fontsize=12, weight='bold', color='black', ha='center', va='center', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2', alpha=0.8), zorder=11)

    ax1.set_title('A. Diagnostic Manifold (Cosine Projection)', fontsize=25, pad=10)
    ax1.set_xlabel('t-SNE Dimension 1 ', fontsize=20)
    ax1.set_ylabel('t-SNE Dimension 2 ', fontsize=20)
    ax1.tick_params(axis='both', which='major', labelsize=15)

    custom_lines = [
        Line2D([0], [0], marker='.', color='w', markerfacecolor='gray', alpha=0.3, markersize=15, label='Synthetic Boundary'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='white', markersize=9, label='Real Training Data'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markeredgecolor='black', markersize=8, label='Hold-out Test Data', linewidth=0.1)
    ]
    ax1.legend(handles=custom_lines, loc='upper right', fontsize=14, frameon=True)
 
    y_true_agg, y_pred_agg = load_aggregated_predictions(DATA_NAME, COMB)
    cm = confusion_matrix(y_true_agg, y_pred_agg, labels=np.arange(num_classes))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    annot_matrix = np.empty_like(cm_normalized, dtype=object)
    for r in range(cm_normalized.shape[0]):
        for c in range(cm_normalized.shape[1]):
            annot_matrix[r, c] = f"{cm_normalized[r, c]:.2f}" if cm_normalized[r, c] >= 0.005 else ""

    heatmap = sns.heatmap(cm_normalized, annot=annot_matrix, fmt='', annot_kws={"fontsize": 10}, xticklabels=CLASS_NAMES_19, yticklabels=CLASS_NAMES_19, cmap="Blues", vmin=0.0, vmax=1.0, ax=ax2, cbar_kws={'label': 'Proportion of Predictions'})
 
    cb = heatmap.collections[0].colorbar
    cb.ax.yaxis.label.set_size(16)   
    cb.ax.tick_params(labelsize=12)  
 
    ax2.set_xticks(np.arange(len(CLASS_NAMES_19)) + 0.5)
    ax2.set_yticks(np.arange(len(CLASS_NAMES_19)) + 0.5)
    ax2.set_xticklabels(CLASS_NAMES_19, rotation=45) 
    ax2.set_yticklabels(CLASS_NAMES_19, rotation=0)
    ax2.tick_params(axis='both', which='major', labelsize=15)

    ax2.set_title(f"B. Diagnostic Phenotype Aggregated Errors", fontsize=25, pad=10)
    ax2.set_ylabel('True Label', fontsize=20)
    ax2.set_xlabel('Predicted Label', fontsize=20)

    plt.tight_layout()
    plt.savefig(f"figures/visual_eval.pdf", bbox_inches='tight')
    plt.show()




def plot_dataset_overview(): 
    CLASS_NAMES_19 = [
        'B_BC', 'BC', 'BL', 'BT', 'CC', 'EC', 'GC', 'B_GL', 'GL', 
        'HC', 'LK', 'B_OV', 'OV', 'PC', 'B_PR', 'PR', 'B_SA', 'SA', 'healthy'
    ]
    
    df = pd.read_csv('data/sorted.csv')
    X_raw = df.drop(columns=['label']).values
    y_raw = df['label'].values.astype(int)

    num_classes = len(CLASS_NAMES_19)
    cmap = plt.get_cmap('tab20', num_classes)
    colors = [cmap(i) for i in range(num_classes)]
 
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)
    fig, axes = plt.subplots(1, 2, figsize=(22, 9), gridspec_kw={'width_ratios': [1, 1.2]})
 
    counts = [np.sum(y_raw == i) for i in range(num_classes)]
    
    axes[0].bar(CLASS_NAMES_19, counts, color=colors, edgecolor='black', linewidth=0.5, alpha=0.9)
    axes[0].set_title('A. Data Distribution (Sample Counts)', fontsize=25, pad=15)
    axes[0].set_ylabel('Number of Samples', fontsize=20)
    axes[0].tick_params(axis='y', labelsize=15)
    axes[0].tick_params(axis='x', rotation=45, labelsize=15)
    
    max_count = max(counts)
    for i, count in enumerate(counts):
        axes[0].text(i, count + (max_count * 0.015), str(count), 
                     ha='center', va='bottom', fontsize=12, color='.2')
 
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    tsne = OpenTSNE(n_components=2, metric="euclidean", perplexity=45, 
                    initialization="pca", random_state=42, n_jobs=-1)
    X_tsne = tsne.fit(X_scaled)

    axes[1].grid(True, linestyle='--', alpha=0.3)

    for i, cls_name in enumerate(CLASS_NAMES_19):
        mask = (y_raw == i)
        axes[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], color=colors[i], 
                        alpha=1, s=40, marker='o', linewidth=0.5, edgecolor='white')

    axes[1].set_title('B. Raw Data Manifold (t-SNE Projection)', fontsize=25, pad=15)
    axes[1].set_xlabel('t-SNE Dimension 1', fontsize=20)
    axes[1].set_ylabel('t-SNE Dimension 2', fontsize=20)
    axes[1].tick_params(axis='both', labelsize=15)
 
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=colors[i], 
                              markersize=10, label=CLASS_NAMES_19[i]) for i in range(num_classes)]
     
    axes[1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                   title="Label", framealpha=0.95, title_fontsize='20', fontsize='15')
 
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig('figures/data.pdf', bbox_inches='tight')
    plt.show()


def plot_latex_table():
    warnings.filterwarnings('ignore')

    DATA_NAME = 'miRNA'
    RAW_DATA_PATH = "data/sorted.csv"
    TASKS = {2: 'Tissue-of-Origin', 0: 'Disease'}

    MODEL_MAP_TOP = {
        'LR': 'Logistic Regression',
        'RF': 'Random Forest',
        'XGB': 'XGBoost',
        '1D': '1D-CNN',
        'HEAD': 'HEAD Ensemble',
        'TABNET': 'TabNet'
    }
    MODEL_MAP_BOTTOM = {
        'MLP': 'Vanilla MLP',
        'OURS': '\\textbf{OURS}'
    }

    results = {task: {} for task in TASKS}

    def get_y_true(task_id, fold_num):
        """Recreates the exact true labels for a given fold based on your data mapping."""
        df = pd.read_csv(RAW_DATA_PATH)
        y_raw = df['label'].copy()
        
        if task_id == 1:
            mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 0, 8: 7, 9: 8, 10: 9, 11: 0, 12: 10, 13: 11, 14: 0, 15: 12, 16: 0, 17: 13, 18: 14}
            y = y_raw.map(mapping).values
        elif task_id == 2:
            mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 6, 9: 7, 10: 8, 11: 9, 12: 9, 13: 10, 14: 11, 15: 11, 16: 12, 17: 12, 18: 13}
            y = y_raw.map(mapping).values
        else: # Task 0
            y = y_raw.values
            
        outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        splits = list(outer_cv.split(np.zeros((len(y), 1)), y))
        _, test_idx = splits[fold_num - 1]
        
        return y[test_idx]


    for task_id in TASKS.keys():
        for model_code in list(MODEL_MAP_TOP.keys()) + list(MODEL_MAP_BOTTOM.keys()):
            f1_list, bacc_list, auroc_list = [], [], []
            
            npz_path = f"result/baselines/{DATA_NAME}_OOD_probs_comb{task_id}_{model_code}.npz"
            
            if os.path.exists(npz_path):
                probs_data = np.load(npz_path, allow_pickle=True)
                
                for fold in range(1, 6):
                    prob_key = f"fold_{fold}_real_probs"
                    if prob_key in probs_data:
                        probs = probs_data[prob_key]
                        if probs.dtype == object: 
                            probs = np.vstack(probs).astype(float)
                        
                        y_true_key = f"fold_{fold}_y_true"
                        if y_true_key in probs_data:
                            y_true = probs_data[y_true_key]
                        else:
                            y_true = get_y_true(task_id, fold)
                        
                        y_pred = np.argmax(probs, axis=1)
                        
                        f1_list.append(f1_score(y_true, y_pred, average='macro'))
                        bacc_list.append(balanced_accuracy_score(y_true, y_pred))
                        
                        try:
                            safe_probs = probs.astype(float)
                            
                            is_valid_probs = (
                                not np.isnan(safe_probs).any() and 
                                not np.isinf(safe_probs).any() and 
                                np.all(safe_probs >= 0.0) and 
                                np.all(safe_probs <= 1.0) and 
                                np.allclose(np.sum(safe_probs, axis=1), 1.0, atol=0.01)
                            )
                            
                            if not is_valid_probs:
                                safe_probs = np.nan_to_num(safe_probs, nan=0.0, posinf=1.0, neginf=0.0)
                                safe_probs = softmax(safe_probs, axis=1)
                            
                            auroc_list.append(roc_auc_score(y_true, safe_probs, multi_class='ovr', average='macro'))
                        except ValueError:
                            auroc_list.append(np.nan) 
                    
                results[task_id][model_code] = {
                    'f1': (np.nanmean(f1_list), np.nanstd(f1_list)) if f1_list else None,
                    'bacc': (np.nanmean(bacc_list), np.nanstd(bacc_list)) if bacc_list else None,
                    'auroc': (np.nanmean(auroc_list), np.nanstd(auroc_list)) if auroc_list else None
                }
            else:
                results[task_id][model_code] = None

    def fmt(stats):
        if stats is None or np.isnan(stats[0]): return "---"
        return f"{stats[0]:.3f} $\\pm$ {stats[1]:.3f}"

    latex = [
        "\\begin{table*}[t]",
        "    \\centering",
        "    \\caption{Multiclass Diagnostic Performance on Serum miRNA across baselines. Metrics are presented as Mean $\\pm$ Standard Deviation across the 5 outer test folds. Best results in \\textbf{bold} and second best results \\underline{underlined}.}",
        "    \\label{tab:performance}",
        "    \\setlength{\\tabcolsep}{4pt} % Compress columns slightly to fit \pm",
        "    \\begin{tabular}{@{}lcccccc@{}}",
        "    \\toprule",
        "    & \\multicolumn{3}{c}{\\textbf{Tissue-of-Origin (14 Classes)}} & \\multicolumn{3}{c}{\\textbf{Disease (19 Classes)}} \\\\",
        "    \\cmidrule(lr){2-4} \\cmidrule(l){5-7}",
        "    \\textbf{Model} & \\textbf{Macro F1} & \\textbf{Bal. Acc} & \\textbf{AUROC} & \\textbf{Macro F1} & \\textbf{Bal. Acc} & \\textbf{AUROC} \\\\",
        "    \\midrule"
    ]

    def generate_rows(model_dict):
        for code, name in model_dict.items():
            row = [name]
            for task_id in [2, 0]:
                data = results[task_id].get(code)
                if data:
                    row.extend([fmt(data['f1']), fmt(data['bacc']), fmt(data['auroc'])])
                else:
                    row.extend(["---", "---", "---"])
            latex.append("    " + " & ".join(row) + " \\\\")

    generate_rows(MODEL_MAP_TOP)
    latex.append("    \\midrule")
    generate_rows(MODEL_MAP_BOTTOM)
    latex.extend(["    \\bottomrule", "    \\end{tabular}", "\\end{table*}"])

    print("\n".join(latex))