import os
import pandas as pd
import kagglehub

def load_metabric_data():
    """
    Downloads the METABRIC dataset using kagglehub and returns a preprocessed DataFrame.
    """
    # Download dataset
    path = kagglehub.dataset_download("raghadalharbi/breast-cancer-gene-expression-profiles-metabric")
    
    # Load dataset
    df = pd.read_csv(os.path.join(path, "METABRIC_RNA_Mutation.csv"))
    
    # Preprocessing logic matching the notebook
    survival_time_col = 'overall_survival_months'
    event_col = 'overall_survival'
    clinical_features = [
        'age_at_diagnosis', 'tumor_size', 'lymph_nodes_examined_positive',
        'neoplasm_histologic_grade', 'mutation_count',
        'chemotherapy', 'hormone_therapy', 'radio_therapy',
        'type_of_breast_surgery', 'cancer_type_detailed', 'cellularity',
        'pam50_+_claudin-low_subtype', 'er_status', 'her2_status',
        'tumor_other_histologic_subtype', 'inferred_menopausal_state', 'integrative_cluster'
    ]
    
    available_features = [col for col in clinical_features if col in df.columns]
    necessary_columns = [survival_time_col, event_col] + available_features
    all_columns = set(df.columns)
    unnecessary_columns = all_columns - set(necessary_columns)
    
    # Identify high missing columns
    missing_percent = df.isnull().sum() / len(df) * 100
    high_missing_cols = missing_percent[missing_percent > 50].index
    
    drop_columns = list(set(list(high_missing_cols) + list(unnecessary_columns)))
    drop_columns = [col for col in drop_columns if col not in [survival_time_col, event_col]]
    
    data_reduced = df.drop(columns=drop_columns)
    
    # Drop highly correlated feature as in notebook
    if 'nottingham_prognostic_index' in data_reduced.columns:
        data_reduced = data_reduced.drop(columns=['nottingham_prognostic_index'])
        
    # Drop NaN values for clean training
    data_reduced = data_reduced.dropna()
    
    return data_reduced

if __name__ == "__main__":
    df = load_metabric_data()
    print(f"Loaded dataset with shape: {df.shape}")
