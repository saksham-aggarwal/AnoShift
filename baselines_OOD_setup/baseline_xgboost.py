# XGBoost Anomaly Detection on AnoShift Dataset

## 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc
import xgboost as xgb
import time
import os
import sys

# Set random seed for reproducibility
np.random.seed(42)

def load_train_year(anoshift_db_path, year):
    if year <= 2010:
        df = pd.read_parquet(os.path.join(anoshift_db_path, f'subset/{year}_subset.parquet'))
    else:
        sys.exit(-1)
    df = df.reset_index(drop=True)

    print(f"Loading {year} subset for train", df)
    return df

def load_test_year(anoshift_db_path, year):
    if year <= 2010:
        df = pd.read_parquet(os.path.join(anoshift_db_path, f'subset/{year}_subset_valid.parquet'))
    else:
        df = pd.read_parquet(os.path.join(anoshift_db_path, f'subset/{year}_subset.parquet'))
    df = df.reset_index(drop=True)
    
    print(f"Loading {year} subset for test", df)
    return df 

def rename_columns(df):
    categorical_cols = ["0", "1", "2", "3", "13"]
    numerical_cols = ["4", "5", "6", "7", "8", "9", "10", "11", "12"]
    additional_cols = ["14", "15", "16", "17", "19"]
    label_col = ["18"]

    new_names = []
    for col_name in df.columns.values:
        if col_name in numerical_cols:
            df[col_name] = pd.to_numeric(df[col_name])
            new_names.append((col_name, "num_" + col_name))
        elif col_name in categorical_cols:
            new_names.append((col_name, "cat_" + col_name))
        elif col_name in additional_cols:
            new_names.append((col_name, "bonus_" + col_name))
        elif col_name in label_col:
            df[col_name] = pd.to_numeric(df[col_name])
            new_names.append((col_name, "label"))
        else:
            new_names.append((col_name, col_name))
    df.rename(columns=dict(new_names), inplace=True)

    print("Renamed columns:", df)

    return df

def preprocess(df, enc=None):
    if not enc:
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(df.loc[:,['cat_' in i for i in df.columns]])
    
    num_cat_features = enc.transform(df.loc[:,['cat_' in i for i in df.columns]]).toarray()

    df_catnum = pd.DataFrame(num_cat_features)
    df_catnum = df_catnum.add_prefix('catnum_')

    df = df.reset_index(drop=True)
    df_new = pd.concat([df, df_catnum], axis=1)
   
    df_new.loc[df_new['label'] < 0, 'label'] = -1
    df_new['label'].replace({1:0}, inplace=True)
    df_new['label'].replace({-1:1}, inplace=True)

    print(df_new, enc)
    
    return df_new, enc

def get_train(anoshift_db_path, train_data_percent=1):
    dfs = []

    for year in train_years: 
        df_year = load_train_year(anoshift_db_path, year)
        dfs.append(df_year)
    
    df_all_years = pd.concat(dfs, ignore_index=True)
    df_all_years = rename_columns(df_all_years)
    df_new, ohe_enc = preprocess(df_all_years)    

    # Get all feature columns (numerical and one-hot encoded categorical)
    feature_cols = [col for col in df_new.columns if col.startswith('num_') or col.startswith('catnum_')]
    
    # Sample a fraction of data if requested
    df_sampled = df_new.sample(frac=train_data_percent)
    
    # Extract features and labels for both normal and anomalous samples
    X_train = df_sampled[feature_cols].to_numpy()
    y_train = df_sampled['label'].to_numpy()
    
    # Calculate statistics (can still be based on normal samples only if preferred)
    normal_samples = df_sampled[df_sampled['label'] == 0]
    normal_features = normal_samples[feature_cols].to_numpy()
    data_mean = normal_features.mean(0)[None,:]
    data_std = normal_features.std(0)[None,:]
    data_std[data_std==0] = 1

    return X_train, y_train, ohe_enc, data_mean, data_std, df_new

def get_n_test_splits():
    return len(test_years)

# def get_test(anoshift_db_path, idx, ohe_enc):
#     year = test_years[idx]
    
#     df_year = load_test_year(anoshift_db_path, year)
#     df_year = rename_columns(df_year)
#     df_test, _ = preprocess(df_year, ohe_enc)
#     numerical_cols = df_test.columns.to_numpy()[['num_' in i for i in df_test.columns]]
#     X_test = df_test[numerical_cols].to_numpy()
#     y_test = df_test["label"].to_numpy()

#     X_test = np.nan_to_num(X_test)

#     print("X_test shape:", X_test)
#     print("y_test shape:", y_test)
#     print("df_test shape:", df_test)
 
#     return X_test, y_test, df_test

def get_test(anoshift_db_path, idx, ohe_enc):
    year = test_years[idx]
    
    df_year = load_test_year(anoshift_db_path, year)
    df_year = rename_columns(df_year)
    df_test, _ = preprocess(df_year, ohe_enc)
    
    # Use the same feature selection approach as get_train()
    feature_cols = [col for col in df_test.columns if col.startswith('num_') or col.startswith('catnum_')]
    X_test = df_test[feature_cols].to_numpy()
    y_test = df_test["label"].to_numpy()

    X_test = np.nan_to_num(X_test)

    print("X_test shape:", X_test.shape)  # Changed to print shape, not the array itself
    print("y_test shape:", y_test.shape)  # Changed to print shape
    print("df_test shape:", df_test.shape)  # Changed to print shape
 
    return X_test, y_test, df_test

# Function to check if the path exists
def check_anoshift_path(path):
    if not os.path.exists(path):
        print(f"ERROR: AnoShift database path '{path}' does not exist.")
        print("Please set the correct path to the AnoShift database.")
        return False
    
    print(f"INFO: AnoShift database path '{path}' exists.")
    return True

# Load and preprocess the training data
def load_and_preprocess_data(anoshift_db_path):
    if not check_anoshift_path(anoshift_db_path):
        return None, None, None, None, None
    
    try:
        print("Loading and preprocessing AnoShift training data...")
        X_train_iso, y_train_iso, ohe_enc, data_mean, data_std, df_train = get_train(anoshift_db_path)
        
        # Extract features for XGBoost (including both numerical and one-hot encoded categorical features)
        feature_cols = [col for col in df_train.columns if col.startswith('num_') or col.startswith('catnum_')]
        
        # Split data into training and validation sets (80/20 split)
        from sklearn.model_selection import train_test_split
        
        # Use stratified sampling to maintain class distribution
        train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42, stratify=df_train['label'])
        # train_df = df_train
        # val_df = df_train
        
        # Create proper training set with both normal and abnormal samples
        X_train_all = train_df[feature_cols]
        y_train = train_df["label"].values
        
        # Create validation set
        X_val = val_df[feature_cols]
        y_val = val_df["label"].values
        
        print(f"Training data shape: {X_train_all.shape}")
        print(f"Training labels distribution: Normal: {(y_train == 0).sum()}, Abnormal: {(y_train == 1).sum()}")
        print(f"Anomaly percentage in training: {100 * y_train.mean():.2f}%")
        
        print(f"Validation data shape: {X_val.shape}")
        print(f"Validation labels distribution: Normal: {(y_val == 0).sum()}, Abnormal: {(y_val == 1).sum()}")
        print(f"Anomaly percentage in validation: {100 * y_val.mean():.2f}%")
        
        return X_train_all, y_train, X_val, y_val, ohe_enc
    
    except Exception as e:
        print(f"Error loading AnoShift dataset: {str(e)}")
        return None, None, None, None, None
    
# Load first test split for evaluation
def load_test_data(anoshift_db_path, ohe_enc, test_idx=0):
    if not check_anoshift_path(anoshift_db_path):
        return None, None, None
    
    try:
        print(f"Loading test data for year {test_years[test_idx]}...")
        X_test, y_test, df_test = get_test(anoshift_db_path, test_idx, ohe_enc)
        
        # Get feature columns (both numerical and one-hot encoded)
        feature_cols = [col for col in df_test.columns if col.startswith('num_') or col.startswith('catnum_')]
        
        # Extract proper feature set
        X_test_all = df_test[feature_cols]
        
        print(f"Test data shape: {X_test_all.shape}")
        print(f"Anomaly percentage in test: {100 * y_test.mean():.2f}%")
        
        return X_test_all, y_test, df_test
    
    except Exception as e:
        print(f"Error loading test data: {str(e)}")
        return None, None, None
    
def build_xgboost_model(X_train, y_train, X_val, y_val):
    if X_train is None or X_val is None:
        return None
    
    # Calculate class weights to handle imbalance based on training data
    normal_count = np.sum(y_train == 0)
    anomaly_count = np.sum(y_train == 1)
    
    if anomaly_count > 0:
        scale_pos_weight = normal_count / anomaly_count
    else:
        scale_pos_weight = 100  # Default if no anomalies in training
    
    print(f"Class imbalance ratio (normal:anomaly): {scale_pos_weight:.2f}")
    print(f"Training data contains {normal_count} normal samples and {anomaly_count} anomalies")

    # Calculate class weights to handle imbalance based on training data
    normal_count_val = np.sum(y_val == 0)
    anomaly_count_val = np.sum(y_val == 1)
    print(f"Validation data contains {normal_count_val} normal samples and {anomaly_count_val} anomalies")
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Set parameters - optimized for imbalanced binary classification
    params = {
        'objective': 'binary:logistic',
        'eval_metric': ['auc', 'logloss', 'error'],
        'scale_pos_weight': scale_pos_weight,
        'max_depth': 6,
        'eta': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 1,
        'gamma': 0.1,  # Minimum loss reduction for further partition
        'gpu_id': -1,  # Set to appropriate GPU ID if available, -1 for CPU
        'tree_method': 'hist'  # Use 'gpu_hist' for GPU
    }
    
    # Set evaluation list
    evallist = [(dtrain, 'train'), (dval, 'validation')]
    
    # Train model
    print("Training XGBoost model for supervised anomaly detection...")
    start_time = time.time()
    
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,
        evals=evallist,
        early_stopping_rounds=20,
        verbose_eval=50
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Basic model statistics
    y_train_pred = model.predict(dtrain)
    train_accuracy = np.mean((y_train_pred > 0.5) == y_train)
    
    y_val_pred = model.predict(dval)
    val_accuracy = np.mean((y_val_pred > 0.5) == y_val)
    
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    
    return model

## 6. Evaluate Model Performance
def evaluate_model(model, X_val, y_val, X_test, y_test):
    if model is None or X_val is None or X_test is None:
        return None
    
    # Convert to pandas DataFrame if numpy arrays
    if isinstance(X_val, np.ndarray):
        X_val = pd.DataFrame(X_val)
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test)
    
    # Create DMatrix for validation and test data
    dval = xgb.DMatrix(X_val)
    dtest = xgb.DMatrix(X_test)
    
    # Make predictions
    val_preds = model.predict(dval)
    test_preds = model.predict(dtest)
    
    # Convert probabilities to binary predictions with default threshold
    val_pred_binary = (val_preds > 0.5).astype(int)
    test_pred_binary = (test_preds > 0.5).astype(int)
    
    # Function to calculate and display macro F1 score and class-wise F1 scores
    def calculate_macro_f1(y_true, y_pred, set_name):
        from sklearn.metrics import f1_score
        
        # Calculate F1 score with macro averaging
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        # Calculate class-wise F1 scores
        class_0_f1 = f1_score(y_true, y_pred, pos_label=0, average='binary')
        class_1_f1 = f1_score(y_true, y_pred, pos_label=1, average='binary')
        
        print(f"\n{set_name} F1 Scores:")
        print(f"  Class 0 (Normal) F1 Score: {class_0_f1:.4f}")
        print(f"  Class 1 (Anomaly) F1 Score: {class_1_f1:.4f}")
        print(f"  Macro F1 Score: {macro_f1:.4f}")
        
        return macro_f1, class_0_f1, class_1_f1
    
    # Validation set metrics
    print("Validation Set Results:")
    print(confusion_matrix(y_val, val_pred_binary))
    print(classification_report(y_val, val_pred_binary))
    
    if np.unique(y_val).shape[0] > 1:  # Check if we have both classes
        val_auc = roc_auc_score(y_val, val_preds)
        print(f"Validation AUC: {val_auc:.4f}")
        # Calculate macro F1 for validation set
        val_macro_f1, val_normal_f1, val_anomaly_f1 = calculate_macro_f1(y_val, val_pred_binary, "Validation")
    
    # Test set metrics
    print("\nTest Set Results:")
    print(confusion_matrix(y_test, test_pred_binary))
    print(classification_report(y_test, test_pred_binary))
    
    if np.unique(y_test).shape[0] > 1:  # Check if we have both classes
        test_auc = roc_auc_score(y_test, test_preds)
        print(f"Test AUC: {test_auc:.4f}")
        # Calculate macro F1 for test set
        test_macro_f1, test_normal_f1, test_anomaly_f1 = calculate_macro_f1(y_test, test_pred_binary, "Test")
    
    # Plot ROC and Precision-Recall curves
    # plt.figure(figsize=(15, 12))  # Increased height for additional subplot
    
    # # If we have both classes in validation set
    # if np.unique(y_val).shape[0] > 1:
    #     # Precision-Recall curve for validation
    #     plt.subplot(221)
    #     precision, recall, _ = precision_recall_curve(y_val, val_preds)
    #     pr_auc = auc(recall, precision)
    #     plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Validation Precision-Recall Curve')
    #     plt.legend()
    #     plt.grid(True)
    
    # # If we have both classes in test set
    # if np.unique(y_test).shape[0] > 1:
    #     # Precision-Recall curve for test
    #     plt.subplot(222)
    #     precision, recall, _ = precision_recall_curve(y_test, test_preds)
    #     pr_auc = auc(recall, precision)
    #     plt.plot(recall, precision, label=f'PR AUC = {pr_auc:.4f}')
    #     plt.xlabel('Recall')
    #     plt.ylabel('Precision')
    #     plt.title('Test Precision-Recall Curve')
    #     plt.legend()
    #     plt.grid(True)
        
    #     # Add F1 score visualization
    #     plt.subplot(223)
    #     # Bar chart of F1 scores
    #     f1_scores = {
    #         'Normal (Val)': val_normal_f1,
    #         'Anomaly (Val)': val_anomaly_f1,
    #         'Macro (Val)': val_macro_f1,
    #         'Normal (Test)': test_normal_f1,
    #         'Anomaly (Test)': test_anomaly_f1,
    #         'Macro (Test)': test_macro_f1
    #     }
        
    #     x = range(len(f1_scores))
    #     plt.bar(x, f1_scores.values())
    #     plt.xticks(x, f1_scores.keys(), rotation=45)
    #     plt.ylabel('F1 Score')
    #     plt.title('F1 Scores Comparison')
        
    #     # Add prediction distribution
    #     plt.subplot(224)
    #     plt.hist(test_preds[y_test==0], bins=30, alpha=0.5, label='Normal', density=True)
    #     plt.hist(test_preds[y_test==1], bins=30, alpha=0.5, label='Anomaly', density=True)
    #     plt.axvline(x=0.5, color='r', linestyle='--', label='Default Threshold')
    #     plt.xlabel('Prediction Score')
    #     plt.ylabel('Density')
    #     plt.title('Test Prediction Distribution')
    #     plt.legend()
    
    # plt.tight_layout()
    # plt.show()
    
    # Store evaluation metrics
    eval_metrics = {
        'threshold': 0.5,  # Default threshold
        'test_auc': test_auc if np.unique(y_test).shape[0] > 1 else None,
        'test_macro_f1': test_macro_f1 if np.unique(y_test).shape[0] > 1 else None,
        'test_normal_f1': test_normal_f1 if np.unique(y_test).shape[0] > 1 else None,
        'test_anomaly_f1': test_anomaly_f1 if np.unique(y_test).shape[0] > 1 else None
    }
    
    return test_preds, eval_metrics

## 8. Anomaly Threshold Analysis
def analyze_anomaly_thresholds(y_test, test_preds):
    if test_preds is None or y_test is None:
        return None, None
    
    threshold = 0.5
    
    # Calculate metrics for each threshold
    results = []
    pred_binary = (test_preds > threshold).astype(int)
    
    # Calculate traditional metrics
    tn, fp, fn, tp = confusion_matrix(y_test, pred_binary, labels=[0,1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    
    # Calculate macro F1 score
    from sklearn.metrics import f1_score
    macro_f1 = f1_score(y_test, pred_binary, average='macro')
    
    # Calculate class-wise F1 scores
    normal_f1 = f1_score(y_test, pred_binary, pos_label=0, average='binary')
    anomaly_f1 = f1_score(y_test, pred_binary, pos_label=1, average='binary')
    
    results.append({
        'Threshold': threshold,
        'Precision': precision,
        'Recall': recall,
        'Specificity': specificity,
        'F1': f1,
        'Macro_F1': macro_f1,
        'Normal_F1': normal_f1,
        'Anomaly_F1': anomaly_f1,
        'TP': tp,
        'FP': fp,
        'TN': tn,
        'FN': fn,
        'FPR': fpr,
    })
    
    return results

if __name__ == "__main__":

    #Add start time
    start_time = time.time()
    print("Starting XGBoost Anomaly Detection on AnoShift Dataset...")

    train_years = [2006, 2007, 2008, 2009, 2010]
    test_years = [2014, 2015]

    anoshift_db_path = '/Users/sakshamaggarwal/Documents/Git_Repos/cybersec_research/AnoShift-RIT/datasets/Kyoto-2016_AnoShift/'  # Change this to your AnoShift database path

    # Load the data
    X_train, y_train, X_val, y_val, ohe_enc = load_and_preprocess_data(anoshift_db_path)

    xgb_model = build_xgboost_model(X_train, y_train, X_val, y_val)

    for idx in range(len(test_years)):
    # Load test data for the current year
        X_test, y_test, df_test = load_test_data(anoshift_db_path, ohe_enc, test_idx=idx)
        
        # Evaluate model on the test data
        print(f"Evaluating model on test data for year {test_years[idx]}...")
        test_predictions, eval_metrics = evaluate_model(xgb_model, X_val, y_val, X_test, y_test)
        
        # Print evaluation metrics
        print(f"Test Predictions for year {test_years[idx]}: {test_predictions}")
        print(f"Evaluation Metrics for year {test_years[idx]}: {eval_metrics}")

        # Analyze anomaly thresholds
        threshold_results = analyze_anomaly_thresholds(y_test, test_predictions)
        print(f"Threshold Analysis Results for year {test_years[idx]}: {threshold_results}")

        # Write all the above information to a file year-wise.
        with open(f"anoshift_5y_subset_results_{test_years[idx]}.txt", "w") as f:
            f.write(f"Test Predictions for year {test_years[idx]}: {test_predictions}\n")
            f.write(f"Evaluation Metrics for year {test_years[idx]}: {eval_metrics}\n")
            f.write(f"Threshold Analysis Results for year {test_years[idx]}: {threshold_results}\n")

    # Print total execution time
    end_time = time.time()
    total_time = end_time - start_time

    print(f"Total execution time: {total_time:.2f} seconds")
    print("XGBoost Anomaly Detection on AnoShift Dataset completed.")
