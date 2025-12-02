#!/usr/bin/env python
# coding: utf-8

"""
Loan Data Analysis and Modeling Script

This script performs data cleaning, preprocessing, feature engineering,
and model evaluation on a loan dataset. It compares Logistic Regression,
Decision Tree, and Random Forest models on PCA-transformed, scaled, and
raw data.
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.base import clone

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def load_data(filepath):
    """
    Loads the dataset from the specified CSV file.
    """
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None

def clean_data(df):
    """
    Handles missing values and data type conversions.
    """
    df = df.copy()
    
    # Handle 'Dependents' column
    # '3+' is replaced by '3'. Missing values filled with mode. Converted to int.
    if 'Dependents' in df.columns:
        mode_dep = df['Dependents'].mode()[0]
        df['Dependents'] = df['Dependents'].replace('3+', '3').fillna(mode_dep).astype('int32')
    
    # Fill categorical columns with mode
    category_col = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']
    for column in category_col:
        if column in df.columns:
            mode_val = df[column].mode()[0]
            df[column].fillna(mode_val, inplace=True)
        
    # Fill numeric-like categorical columns with mode
    numeric_category_col = ['Credit_History', 'Loan_Amount_Term']
    for column in numeric_category_col:
        if column in df.columns:
            mode_val = df[column].mode()[0]
            df[column].fillna(mode_val, inplace=True)

    # Fill LoanAmount with median (due to skewed distribution)
    if 'LoanAmount' in df.columns:
        median_loan = df['LoanAmount'].median()
        df['LoanAmount'].fillna(median_loan, inplace=True)
    
    return df

def remove_outliers(df):
    """
    Removes outliers based on predefined thresholds.
    """
    # Filtering conditions based on domain knowledge/analysis
    # Note: Hardcoded values from original script
    mask = np.ones(len(df), dtype=bool)
    
    if 'ApplicantIncome' in df.columns:
        mask &= (df['ApplicantIncome'] <= 50000)
    if 'CoapplicantIncome' in df.columns:
        mask &= (df['CoapplicantIncome'] <= 20000)
    if 'LoanAmount' in df.columns:
        mask &= (df['LoanAmount'] <= 500)
    
    df_clean = df[mask].copy()
    
    # Drop Loan_ID as it is not needed for modeling
    if 'Loan_ID' in df_clean.columns:
        df_clean = df_clean.drop(columns='Loan_ID')
        
    return df_clean

def encode_features(df):
    """
    Encodes categorical features using Label Encoding and One-Hot Encoding.
    """
    df_encoded = df.copy()
    
    # Label Encoding for binary/ordinal variables
    le = LabelEncoder()
    cate_cols_le = ['Gender', 'Married', 'Education', 'Self_Employed', 'Loan_Status']
    for col in cate_cols_le:
        if col in df_encoded.columns:
            df_encoded[col] = le.fit_transform(df_encoded[col])
            
    # One-Hot Encoding for 'Property_Area'
    if 'Property_Area' in df_encoded.columns:
        df_encoded = pd.get_dummies(df_encoded, columns=['Property_Area'])
        
    return df_encoded

def visualize_data(df):
    """
    Generates exploratory data analysis plots.
    """
    # Categorical Plots
    category_col = ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']
    for column in category_col:
        if column in df.columns:
            plt.figure(figsize=(5, 8))
            sns.countplot(x=df[column])
            plt.xlabel(f'{column}')
            plt.ylabel('Count')
            plt.title(f'Count plot of {column}')
            plt.show()

    # Numeric Plots
    numeric_vals = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in numeric_vals:
        if col in df.columns:
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 2, 1)
            sns.histplot(df[col], kde=True)
            plt.subplot(1, 2, 2)
            sns.boxplot(x=df[col])
            plt.tight_layout()
            plt.show()

def evaluate_and_report(x_train, y_train, x_test, y_test, models_list, data_description):
    """
    Trains, evaluates, and reports metrics for a list of models.
    """
    print(f"\n--- Evaluation for {data_description} ---")
    for model_name, model_prototype in models_list:
        # Clone the model to ensure a fresh instance
        model = clone(model_prototype)
        model.fit(x_train, y_train)
        
        y_train_pred = model.predict(x_train)
        y_test_pred = model.predict(x_test)
        
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        print(f"\nModel: {model_name}")
        print(f"Train Accuracy: {train_acc:.4f}")
        print(f"Test Accuracy:  {test_acc:.4f}")
        
        # Uncomment to see detailed reports
        # print('Confusion Matrix:\n', confusion_matrix(y_test, y_test_pred))
        # print('Classification Report:\n', classification_report(y_test, y_test_pred))

def main():
    # 1. Load Data
    filepath = 'loan_data_set.csv'
    print(f"Loading data from {filepath}...")
    df = load_data(filepath)
    if df is None:
        return

    # 2. Clean Data
    print("Cleaning data...")
    df = clean_data(df)
    
    # Optional: Visualize Data
    # visualize_data(df)

    # 3. Remove Outliers
    print("Removing outliers...")
    df_no_outliers = remove_outliers(df)
    print(f"Rows after removing outliers: {len(df_no_outliers)} (Original: {len(df)})")

    # 4. Feature Encoding
    print("Encoding features...")
    df_encoded = encode_features(df_no_outliers)
    
    # 5. Split Data
    print("Splitting data...")
    X = df_encoded.drop(columns='Loan_Status').values
    y = df_encoded['Loan_Status'].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 6. Scaling
    print("Scaling data...")
    mm = MinMaxScaler()
    X_train_scaled = mm.fit_transform(X_train)
    X_test_scaled = mm.transform(X_test)
    
    # 7. PCA
    print("Applying PCA...")
    # Using 8 components as determined by original analysis
    pca = PCA(n_components=8)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    # 8. Modeling & Evaluation
    models = [
        ('Logistic Regression', LogisticRegression()),
        ('Decision Tree', DecisionTreeClassifier()),
        ('Random Forest', RandomForestClassifier())
    ]
    
    evaluate_and_report(X_train_pca, y_train, X_test_pca, y_test, models, "PCA Transformed Data")
    evaluate_and_report(X_train_scaled, y_train, X_test_scaled, y_test, models, "Scaled Data (No PCA)")
    evaluate_and_report(X_train, y_train, X_test, y_test, models, "Raw Data (No Scaling/PCA)")

if __name__ == "__main__":
    main()
