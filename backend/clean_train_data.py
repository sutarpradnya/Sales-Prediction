
import pandas as pd
import os

def clean_train_data(input_path, output_path):
    print(f"Loading data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # 1. Basic Inspection
    print("Initial Data Info:")
    print(df.info())
    
    # 2. Handling Missing Values
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())
    
    # Fill missing values
    # For 'Postal Code', it might be float due to NaNs. Let's fill with mode or 0, then maybe converting to int/str
    if 'Postal Code' in df.columns:
        df['Postal Code'] = df['Postal Code'].fillna(0).astype(int)

    # For other object columns, fill with mode
    for col in df.columns:
        if df[col].dtype == 'object':
             if not df[col].mode().empty:
                df[col] = df[col].fillna(df[col].mode()[0])
        elif df[col].dtype in ['float64', 'int64']:
             df[col] = df[col].fillna(df[col].mean())

    print("\nMissing values after cleaning:")
    print(df.isnull().sum())

    # 3. Remove Duplicates
    print(f"\nDuplicates found: {df.duplicated().sum()}")
    df = df.drop_duplicates()

    # 4. Standardizing Date Format
    # The view_file output showed dates like '08/11/2017' (dd/mm/yyyy)
    date_cols = ['Order Date', 'Ship Date']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], format='%d/%m/%Y', errors='coerce')

    # Drop rows where critical dates couldn't be parsed
    df = df.dropna(subset=date_cols)

    # 5. Drop irrelevant columns
    if 'Row ID' in df.columns:
        df = df.drop(columns=['Row ID'])

    # 6. Handling Outliers in Sales
    # Using IQR method to cap outliers
    if 'Sales' in df.columns:
        Q1 = df['Sales'].quantile(0.25)
        Q3 = df['Sales'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # print(f"Sales Outlier bounds: {lower_bound} to {upper_bound}")
        # Cap values
        df['Sales'] = df['Sales'].clip(lower=lower_bound, upper=upper_bound)

    # 7. Save Cleaned Data
    print(f"\nSaving cleaned data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Data cleaning complete!")

if __name__ == "__main__":
    # Define paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(base_dir, 'Dataset', 'train.csv')
    output_file = os.path.join(base_dir, 'Dataset', 'cleaned_train.csv')
    
    clean_train_data(input_file, output_file)
