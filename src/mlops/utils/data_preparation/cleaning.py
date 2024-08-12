import pandas as pd
import numpy as np


def clean(df: pd.DataFrame)-> pd.DataFrame:
    
    df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.NaN)

    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].median())
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].median())
    df['BMI'] = df['BMI'].fillna(df['BMI'].median())

    return df