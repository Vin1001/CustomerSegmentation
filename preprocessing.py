import pandas as pd
import numpy as np
import streamlit as st

from sklearn.preprocessing import (
    StandardScaler,
    LabelEncoder
)

# ---------------------------------------------------------
# REQUIRED COLUMNS
# ---------------------------------------------------------

OPTIONAL_COLUMNS = [
    "Customer ID",
    "CustomerID",
    "Age",
    "Gender",
    "Annual Income",
    "Income",
    "Spending Score",
    "Total Orders",
    "Purchase Frequency",
    "Total Purchase Value",
    "Last Purchase Date"
]

# ---------------------------------------------------------
# DATASET VALIDATION
# ---------------------------------------------------------


def validate_dataset(df):
    """
    Performs basic validation.
    """

    if df.empty:
        return False, "Dataset is empty."

    if len(df.columns) < 2:
        return False, "Dataset contains too few columns."

    return True, "Dataset uploaded successfully."


# ---------------------------------------------------------
# STANDARDIZE COLUMN NAMES
# ---------------------------------------------------------


def standardize_columns(df):

    df = df.copy()

    df.columns = (
        df.columns
        .str.strip()
        .str.replace("_", " ")
        .str.replace("-", " ")
    )

    return df


# ---------------------------------------------------------
# MISSING VALUE HANDLING
# ---------------------------------------------------------


def handle_missing_values(df):

    df = df.copy()

    numeric = df.select_dtypes(include=np.number).columns

    categorical = df.select_dtypes(exclude=np.number).columns

    for col in numeric:

        df[col].fillna(df[col].median(), inplace=True)

    for col in categorical:

        if df[col].mode().empty:
            df[col].fillna("Unknown", inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    return df


# ---------------------------------------------------------
# REMOVE DUPLICATES
# ---------------------------------------------------------


def remove_duplicates(df):

    before = len(df)

    df = df.drop_duplicates()

    after = len(df)

    st.info(f"Removed {before-after} duplicate rows.")

    return df


# ---------------------------------------------------------
# DATE CONVERSION
# ---------------------------------------------------------


def convert_dates(df):

    df = df.copy()

    for col in df.columns:

        if "date" in col.lower():

            try:

                df[col] = pd.to_datetime(df[col])

            except:

                pass

    return df


# ---------------------------------------------------------
# LABEL ENCODING
# ---------------------------------------------------------


def encode_categorical(df):

    df = df.copy()

    categorical = df.select_dtypes(include="object").columns

    encoders = {}

    for col in categorical:

        encoder = LabelEncoder()

        df[col] = encoder.fit_transform(
            df[col].astype(str)
        )

        encoders[col] = encoder

    return df, encoders


# ---------------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------------


def select_features(df):

    ignore = [
        "Customer ID",
        "CustomerID",
        "Name",
        "Customer Name"
    ]

    columns = []

    for col in df.columns:

        if col not in ignore:

            columns.append(col)

    numeric = df[columns].select_dtypes(include=np.number)

    return numeric


# ---------------------------------------------------------
# FEATURE SCALING
# ---------------------------------------------------------


def scale_features(df):

    scaler = StandardScaler()

    scaled = scaler.fit_transform(df)

    scaled_df = pd.DataFrame(
        scaled,
        columns=df.columns
    )

    return scaled_df, scaler


# ---------------------------------------------------------
# COMPLETE CLEANING PIPELINE
# ---------------------------------------------------------


def clean_dataset(df):

    df = standardize_columns(df)

    df = handle_missing_values(df)

    df = remove_duplicates(df)

    df = convert_dates(df)

    return df


# ---------------------------------------------------------
# COMPLETE PREPROCESSING PIPELINE
# ---------------------------------------------------------


def preprocess_features(df):
    """
    Returns

    numeric_df
    scaled_numpy_array
    """

    df = df.copy()

    df, encoders = encode_categorical(df)

    numeric = select_features(df)

    scaled_df, scaler = scale_features(numeric)

    return numeric, scaled_df, scaler


# ---------------------------------------------------------
# DATASET INFORMATION
# ---------------------------------------------------------


def dataset_information(df):

    info = pd.DataFrame({
        "Column": df.columns,
        "Datatype": df.dtypes.astype(str),
        "Missing": df.isna().sum().values,
        "Unique": df.nunique().values
    })

    return info


# ---------------------------------------------------------
# OUTLIER DETECTION (IQR)
# ---------------------------------------------------------


def detect_outliers(df):

    numeric = df.select_dtypes(include=np.number)

    outlier_summary = {}

    for col in numeric.columns:

        Q1 = numeric[col].quantile(0.25)
        Q3 = numeric[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = numeric[
            (numeric[col] < lower) |
            (numeric[col] > upper)
        ]

        outlier_summary[col] = len(outliers)

    return outlier_summary


# ---------------------------------------------------------
# REMOVE OUTLIERS (OPTIONAL)
# ---------------------------------------------------------


def remove_outliers(df):

    numeric = df.select_dtypes(include=np.number)

    clean_df = df.copy()

    for col in numeric.columns:

        Q1 = clean_df[col].quantile(.25)
        Q3 = clean_df[col].quantile(.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        clean_df = clean_df[
            (clean_df[col] >= lower) &
            (clean_df[col] <= upper)
        ]

    return clean_df


# ---------------------------------------------------------
# DISPLAY PREPROCESSING SUMMARY
# ---------------------------------------------------------


def preprocessing_summary(df):

    st.subheader("Preprocessing Summary")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        df.shape[0]
    )

    c2.metric(
        "Columns",
        df.shape[1]
    )

    c3.metric(
        "Missing Values",
        int(df.isna().sum().sum())
    )

    st.dataframe(
        dataset_information(df),
        use_container_width=True
    )