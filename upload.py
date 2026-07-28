import streamlit as st
import pandas as pd


SUPPORTED_EXTENSIONS = ["csv"]


def upload_dataset():
    """
    Displays the file uploader widget and returns
    the uploaded file object.

    Returns
    -------
    UploadedFile or None
    """

    uploaded_file = st.file_uploader(
        label="Upload Customer Dataset (.csv)",
        type=SUPPORTED_EXTENSIONS,
        help="Upload a CSV file containing customer information."
    )

    return uploaded_file


def load_dataset(uploaded_file):
    """
    Reads uploaded CSV into a DataFrame.

    Parameters
    ----------
    uploaded_file : UploadedFile

    Returns
    -------
    pd.DataFrame
    """

    try:
        df = pd.read_csv(uploaded_file)

        return df

    except UnicodeDecodeError:

        df = pd.read_csv(uploaded_file, encoding="latin1")
        return df

    except Exception as e:

        st.error(f"Unable to read file.\n\n{e}")
        return None


def dataset_summary(df):
    """
    Displays a quick dataset summary.
    """

    st.subheader("Dataset Summary")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isna().sum().sum()))

    st.write("### Column Information")

    info = pd.DataFrame({
        "Column": df.columns,
        "Datatype": df.dtypes.astype(str),
        "Missing": df.isna().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        info,
        use_container_width=True
    )


def preview_dataset(df, rows=10):
    """
    Shows the first few rows.
    """

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(rows),
        use_container_width=True
    )


def basic_statistics(df):
    """
    Displays descriptive statistics.
    """

    st.subheader("Descriptive Statistics")

    numeric = df.select_dtypes(include="number")

    if numeric.empty:
        st.info("No numeric columns found.")
        return

    st.dataframe(
        numeric.describe().T,
        use_container_width=True
    )


def show_missing_values(df):
    """
    Displays missing value information.
    """

    st.subheader("Missing Values")

    missing = pd.DataFrame({
        "Missing Count": df.isna().sum(),
        "Missing %": (
            df.isna().mean() * 100
        ).round(2)
    })

    st.dataframe(
        missing,
        use_container_width=True
    )


def duplicate_summary(df):
    """
    Displays duplicate statistics.
    """

    duplicates = df.duplicated().sum()

    st.subheader("Duplicate Records")

    st.metric(
        "Duplicate Rows",
        duplicates
    )


def display_dataset_information(df):
    """
    Master function used by app.py.
    """

    preview_dataset(df)

    dataset_summary(df)

    basic_statistics(df)

    show_missing_values(df)

    duplicate_summary(df)