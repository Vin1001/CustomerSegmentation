import streamlit as st
import pandas as pd
import plotly.express as px

from upload import (
    upload_dataset,
    load_dataset,
    display_dataset_information
)

from preprocessing import (
    validate_dataset,
    clean_dataset,
    preprocess_features,
)

from clustering import (
    perform_clustering,
    elbow_method,
    silhouette_analysis,
)

from visualization import (
    customer_overview,
    correlation_heatmap,
    distribution_plots,
    cluster_visualization,
    clustering_dashboard,
    income_spending_plot,
    cluster_profiles,
)

from reports import (
    generate_business_insights,
    report_downloads
)

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="E-Commerce Customer Segmentation",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 E-Commerce Customer Segmentation Dashboard")
st.markdown("---")

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Module",
    [
        "Dataset Upload",
        "Customer Overview",
        "Data Cleaning",
        "EDA",
        "K-Means Clustering",
        "Visualization",
        "Business Insights",
        "Download Report"
    ]
)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "df" not in st.session_state:
    st.session_state.df = None

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

if "scaled_data" not in st.session_state:
    st.session_state.scaled_data = None

if "clustered_df" not in st.session_state:
    st.session_state.clustered_df = None

if "model" not in st.session_state:
    st.session_state.model = None

# ===================================================
# DATASET UPLOAD
# ===================================================

if page == "Dataset Upload":

    st.header("Upload Customer Dataset")

    uploaded = upload_dataset()

    if uploaded is not None:

        df = load_dataset(uploaded)

        valid, message = validate_dataset(df)

        if valid:

            st.success(message)

            st.session_state.df = df

            display_dataset_information(df)

        else:

            st.error(message)

# ===================================================
# CUSTOMER OVERVIEW
# ===================================================

elif page == "Customer Overview":

    if st.session_state.df is None:

        st.warning("Please upload a dataset first.")

    else:

        customer_overview(st.session_state.df)

# ===================================================
# DATA CLEANING
# ===================================================

elif page == "Data Cleaning":

    if st.session_state.df is None:

        st.warning("Upload dataset first.")

    else:

        st.header("Data Cleaning")

        cleaned = clean_dataset(st.session_state.df)

        st.session_state.processed_df = cleaned

        st.success("Dataset cleaned successfully.")

        st.dataframe(cleaned.head())

        st.write(cleaned.describe())

# ===================================================
# EDA
# ===================================================

elif page == "EDA":

    if st.session_state.processed_df is None:

        st.warning("Please clean the dataset first.")

    else:

        df = st.session_state.processed_df

        st.header("Exploratory Data Analysis")

        correlation_heatmap(df)

        distribution_plots(df)

# ===================================================
# CLUSTERING
# ===================================================

elif page == "K-Means Clustering":

    if st.session_state.processed_df is None:

        st.warning("Please preprocess data first.")

    else:

        df = st.session_state.processed_df

        numeric_df, scaled, scaler = preprocess_features(df)

        st.session_state.feature_names = numeric_df.columns.tolist()

        st.session_state.scaled_data = scaled

        st.subheader("Optimal Number of Clusters")

        elbow_method(scaled)

        silhouette_analysis(scaled)

        k = st.slider(
            "Select Number of Clusters",
            2,
            10,
            4
        )

        clustered_df, model = perform_clustering(
            df,
            scaled,
            k
        )

        st.session_state.clustered_df = clustered_df
        st.session_state.model = model

        st.success(f"K-Means completed with {k} clusters.")

        st.dataframe(clustered_df.head())

# ===================================================
# VISUALIZATION
# ===================================================

elif page == "Visualization":

    if st.session_state.clustered_df is None:

        st.warning("Run clustering first.")

    else:

        df = st.session_state.clustered_df

        st.header("Cluster Visualization")

        cluster_visualization(
    st.session_state.clustered_df,
    st.session_state.scaled_data
        )

        clustering_dashboard(
    st.session_state.clustered_df,
    # st.session_state.processed_df,
    st.session_state.scaled_data,
    st.session_state.model
)

        income_spending_plot(df)

        cluster_profiles(df)

# ===================================================
# BUSINESS INSIGHTS
# ===================================================

elif page == "Business Insights":

    if st.session_state.clustered_df is None:

        st.warning("Run clustering first.")

    else:

        st.header("Business Insights")

        insights = generate_business_insights(
            st.session_state.clustered_df
        )

        for item in insights:

            st.success(item)

# ===================================================
# REPORT
# ===================================================

elif page == "Download Report":

    if st.session_state.clustered_df is None:

        st.warning("Nothing to export.")

    else:

        # report = generate_report(
        #     st.session_state.clustered_df
        # )

        # csv = st.session_state.clustered_df.to_csv(index=False)

        # st.download_button(
        #     "⬇ Download Clustered CSV",
        #     csv,
        #     file_name="clustered_customers.csv",
        #     mime="text/csv"
        # )

        # st.download_button(
        #     "⬇ Download Business Report",
        #     report,
        #     file_name="business_report.txt",
        #     mime="text/plain"
        # )
        report_downloads(
    st.session_state.clustered_df,
    st
)