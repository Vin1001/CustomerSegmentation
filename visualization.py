import pandas as pd
import numpy as np

import uuid
import streamlit as st

def show_plot(fig):
        st.plotly_chart(

        fig,

        use_container_width=True,
        key = (uuid.uuid4())

    )

import plotly.express as px
import plotly.graph_objects as go

from clustering import (
    pca_2d,
    pca_3d,
    cluster_profile
)

# ==========================================================
# CUSTOMER OVERVIEW
# ==========================================================

def customer_overview(df):

    st.header("Customer Overview")

    numeric = df.select_dtypes(include=np.number)

    total_customers = len(df)

    avg_income = np.mean(numeric["annual_income"])
    avg_spending = np.mean(numeric["avg_monthly_spend"])
    avg_purchase = np.mean(numeric["avg_order_value"])

    income_cols = [
        "Annual Income",
        "Income"
    ]

    spending_cols = [
        "Spending Score"
    ]

    purchase_cols = [
        "Total Purchase Value",
        "Purchase Value"
    ]

    for col in income_cols:

        if col in df.columns:
            avg_income = df[col].mean()
            break

    for col in spending_cols:

        if col in df.columns:
            avg_spending = df[col].mean()
            break

    for col in purchase_cols:

        if col in df.columns:
            avg_purchase = df[col].mean()
            break

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Customers",
        f"{total_customers:,}"
    )

    c2.metric(
        "Average Income",
        f"{avg_income:.2f}"
    )

    c3.metric(
        "Average Spending",
        f"{avg_spending:.2f}"
    )

    c4.metric(
        "Average Purchase",
        f"{avg_purchase:.2f}"
    )

    st.markdown("---")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

# ==========================================================
# CORRELATION HEATMAP
# ==========================================================

def correlation_heatmap(df):

    st.subheader("Correlation Heatmap")

    numeric = df.select_dtypes(include=np.number)

    if numeric.shape[1] < 2:

        st.warning("Not enough numeric columns.")

        return

    corr = numeric.corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        origin="lower"
    )

    fig.update_layout(
        height=650
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# DISTRIBUTION PLOTS
# ==========================================================

def distribution_plots(df):

    st.subheader("Feature Distributions")

    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:

        st.info("No numeric columns found.")

        return

    feature = st.selectbox(
        "Select Feature",
        numeric.columns
    )

    fig = px.histogram(
        df,
        x=feature,
        nbins=30,
        marginal="box",
        title=f"{feature} Distribution"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# BOXPLOTS
# ==========================================================

def boxplots(df):

    st.subheader("Box Plot")

    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:

        return

    feature = st.selectbox(
        "Select Numeric Feature",
        numeric.columns,
        key="boxplot"
    )

    fig = px.box(
        df,
        y=feature,
        points="outliers",
        title=f"{feature} Box Plot"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# ==========================================================
# MISSING VALUE VISUALIZATION
# ==========================================================

def missing_value_chart(df):

    st.subheader("Missing Values")

    missing = df.isna().sum()

    missing = missing[missing > 0]

    if len(missing) == 0:

        st.success("No missing values detected.")

        return

    chart = pd.DataFrame({

        "Column": missing.index,

        "Missing": missing.values

    })

    fig = px.bar(

        chart,

        x="Column",

        y="Missing",

        color="Missing",

        title="Missing Values"

    )

    show_plot(fig)

# ==========================================================
# DATA QUALITY SUMMARY
# ==========================================================

def data_quality_dashboard(df):

    st.subheader("Data Quality")

    rows = len(df)

    cols = len(df.columns)

    duplicates = df.duplicated().sum()

    missing = df.isna().sum().sum()

    numeric = len(

        df.select_dtypes(include=np.number).columns

    )

    categorical = len(

        df.select_dtypes(exclude=np.number).columns

    )

    c1, c2, c3 = st.columns(3)

    c4, c5, c6 = st.columns(3)

    c1.metric("Rows", rows)

    c2.metric("Columns", cols)

    c3.metric("Duplicates", duplicates)

    c4.metric("Missing", missing)

    c5.metric("Numeric", numeric)

    c6.metric("Categorical", categorical)

# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

def summary_statistics(df):

    st.subheader("Summary Statistics")

    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:

        st.info("No numeric features.")

        return

    summary = numeric.describe().T

    st.dataframe(

        summary,

        use_container_width=True

    )

# ==========================================================
# CATEGORICAL DISTRIBUTION
# ==========================================================

def categorical_distribution(df):

    categorical = df.select_dtypes(

        exclude=np.number

    )

    if categorical.empty:

        return

    st.subheader("Categorical Distribution")

    column = st.selectbox(

        "Categorical Feature",

        categorical.columns,

        key="cat_dist"

    )

    counts = (

        df[column]

        .value_counts()

        .reset_index()

    )

    counts.columns = [

        column,

        "Count"

    ]

    fig = px.bar(

        counts,

        x=column,

        y="Count",

        color="Count",

        title=column

    )

    show_plot(fig)

# ==========================================================
# NUMERIC FEATURE TABLE
# ==========================================================

def numeric_summary(df):

    st.subheader("Numeric Features")

    numeric = df.select_dtypes(

        include=np.number

    )

    info = pd.DataFrame({

        "Feature": numeric.columns,

        "Missing": numeric.isna().sum().values,

        "Mean": numeric.mean().values,

        "Std": numeric.std().values,

        "Minimum": numeric.min().values,

        "Maximum": numeric.max().values

    })

    st.dataframe(

        info,

        use_container_width=True

    )

# ==========================================================
# COMPLETE EDA DASHBOARD
# ==========================================================

def eda_dashboard(df):

    customer_overview(df)

    data_quality_dashboard(df)

    summary_statistics(df)

    correlation_heatmap(df)

    distribution_plots(df)

    boxplots(df)

    categorical_distribution(df)

    numeric_summary(df)

    missing_value_chart(df)


# ==========================================================
# PCA 2D CLUSTER VISUALIZATION
# ==========================================================

def cluster_visualization(df, scaled_data=None):

    st.subheader("Customer Clusters (PCA 2D)")

    if scaled_data is None:
        st.warning("Scaled data required for PCA visualization.")
        return

    pca_df = pca_2d(df, scaled_data)

    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="Cluster",
        hover_data=["Cluster"],
        title="Customer Segments using PCA"
    )

    fig.update_traces(marker=dict(size=10))

    st.plotly_chart(
        fig,
        use_container_width=True,
        key = (uuid.uuid4())
    )


# ==========================================================
# PCA 3D
# ==========================================================

def cluster_visualization_3d(df, scaled_data):

    st.subheader("3D Customer Segmentation")

    pca_df = pca_3d(df, scaled_data)

    fig = px.scatter_3d(
        pca_df,
        x="PC1",
        y="PC2",
        z="PC3",
        color="Cluster",
        opacity=0.8
    )

    fig.update_traces(marker=dict(size=5))

    st.plotly_chart(
        fig,
        use_container_width=True,
        key = (uuid.uuid4())
    )


# ==========================================================
# INCOME VS SPENDING
# ==========================================================

def income_spending_plot(df):

    income_col = None
    spending_col = None

    income_candidates = [
        "Annual Income",
        "Income"
    ]

    spending_candidates = [
        "Spending Score"
    ]

    for col in income_candidates:

        if col in df.columns:
            income_col = col
            break

    for col in spending_candidates:

        if col in df.columns:
            spending_col = col
            break

    if income_col is None or spending_col is None:

        st.info(
            "Income and Spending columns not found."
        )

        return

    fig = px.scatter(

        df,

        x=income_col,

        y=spending_col,

        color="Cluster",

        hover_data=df.columns,

        title="Income vs Spending Score"

    )

    fig.update_layout(height=650)

    st.plotly_chart(

        fig,

        use_container_width=True,
        key = (uuid.uuid4())

    )


# ==========================================================
# CLUSTER SIZE
# ==========================================================

def cluster_size_chart(df):

    st.subheader("Customers per Cluster")

    counts = (

        df["Cluster"].value_counts().sort_index().reset_index()

    )

    counts.columns = [

        "Cluster",

        "Customers"

    ]

    fig = px.bar(

        counts,

        x="Cluster",

        y="Customers",

        color="Customers",

        text="Customers"

    )

    show_plot(fig)


# ==========================================================
# CLUSTER PROFILE TABLE
# ==========================================================

def cluster_profiles(df):

    st.subheader("Cluster Profiles")

    profile = cluster_profile(df)

    st.dataframe(

        profile,

        use_container_width=True

    )


# ==========================================================
# PROFILE HEATMAP
# ==========================================================

def cluster_profile_heatmap(df):

    st.subheader("Cluster Mean Heatmap")

    profile = cluster_profile(df)

    fig = px.imshow(

        profile,

        text_auto=".2f",

        color_continuous_scale="Viridis",

        aspect="auto"

    )

    st.plotly_chart(

        fig,

        use_container_width=True,
        key = (uuid.uuid4())

    )


# ==========================================================
# SEGMENT DISTRIBUTION
# ==========================================================

def segment_distribution(df):

    if "Segment" not in df.columns:

        return

    st.subheader("Business Segments")

    counts = (

        df["Segment"]

        .value_counts()

        .reset_index()

    )

    counts.columns = [

        "Segment",

        "Customers"

    ]

    fig = px.pie(

        counts,

        values="Customers",

        names="Segment",

        hole=.45

    )

    st.plotly_chart(

        fig,

        use_container_width=True,
        key = (uuid.uuid4())

    )


# ==========================================================
# RADAR CHART
# ==========================================================

def radar_chart(df):

    st.subheader("Cluster Comparison")

    profile = cluster_profile(df)

    categories = profile.columns.tolist()

    fig = go.Figure()

    for cluster in profile.index:

        fig.add_trace(

            go.Scatterpolar(

                r=profile.loc[cluster].values,

                theta=categories,

                fill="toself",

                name=f"Cluster {cluster}"

            )

        )

    fig.update_layout(

        polar=dict(

            radialaxis=dict(

                visible=True

            )

        ),

        showlegend=True

    )

    show_plot(fig)


# ==========================================================
# FEATURE COMPARISON
# ==========================================================

def compare_feature(df):

    numeric = df.select_dtypes(include="number")

    if numeric.empty:

        return

    feature = st.selectbox(

        "Compare Feature",

        numeric.columns,

        key="compare"

    )

    fig = px.box(

        df,

        x="Cluster",

        y=feature,

        color="Cluster",

        points="all"

    )

    st.plotly_chart(

        fig,

        use_container_width=True,
        key = (uuid.uuid4())

    )


# ==========================================================
# CLUSTER CENTROID HEATMAP
# ==========================================================

def centroid_heatmap(model, feature_names):

    st.subheader("Cluster Centers")

    centers = pd.DataFrame(

        model.cluster_centers_,

        columns=feature_names

    )

    fig = px.imshow(

        centers,

        text_auto=".2f",

        color_continuous_scale="RdYlGn",

        aspect="auto"

    )

    st.plotly_chart(

        fig,

        use_container_width=True,
        key = (uuid.uuid4())

    )


# ==========================================================
# COMPLETE CLUSTER DASHBOARD
# ==========================================================

def clustering_dashboard(df, scaled_data, model=None):

    cluster_size_chart(df)

    cluster_visualization(df, scaled_data)

    cluster_visualization_3d(df, scaled_data)

    income_spending_plot(df)

    cluster_profiles(df)

    cluster_profile_heatmap(df)

    radar_chart(df)

    compare_feature(df)

    segment_distribution(df)

    if model is not None:

        feature_names = df.select_dtypes(
            include="number"
        ).drop(
            columns=["Cluster"],
            errors="ignore"
        ).columns

        # centroid_heatmap(
        #     model,
        #     feature_names
        # )

import io

# ==========================================================
# FILTER DATASET
# ==========================================================

def filter_dashboard(df):

    st.subheader("Interactive Dataset Filter")

    filtered = df.copy()

    categorical = filtered.select_dtypes(
        exclude="number"
    ).columns

    for col in categorical:

        values = st.multiselect(
            f"{col}",
            sorted(filtered[col].dropna().unique()),
            default=sorted(filtered[col].dropna().unique())
        )

        if values:

            filtered = filtered[
                filtered[col].isin(values)
            ]

    st.write(f"Filtered Rows : {len(filtered)}")

    st.dataframe(
        filtered,
        use_container_width=True
    )

    return filtered


# ==========================================================
# EXPORT FIGURE
# ==========================================================

def download_dataframe(df):

    csv = df.to_csv(index=False).encode()

    st.download_button(

        "Download Filtered Dataset",

        csv,

        file_name="filtered_dataset.csv",

        mime="text/csv"

    )


# ==========================================================
# BUSINESS INSIGHT CARDS
# ==========================================================

def business_dashboard(df):

    st.header("Business Insights")

    if "Segment" not in df.columns:

        st.warning(
            "Segment labels not available."
        )

        return

    counts = df["Segment"].value_counts()

    cols = st.columns(len(counts))

    for i, segment in enumerate(counts.index):

        cols[i].metric(

            segment,

            counts[segment]

        )

    st.markdown("---")

    recommendations(df)


# ==========================================================
# BUSINESS RECOMMENDATIONS
# ==========================================================

def recommendations(df):

    st.subheader("Marketing Recommendations")

    if "Segment" not in df.columns:

        return

    segments = df["Segment"].unique()

    recommendations_map = {

        "High Value":
        "🎯 Offer premium membership, loyalty rewards and exclusive discounts.",

        "Potential Customers":
        "📧 Send personalized offers to encourage more purchases.",

        "Impulse Buyers":
        "🛍 Promote limited-time offers and bundle discounts.",

        "Budget Customers":
        "💸 Target with coupons, cashback and price-sensitive campaigns."

    }

    for segment in segments:

        st.success(
            f"**{segment}**\n\n"
            f"{recommendations_map.get(segment,'No recommendation available.')}"
        )


# ==========================================================
# CLUSTER INSIGHT TABLE
# ==========================================================

def cluster_statistics(df):

    st.subheader("Cluster Statistics")

    summary = (

        df.groupby("Cluster")

        .agg(

            Customers=("Cluster","count")

        )

    )

    st.dataframe(

        summary,

        use_container_width=True

    )


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

def feature_variability(df):

    st.subheader("Feature Variability")

    numeric = df.select_dtypes(include="number")

    if numeric.empty:

        return

    std = numeric.std().sort_values(
        ascending=False
    )

    chart = std.reset_index()

    chart.columns = [

        "Feature",

        "Standard Deviation"

    ]

    fig = px.bar(

        chart,

        x="Feature",

        y="Standard Deviation",

        color="Standard Deviation"

    )

    show_plot(fig)


# ==========================================================
# SEGMENT COMPARISON TABLE
# ==========================================================

def segment_summary(df):

    if "Segment" not in df.columns:

        return

    st.subheader("Segment Summary")

    table = (

        df.groupby("Segment")

        .mean(numeric_only=True)

        .round(2)

    )

    st.dataframe(

        table,

        use_container_width=True

    )


# ==========================================================
# KPI DASHBOARD
# ==========================================================

def dashboard_kpis(df):

    st.header("Executive Dashboard")

    total = len(df)

    clusters = df["Cluster"].nunique()

    avg_missing = int(df.isna().sum().sum())

    duplicates = int(df.duplicated().sum())

    c1,c2,c3,c4 = st.columns(4)

    c1.metric(
        "Customers",
        total
    )

    c2.metric(
        "Clusters",
        clusters
    )

    c3.metric(
        "Missing Values",
        avg_missing
    )

    c4.metric(
        "Duplicate Rows",
        duplicates
    )


# ==========================================================
# COMPLETE BUSINESS DASHBOARD
# ==========================================================

def executive_dashboard(df):

    dashboard_kpis(df)

    business_dashboard(df)

    cluster_statistics(df)

    segment_summary(df)

    feature_variability(df)


# ==========================================================
# ABOUT PROJECT
# ==========================================================

def about_page():

    st.header("About")

    st.markdown("""

### E-Commerce Customer Segmentation Dashboard

Developed using

- Streamlit
- Pandas
- Plotly
- Scikit-Learn
- KMeans Clustering
- PCA

Features

- Dataset Upload
- Data Cleaning
- Exploratory Data Analysis
- Customer Segmentation
- Business Insights
- Interactive Dashboard

""")


# ==========================================================
# FOOTER
# ==========================================================

def footer():

    st.markdown("---")

    st.caption(
        "E-Commerce Customer Segmentation Dashboard"
    )