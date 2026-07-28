import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score
)

# --------------------------------------------------------
# ELBOW METHOD
# --------------------------------------------------------

def elbow_method(scaled_data, max_clusters=10):
    """
    Displays Elbow Method plot.
    """

    inertias = []

    K = range(2, max_clusters + 1)

    for k in K:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        model.fit(scaled_data)

        inertias.append(model.inertia_)

    fig = px.line(
        x=list(K),
        y=inertias,
        markers=True,
        labels={
            "x": "Number of Clusters (K)",
            "y": "Inertia"
        },
        title="Elbow Method"
    )

    st.plotly_chart(fig, use_container_width=True)

    return inertias


# --------------------------------------------------------
# SILHOUETTE ANALYSIS
# --------------------------------------------------------

def silhouette_analysis(scaled_data, max_clusters=10):

    scores = []

    K = range(2, max_clusters + 1)

    for k in K:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(scaled_data)

        score = silhouette_score(
            scaled_data,
            labels
        )

        scores.append(score)

    fig = px.line(
        x=list(K),
        y=scores,
        markers=True,
        labels={
            "x": "Number of Clusters",
            "y": "Silhouette Score"
        },
        title="Silhouette Analysis"
    )

    st.plotly_chart(fig, use_container_width=True)

    return scores


# --------------------------------------------------------
# DAVIES BOULDIN
# --------------------------------------------------------

def davies_bouldin_analysis(scaled_data, max_clusters=10):

    scores = []

    K = range(2, max_clusters + 1)

    for k in K:

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(scaled_data)

        score = davies_bouldin_score(
            scaled_data,
            labels
        )

        scores.append(score)

    fig = px.line(
        x=list(K),
        y=scores,
        markers=True,
        labels={
            "x": "Clusters",
            "y": "Davies-Bouldin Index"
        },
        title="Davies-Bouldin Index"
    )

    st.plotly_chart(fig, use_container_width=True)

    return scores


# --------------------------------------------------------
# AUTOMATIC K
# --------------------------------------------------------

def get_optimal_clusters(scaled_data):

    scores = []

    for k in range(2,11):

        model = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=10
        )

        labels = model.fit_predict(scaled_data)

        score = silhouette_score(
            scaled_data,
            labels
        )

        scores.append(score)

    best = np.argmax(scores) + 2

    return best


# --------------------------------------------------------
# MAIN KMEANS
# --------------------------------------------------------

def perform_clustering(
        original_df,
        scaled_data,
        n_clusters
):

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=20
    )

    labels = model.fit_predict(scaled_data)

    clustered = original_df.copy()

    clustered["Cluster"] = labels

    return clustered, model


# --------------------------------------------------------
# PCA 2D
# --------------------------------------------------------

def pca_2d(clustered_df, scaled_data):

    pca = PCA(
        n_components=2,
        random_state=42
    )

    components = pca.fit_transform(
        scaled_data
    )

    df = pd.DataFrame()

    df["PC1"] = components[:,0]
    df["PC2"] = components[:,1]
    df["Cluster"] = clustered_df["Cluster"]

    return df


# --------------------------------------------------------
# PCA 3D
# --------------------------------------------------------

def pca_3d(clustered_df, scaled_data):

    pca = PCA(
        n_components=3,
        random_state=42
    )

    comp = pca.fit_transform(
        scaled_data
    )

    df = pd.DataFrame()

    df["PC1"] = comp[:,0]
    df["PC2"] = comp[:,1]
    df["PC3"] = comp[:,2]
    df["Cluster"] = clustered_df["Cluster"]

    return df


# --------------------------------------------------------
# CLUSTER QUALITY
# --------------------------------------------------------

def clustering_metrics(
        scaled_data,
        labels
):

    sil = silhouette_score(
        scaled_data,
        labels
    )

    db = davies_bouldin_score(
        scaled_data,
        labels
    )

    c1,c2 = st.columns(2)

    c1.metric(
        "Silhouette Score",
        round(sil,3)
    )

    c2.metric(
        "Davies-Bouldin Index",
        round(db,3)
    )


# --------------------------------------------------------
# CLUSTER SUMMARY
# --------------------------------------------------------

def cluster_summary(df):

    summary = df.groupby("Cluster").agg(

        Customers=("Cluster","count")

    )

    return summary


# --------------------------------------------------------
# PROFILE NUMERIC FEATURES
# --------------------------------------------------------

def cluster_profile(df):

    numeric = df.select_dtypes(
        include=np.number
    )

    profile = numeric.groupby(
        df["Cluster"]
    ).mean()

    return profile.round(2)


# --------------------------------------------------------
# SEGMENT LABELS
# --------------------------------------------------------

def assign_segment_labels(df):

    if "Annual Income" not in df.columns:
        return df

    if "Spending Score" not in df.columns:
        return df

    profile = df.groupby("Cluster")[[
        "Annual Income",
        "Spending Score"
    ]].mean()

    labels = {}

    income_mean = profile["Annual Income"].median()
    spend_mean = profile["Spending Score"].median()

    for cluster,row in profile.iterrows():

        income = row["Annual Income"]
        spend = row["Spending Score"]

        if income >= income_mean and spend >= spend_mean:

            labels[cluster] = "High Value"

        elif income >= income_mean:

            labels[cluster] = "Potential Customers"

        elif spend >= spend_mean:

            labels[cluster] = "Impulse Buyers"

        else:

            labels[cluster] = "Budget Customers"

    df["Segment"] = df["Cluster"].map(labels)

    return df


# --------------------------------------------------------
# CLUSTER CENTERS
# --------------------------------------------------------

def cluster_centers(model, columns):

    centers = pd.DataFrame(
        model.cluster_centers_,
        columns=columns
    )

    centers.index.name = "Cluster"

    return centers.reset_index()