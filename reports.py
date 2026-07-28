import io
from datetime import datetime

import pandas as pd

try:
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import (
        SimpleDocTemplate,
        Paragraph,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.lib import colors

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


# ==========================================================
# BUSINESS INSIGHTS
# ==========================================================

def generate_business_insights(df):
    """
    Returns a list of business insights.
    """

    insights = []

    insights.append(
        f"Total customers analysed : {len(df)}"
    )

    if "Cluster" in df.columns:

        insights.append(
            f"Number of customer clusters : {df['Cluster'].nunique()}"
        )

    if "Segment" in df.columns:

        segment_counts = df["Segment"].value_counts()

        top = segment_counts.idxmax()

        insights.append(
            f"Largest customer segment : {top}"
        )

        for segment, count in segment_counts.items():

            insights.append(
                f"{segment}: {count} customers"
            )

    if "Annual Income" in df.columns:

        insights.append(
            f"Average annual income : {df['Annual Income'].mean():.2f}"
        )

    if "Spending Score" in df.columns:

        insights.append(
            f"Average spending score : {df['Spending Score'].mean():.2f}"
        )

    return insights


# ==========================================================
# SEGMENT RECOMMENDATIONS
# ==========================================================

def segment_recommendations():

    return {

        "High Value":
        "Reward with VIP membership, loyalty programs and premium offers.",

        "Potential Customers":
        "Increase engagement through personalized marketing campaigns.",

        "Impulse Buyers":
        "Promote flash sales, bundles and limited-time offers.",

        "Budget Customers":
        "Provide coupons, discounts and cashback offers."

    }


# ==========================================================
# CLUSTER SUMMARY
# ==========================================================

def cluster_summary(df):

    if "Cluster" not in df.columns:

        return pd.DataFrame()

    summary = (

        df.groupby("Cluster")

        .agg(

            Customers=("Cluster", "count")

        )

    )

    return summary


# ==========================================================
# SEGMENT SUMMARY
# ==========================================================

def segment_summary(df):

    if "Segment" not in df.columns:

        return pd.DataFrame()

    return (

        df.groupby("Segment")

        .mean(numeric_only=True)

        .round(2)

    )


# ==========================================================
# TEXT REPORT
# ==========================================================

def generate_report(df):

    report = []

    report.append(
        "=" * 60
    )

    report.append(
        "E-COMMERCE CUSTOMER SEGMENTATION REPORT"
    )

    report.append(
        "=" * 60
    )

    report.append(
        f"Generated : {datetime.now()}"
    )

    report.append("")

    report.append("EXECUTIVE SUMMARY")

    report.append("------------------------------")

    report.extend(
        generate_business_insights(df)
    )

    report.append("")

    report.append("CLUSTER SUMMARY")

    report.append("------------------------------")

    cluster = cluster_summary(df)

    if not cluster.empty:

        report.append(cluster.to_string())

    report.append("")

    if "Segment" in df.columns:

        report.append("SEGMENT SUMMARY")

        report.append("------------------------------")

        report.append(
            segment_summary(df).to_string()
        )

        report.append("")

        report.append(
            "BUSINESS RECOMMENDATIONS"
        )

        report.append("------------------------------")

        recommendations = segment_recommendations()

        for segment in df["Segment"].unique():

            report.append(
                f"{segment}"
            )

            report.append(
                recommendations.get(
                    segment,
                    "No recommendation available."
                )
            )

            report.append("")

    return "\n".join(report)


# ==========================================================
# EXPORT CSV
# ==========================================================

def export_clustered_csv(df):

    return df.to_csv(index=False).encode("utf-8")


# ==========================================================
# PDF REPORT
# ==========================================================

def generate_pdf_report(df):

    if not REPORTLAB_AVAILABLE:

        return None

    buffer = io.BytesIO()

    doc = SimpleDocTemplate(buffer)

    styles = getSampleStyleSheet()

    story = []

    story.append(
        Paragraph(
            "E-Commerce Customer Segmentation Report",
            styles["Title"]
        )
    )

    story.append(
        Spacer(1, 20)
    )

    story.append(

        Paragraph(

            f"Generated : {datetime.now()}",

            styles["Normal"]

        )

    )

    story.append(
        Spacer(1, 20)
    )

    story.append(
        Paragraph(
            "Executive Summary",
            styles["Heading2"]
        )
    )

    for line in generate_business_insights(df):

        story.append(
            Paragraph(
                line,
                styles["BodyText"]
            )
        )

    story.append(
        Spacer(1, 20)
    )

    if "Cluster" in df.columns:

        cluster = cluster_summary(df)

        data = [

            ["Cluster", "Customers"]

        ]

        for idx, row in cluster.iterrows():

            data.append(

                [

                    str(idx),

                    str(row["Customers"])

                ]

            )

        table = Table(data)

        table.setStyle(

            TableStyle([

                ("BACKGROUND", (0,0), (-1,0), colors.grey),

                ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),

                ("GRID", (0,0), (-1,-1), 1, colors.black),

                ("BACKGROUND", (0,1), (-1,-1), colors.beige)

            ])

        )

        story.append(table)

    story.append(
        Spacer(1, 20)
    )

    if "Segment" in df.columns:

        story.append(

            Paragraph(

                "Business Recommendations",

                styles["Heading2"]

            )

        )

        rec = segment_recommendations()

        for segment in df["Segment"].unique():

            story.append(

                Paragraph(

                    f"<b>{segment}</b>",

                    styles["BodyText"]

                )

            )

            story.append(

                Paragraph(

                    rec.get(segment, ""),

                    styles["BodyText"]

                )

            )

            story.append(
                Spacer(1, 10)
            )

    doc.build(story)

    pdf = buffer.getvalue()

    buffer.close()

    return pdf


# ==========================================================
# DOWNLOAD BUTTONS
# ==========================================================

def report_downloads(df, st):

    st.download_button(

        "📄 Download TXT Report",

        generate_report(df),

        file_name="business_report.txt",

        mime="text/plain"

    )

    st.download_button(

        "📊 Download Clustered CSV",

        export_clustered_csv(df),

        file_name="clustered_customers.csv",

        mime="text/csv"

    )

    if REPORTLAB_AVAILABLE:

        pdf = generate_pdf_report(df)

        st.download_button(

            "📕 Download PDF Report",

            pdf,

            file_name="business_report.pdf",

            mime="application/pdf"

        )