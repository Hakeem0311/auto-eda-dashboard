import streamlit as st
import pandas as pd
import plotly.express as px
import sweetviz as sv
import os

# Page Configuration
st.set_page_config(
    page_title="Auto EDA Dashboard",
    layout="wide"
)

# Sidebar
st.sidebar.title("Auto EDA Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Visualizations",
        "Insights",
        "Reports"
    ]
)

# Title
st.title("Auto EDA Dashboard")

# File Upload
uploaded_file = st.file_uploader(
    "Upload CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    total_cells = df.shape[0] * df.shape[1]
    missing_cells = df.isnull().sum().sum()

    quality_score = round(
        ((total_cells - missing_cells) / total_cells) * 100,
        2
    )

    if page == "Overview":

        st.success("File Uploaded Successfully!")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Rows", df.shape[0])

        with col2:
            st.metric("Columns", df.shape[1])

        with col3:
            st.metric("Duplicates", df.duplicated().sum())

        with col4:
            st.metric("Quality Score", f"{quality_score}%")

        st.subheader("Dataset Shape")
        st.write(df.shape)

        st.subheader("Dataset Head")
        st.dataframe(df.head())

        st.subheader("Dataset Tail")
        st.dataframe(df.tail())

        st.subheader("Column Names")
        st.write(df.columns.tolist())

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Duplicate Rows")
        st.write(df.duplicated().sum())

        st.subheader("Statistical Summary")
        st.dataframe(df.describe(include="all"))

        st.subheader("Dataset Information")

        info_df = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum().values
        })

        st.dataframe(info_df)

    elif page == "Visualizations":

        st.subheader("Correlation Heatmap")

        numeric_df = df.select_dtypes(include=["number"])

        if len(numeric_df.columns) > 1:

            corr = numeric_df.corr()

            fig = px.imshow(
                corr,
                text_auto=True,
                title="Correlation Heatmap"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Missing Values Chart")

        missing = df.isnull().sum()

        fig = px.bar(
            x=missing.index,
            y=missing.values,
            title="Missing Values"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Automatic Histograms")

        numeric_cols = df.select_dtypes(
            include=["number"]
        ).columns

        for col in numeric_cols[:3]:

            fig = px.histogram(
                df,
                x=col,
                title=f"Distribution of {col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.subheader("Outlier Detection")

        for col in numeric_cols[:3]:

            fig = px.box(
                df,
                y=col,
                title=f"Outliers in {col}"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    elif page == "Insights":

        st.subheader("AI Insights")

        if quality_score >= 95:
            quality = "Excellent"
        elif quality_score >= 80:
            quality = "Good"
        elif quality_score >= 60:
            quality = "Average"
        else:
            quality = "Poor"

        st.info(
            f"""
Rows: {df.shape[0]}

Columns: {df.shape[1]}

Missing Values: {missing_cells}

Duplicate Rows: {df.duplicated().sum()}

Dataset Quality: {quality}
"""
        )

        csv = df.to_csv(index=False)

        st.download_button(
            label="Download Dataset",
            data=csv,
            file_name="dataset.csv",
            mime="text/csv"
        )
    elif page == "Reports":

        st.subheader("Dataset Memory Usage")

        memory_usage = df.memory_usage(
            deep=True
        ).sum() / 1024**2

        st.metric(
            "Memory Usage (MB)",
            round(memory_usage, 2)
        )

        st.subheader("Unique Values Analysis")

        unique_df = pd.DataFrame({
            "Column": df.columns,
            "Unique Values": [
                df[col].nunique()
                for col in df.columns
            ]
        })

        st.dataframe(unique_df)

        st.subheader("Target Column Analysis")

        target_col = st.selectbox(
            "Select Target Column",
            df.columns
        )

        st.write(
            df[target_col].value_counts()
        )

        fig = px.pie(
            names=df[target_col].value_counts().index,
            values=df[target_col].value_counts().values,
            title=f"{target_col} Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader("Sweetviz Report")

        if st.button(
            "Generate Sweetviz Report"
        ):

            os.makedirs(
                "sweetviz_reports",
                exist_ok=True
            )

            report = sv.analyze(df)

            report_path = (
                "sweetviz_reports/report.html"
            )

            report.show_html(
                report_path,
                open_browser=False
            )

            st.success(
                "Report Generated Successfully!"
            )

            with open(
                report_path,
                "rb"
            ) as file:

                st.download_button(
                    "Download Sweetviz Report",
                    file,
                    file_name="sweetviz_report.html"
                )

st.markdown("---")
st.markdown("Auto EDA Dashboard Version 1 | Built by Abdul Hakeem Amer")