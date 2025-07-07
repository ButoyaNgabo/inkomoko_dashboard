import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import plotly.graph_objects as go
# --- Page Config ---
st.set_page_config(page_title="Inkomoko Data Collection Dashboard", layout="wide")

# --- Custom Blue / Dark Theme ---
st.markdown("""
    <style>
    body, .stApp {
        background-color: #001f3f;
        color: #FFFFFF;
    }
    .css-18e3th9, .css-1d391kg, .css-1kyxreq {
        background-color: #001f3f !important;
        color: #FFFFFF !important;
    }
    .css-1v0mbdj p, .css-1v0mbdj h1, .css-1v0mbdj h2, .css-1v0mbdj h3 {
        color: #FFFFFF !important;
    }
    .stDownloadButton>button {
        background-color: #0074D9;
        color: #FFFFFF;
        border: 1px solid #0055AA;
    }
    .css-1d391kg {
        background-color: #003366 !important;
    }
    .stDataFrame table {
        color: #FFFFFF !important;
        background-color: #003366 !important;
    }
    section[data-testid="stSidebar"] {
        background-color: #003366;
        color: white;
        padding: 25px;
        border-radius: 10px;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.25);
    }
    .stSelectbox label, .stMultiSelect label, .stDateInput label,
    .stRadio label, .stSlider label {
        color: white !important;
        font-weight: 500;
    }
    select:hover, input:hover {
        border: 1px solid #66B2FF;
    }
    .stButton>button {
        background-color: #0055AA;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 6px 12px;
    }
    .stButton>button:hover {
        background-color: #0074D9;
    }
    </style>
""", unsafe_allow_html=True)

# --- Dark Layout for Plotly ---
dark_layout = dict(
    plot_bgcolor='#121212',
    paper_bgcolor='#121212',
    font=dict(color='white', size=12, family='Arial'),
    xaxis=dict(
        title_font=dict(color='white', size=14, family='Arial'),
        tickfont=dict(color='white', size=12),
        gridcolor='#444444',
        linecolor='white',
        zerolinecolor='#666666'
    ),
    yaxis=dict(
        title_font=dict(color='white', size=14, family='Arial'),
        tickfont=dict(color='white', size=12),
        gridcolor='#444444',
        linecolor='white',
        zerolinecolor='#666666'
    ),
    legend=dict(
        font=dict(color='white'),
        bgcolor='#121212'
    ),
    title_font=dict(color='white')
)

# --- Load Data ---
data_path = "inkomoko_sample_data.csv"
try:
    df = pd.read_csv(data_path, parse_dates=["submission_date"])
except FileNotFoundError:
    st.sidebar.error(f"File '{data_path}' not found.")
    st.stop()

# --- Sample Targets ---
sample_targets = {
    "Bugesera": 46,
    "Kirehe | Mahama": 215,
    "Rubavu": 120,
    "Rusizi | Kamembe": 18
}

# --- Sidebar Filters (Shared for both tabs) ---
st.sidebar.header("Filter Your View")

selected_districts = st.sidebar.multiselect(
    "Districts", options=df["district"].unique(), default=list(sample_targets.keys())
)

selected_enumerators = st.sidebar.multiselect(
    "Enumerators", options=df["enumerator"].unique(), default=df["enumerator"].unique()
)
date_min, date_max = df["submission_date"].min(), df["submission_date"].max()
selected_date = st.sidebar.date_input(
    "Date Range", value=(date_min, date_max), min_value=date_min, max_value=date_max
)

selected_status = st.sidebar.multiselect(
    "Status", options=df["response_status"].unique(), default=df["response_status"].unique()
)

# --- Empty Selection Checks with Warning and Stop ---
if not selected_districts:
    st.warning("⚠️ Please select at least one district to proceed.")
    st.stop()

if not selected_enumerators:
    st.warning("⚠️ Please select at least one enumerator to proceed.")
    st.stop()

if not selected_status:
    st.warning("⚠️ Please select at least one response status to proceed.")
    st.stop()

# --- Additional sidebar inputs for progress dashboard ---
trend_type = st.sidebar.radio("Trend Type", ["Daily", "Cumulative"], horizontal=True)
backcheck_percentage = st.sidebar.slider("Backcheck Percentage", 1,70, 10)

# --- Filter Data Based on Sidebar ---
df_filtered = df[
    df["district"].isin(selected_districts) &
    df["enumerator"].isin(selected_enumerators) &
    df["submission_date"].between(pd.to_datetime(selected_date[0]), pd.to_datetime(selected_date[1])) &
    df["response_status"].isin(selected_status)
]
# --- Tabs ---
tab1, tab2 = st.tabs(["Progress Dashboard", "Data Cleaning"])

# === Tab 1: Progress Dashboard ===
with tab1:
    st.markdown("""
        <div style='background-color: #001f3f; padding: 15px 10px; border-radius: 10px;'>
            <h1 style='text-align: center; color: #FFFFFF;'>Inkomoko Data Collection Progress</h1>
            <hr style='border: 1px solid #FFFFFF;' />
        </div>
    """, unsafe_allow_html=True)

    # Summary Calculation
    total_target = sum([sample_targets.get(d, 0) for d in selected_districts])
    total_collected = df_filtered[df_filtered["response_status"] == "Completed"].shape[0]
    total_non_response = df_filtered[df_filtered["response_status"] == "Non-response"].shape[0]
    percent_complete = (total_collected / total_target * 100) if total_target > 0 else 0

    def metric_card(title, value, subtitle=""):
        st.markdown(f"""
            <div style="
                background-color:#ffffff10;
                padding:20px;
                border:1px solid #1f77b4;
                border-radius:12px;
                text-align:center;
                margin-bottom:10px;
                min-height:120px;
                display:flex;
                flex-direction: column;
                justify-content: center;
            ">
                <div style="font-size:14px; font-weight:600; color:#CCCCCC; margin-bottom:8px;">{title}</div>
                <div style="font-size:26px; font-weight:bold; color:#FFFFFF;">{value}</div>
                <div style="font-size:14px; color:#AAAAAA; margin-top:8px;">{subtitle}</div>
            </div>
        """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Target Sample Size", total_target)
    with col2:
        metric_card("Interviews Completed", total_collected, f"{percent_complete:.1f}%")
    with col3:
        metric_card("Incomplete Interviews", total_non_response)
    with col4:
        metric_card("Coverage Period", "2025-07-15", "to 2025-07-25")

    # District Progress Overview
    st.subheader("District-wise Progress Overview")

    progress_data = []
    for d in selected_districts:
        completed = df_filtered[(df_filtered["district"] == d) & (df_filtered["response_status"] == "Completed")].shape[0]
        target = sample_targets.get(d, 0)
        pct = round((completed / target * 100), 2) if target > 0 else 0.0
        progress_data.append({
            "District": d,
            "Completed": completed,
            "Target": target,
            "Completion %": pct
        })

    progress_df = pd.DataFrame(progress_data).sort_values("Completion %", ascending=False)
    progress_df["Label"] = progress_df["Completion %"].astype(str) + "%"
    progress_df["HoverText"] = (
        "District: " + progress_df["District"] +
        "<br>Completed: " + progress_df["Completed"].astype(str) +
        " out of " + progress_df["Target"].astype(str) +
        " (" + progress_df["Label"] + ")"
    )

    blue_gradient = ["#F2F2F2", "#9DC3E5", "#097ABC"]

    fig = px.bar(
        progress_df,
        x="District",
        y="Completion %",
        text="Label",
        labels={"Completion %": "Completion (%)"},
        range_y=[0, 110],
        title="Completion Progress by District",
        color="Completion %",
        color_continuous_scale=blue_gradient
    )
    fig.update_traces(textposition="inside", hovertemplate=progress_df["HoverText"])
    fig.update_layout(**dark_layout)
    st.plotly_chart(fig, use_container_width=True)

    # Interview Quality Monitoring
    st.subheader("Interview Quality Monitoring")
    iq_tabs = st.tabs(["Interview Duration", "GPS Issues", "Duplicates", "Supervisor Notes"])

    with iq_tabs[0]:
        duration_data = df_filtered[df_filtered["response_status"] == "Completed"].copy()
        if 'enumerators' not in duration_data.columns and 'enumerator' in duration_data.columns:
            duration_data['enumerators'] = duration_data['enumerator']
        elif 'enumerators' not in duration_data.columns:
            duration_data['enumerators'] = "N/A"

        bin_width = 5
        bin_edges = list(range(0, 101, bin_width))
        bin_labels = [f"{left}–{left + bin_width}" for left in bin_edges[:-1]]
        duration_data["bin"] = pd.cut(
            duration_data["interview_duration_min"],
            bins=bin_edges,
            labels=bin_labels,
            include_lowest=True,
            right=False
        )

        grouped = duration_data.groupby(["bin", "district"]).agg({
            "enumerators": lambda x: ", ".join(sorted(set(x.dropna()))),
            "interview_duration_min": "count"
        }).reset_index().rename(columns={"interview_duration_min": "interview_count"})
        grouped = grouped[grouped["interview_count"] > 0]
        grouped = grouped.sort_values(by=["bin", "district"]).reset_index(drop=True)
        grouped["bin"] = grouped["bin"].astype(str).fillna("Unknown")
        grouped["district"] = grouped["district"].astype(str).fillna("Unknown")
        grouped["hover_text"] = (
            "District: " + grouped["district"] + "<br>" +
            "Duration Range: " + grouped["bin"] + " min<br>" +
            "Interviews: " + grouped["interview_count"].astype(str) + "<br>" +
            "Enumerators: " + grouped["enumerators"]
        )
        district_colors_list = [
            "#33A1C9", "#66C2A5", "#FC8D62", "#E78AC3", "#A6D854", "#FFD92F", "#8DA0CB",
        ]
        unique_districts = grouped["district"].unique()
        color_map = {district: district_colors_list[i % len(district_colors_list)] for i, district in enumerate(unique_districts)}

        fig_hist = px.bar(
            grouped,
            x="bin",
            y="interview_count",
            color="district",
            category_orders={"bin": bin_labels},
            color_discrete_map=color_map,
            labels={
                "bin": "Interview Duration (minutes)",
                "interview_count": "Number of Interviews",
                "district": "Districts",
            },
            title="Interview Duration Distribution",
        )
        fig_hist.update_traces(
            hovertemplate=(
                "District: %{customdata[1]}<br>" +
                "Duration Range: %{customdata[0]} min<br>" +
                "Interviews: %{customdata[2]}<br>" +
                "Enumerators: %{customdata[3]}<extra></extra>"
            ),
            customdata=grouped[["bin", "district", "interview_count", "enumerators"]].values,
            marker_line_color='white',
            marker_line_width=1,
        )
        fig_hist.update_layout(
            title_font_color="#66C2A5",
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
            font_color="#F2F2F2",
            xaxis_tickangle=-45,
            barmode="group"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        duration_data['hover_text'] = duration_data.apply(
            lambda row: (
                f"District: {row['district']}<br>"
                f"Duration: {int(row['interview_duration_min'])} min<br>"
                f"Enumerators: {row['enumerators']}"
            ),
            axis=1
        )

        fig_box = px.box(
            duration_data,
            x="district",
            y="interview_duration_min",
            color="district",
            points="all",
            labels={
                "interview_duration_min": "Interview Duration (minutes)",
                "district": "Districts",
            },
            color_discrete_map=color_map,
            title="Interview Duration Spread",
        )

        fig_box.update_traces(
            hovertemplate='%{customdata[0]}<extra></extra>',
            customdata=duration_data[['hover_text']].values
        )
        fig_box.update_layout(
            title_font_color="#66C2A5",
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
            font_color="#F2F2F2"
        )
        st.plotly_chart(fig_box, use_container_width=True)

        duration_outliers = duration_data[
            (duration_data["interview_duration_min"] < 20) |
            (duration_data["interview_duration_min"] > 90)
        ]
        st.markdown("### Interviews with Unusual Durations")
        st.markdown(
            f"<div style='color:#8ccc98; font-size:16px; font-weight:bold;'>Total outliers: {duration_outliers.shape[0]}</div>",
            unsafe_allow_html=True
        )
        if not duration_outliers.empty:
            with st.expander("View Outlier Records"):
                st.dataframe(duration_outliers.sort_values("interview_duration_min"))

    with iq_tabs[1]:
        if "gps_mismatch" in df_filtered.columns:
            gps_flags = df_filtered[df_filtered["gps_mismatch"] == True]
            st.write(f"Interviews with GPS mismatch: {gps_flags.shape[0]}")
            if not gps_flags.empty:
                st.dataframe(gps_flags)
        else:
            st.write("No GPS mismatch data available.")

    with iq_tabs[2]:
        dupes = df_filtered[df_filtered.duplicated(subset=["submission_id"], keep=False)]
        st.write(f"Duplicate Submission IDs: {dupes['submission_id'].nunique()}")
        if not dupes.empty:
            st.dataframe(dupes)

    with iq_tabs[3]:
        if "supervisor_note" in df_filtered.columns:
            flagged_notes = df_filtered[df_filtered["supervisor_note"].notna()]
            st.write(f"Interviews flagged by supervisors: {flagged_notes.shape[0]}")
            if not flagged_notes.empty:
                st.dataframe(flagged_notes)
        else:
            st.write("No supervisor notes available.")

    # Submission Trends
st.subheader("Survey Submission Trends")
daily_subs = df_filtered.groupby(["submission_date", "response_status"]).size().unstack(fill_value=0).sort_index()
title = "Cumulative Submissions" if trend_type == "Cumulative" else "Daily Submissions"

fig = px.line(
    daily_subs.cumsum() if trend_type == "Cumulative" else daily_subs,
    y=["Completed", "Non-response"],
    title=title
)
fig.update_traces(
    hovertemplate="Date: %{x|%b %d, %Y}<br>%{fullData.name}<br>Submitted: %{y}<extra></extra>"
)
fig.update_layout(**dark_layout)
st.plotly_chart(fig, use_container_width=True)


    # Missing Data Overview
st.subheader("Missing Data Overview")
missing_data = df_filtered.isna().mean() * 100
missing_df = missing_data.reset_index()
missing_df.columns = ["Field", "Percent Missing"]
missing_df = missing_df[missing_df["Percent Missing"] > 0]
missing_df = missing_df.sort_values("Percent Missing", ascending=False)

fig = px.bar(
        missing_df,
        x="Field",
        y="Percent Missing",
        title="Missing Data by Field",
        text=missing_df["Percent Missing"].apply(lambda x: f"{x:.1f}%"),
    )
fig.update_layout(
        **dark_layout,
        xaxis_tickangle=-45,
        yaxis_title="Percent Missing (%)",
        yaxis_range=[0, 100],
    )
fig.add_hline(
        y=20,
        line_dash="dash",
        line_color="red",
        annotation_text="20% threshold",
        annotation_position="top left",
        annotation_font_color="red",
    )
fig.update_traces(
        hovertemplate="<b>%{x}</b><br>Missing: %{y:.1f}%<extra></extra>",
        textposition='outside',
    )
st.plotly_chart(fig, use_container_width=True)

# --- Backcheck Sample Tracker ---
st.subheader("Backcheck Sample Tracker (Stratified Sampling)")

# 1. Select districts to include in backcheck
districts_for_backcheck = df_filtered["district"].unique()
selected_backcheck_districts = st.multiselect("Select Districts for Backcheck", options=districts_for_backcheck, default=districts_for_backcheck)

# 2. Filter data by selected districts
sample_pool = df_filtered[df_filtered["district"].isin(selected_backcheck_districts)]

# 3. Compute stratified backcheck sample
backcheck_stratified = []

for district in selected_backcheck_districts:
    df_district = sample_pool[sample_pool["district"] == district]
    n_total = df_district.shape[0]
    n_backcheck = int(np.ceil((backcheck_percentage / 100) * n_total))  # stratified sample size
    sample_district = df_district.sample(n=n_backcheck, random_state=42)
    backcheck_stratified.append(sample_district)

backcheck_sample = pd.concat(backcheck_stratified).reset_index(drop=True)

# 4. Show backcheck sample table
st.write(f"Total Backcheck Interviews: {backcheck_sample.shape[0]}")
st.write("Backcheck Sample by District:")
st.dataframe(backcheck_sample)

# 5. Download button
csv_backcheck = backcheck_sample.to_csv(index=False).encode("utf-8")
st.download_button("Download Backcheck Sample (CSV)", data=csv_backcheck, file_name="backcheck_sample.csv", mime="text/csv")

# 6. Prepare plot data
backcheck_counts = backcheck_sample["district"].value_counts().reset_index()
backcheck_counts.columns = ["district", "backcheck_count"]

# 7. Custom hover text (only using backcheck info now)
backcheck_counts["hover_text"] = backcheck_counts.apply(
    lambda row: f"<b>District:</b> {row['district']}<br><b>Backchecks:</b> {row['backcheck_count']}",
    axis=1
)

# 8. Plot bar chart of backcheck sample per district
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Bar(
    x=backcheck_counts["district"],
    y=backcheck_counts["backcheck_count"],
    text=backcheck_counts["backcheck_count"],
    textposition="inside",
    marker=dict(
        color=backcheck_counts["backcheck_count"],
        colorscale="Blues",
        line=dict(color="white", width=1.5)
    ),
    customdata=backcheck_counts["hover_text"],
    hovertemplate="%{customdata}<extra></extra>"
))

fig.update_layout(
    title="Stratified Backcheck Sample Size per District",
    plot_bgcolor="#121212",
    paper_bgcolor="#121212",
    font=dict(color="white", size=14),
    xaxis=dict(title="District", tickangle=-45),
    yaxis=dict(title="Number of Backchecks"),
    margin=dict(t=60, b=80)
)

st.plotly_chart(fig, use_container_width=True)


# === Tab 2: Enhanced Data Cleaning Pipeline ===
with tab2:
    st.subheader("Advanced Data Cleaning Pipeline")

    cleaned_df = df_filtered.copy()

    all_cols = cleaned_df.columns.tolist()

    # 1. Select columns to clean
    selected_cols = st.multiselect("Select columns to clean", options=all_cols, default=all_cols)

    # 2. Show initial stats before cleaning
    st.markdown("#### Summary Statistics Before Cleaning")
    st.dataframe(cleaned_df[selected_cols].describe(include='all').T)

    # 3. Remove duplicate rows option with column control
    remove_dupes = st.checkbox("Remove duplicate rows?", value=True)
    if remove_dupes:
        subset_cols = st.multiselect("Columns to consider for duplicates", options=selected_cols, default=selected_cols)
        before = cleaned_df.shape[0]
        cleaned_df = cleaned_df.drop_duplicates(subset=subset_cols)
        removed = before - cleaned_df.shape[0]
        st.success(f"Removed {removed} duplicate rows based on columns: {subset_cols}")

    st.markdown("---")

    # 4. Handle missing values
    st.markdown("#### Handle Missing Values")
    miss_cols = st.multiselect("Columns to handle missing values for", options=selected_cols, default=selected_cols)
    miss_strategy = st.radio("Missing Value Strategy", ["Do nothing", "Drop rows with missing", "Fill with Mean", "Fill with Median", "Fill with Mode", "Forward Fill", "Backward Fill"])

    if miss_strategy != "Do nothing":
        if miss_strategy == "Drop rows with missing":
            before = cleaned_df.shape[0]
            cleaned_df = cleaned_df.dropna(subset=miss_cols)
            removed = before - cleaned_df.shape[0]
            st.success(f"Dropped {removed} rows with missing values in selected columns")
        elif miss_strategy == "Fill with Mean":
            for col in miss_cols:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mean())
            st.success(f"Filled missing values with Mean for columns: {miss_cols}")
        elif miss_strategy == "Fill with Median":
            for col in miss_cols:
                if pd.api.types.is_numeric_dtype(cleaned_df[col]):
                    cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
            st.success(f"Filled missing values with Median for columns: {miss_cols}")
        elif miss_strategy == "Fill with Mode":
            for col in miss_cols:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].mode().iloc[0])
            st.success(f"Filled missing values with Mode for columns: {miss_cols}")
        elif miss_strategy == "Forward Fill":
            cleaned_df[miss_cols] = cleaned_df[miss_cols].fillna(method='ffill')
            st.success(f"Applied Forward Fill on columns: {miss_cols}")
        elif miss_strategy == "Backward Fill":
            cleaned_df[miss_cols] = cleaned_df[miss_cols].fillna(method='bfill')
            st.success(f"Applied Backward Fill on columns: {miss_cols}")

    st.markdown("---")

    # 5. Text column transformations
    st.markdown("#### Text Column Transformations")
    text_cols = [col for col in selected_cols if cleaned_df[col].dtype == 'object']
    text_col = st.selectbox("Select text column to transform", options=[None] + text_cols)
    if text_col:
        text_transform = st.selectbox("Choose transformation", ["None", "Lowercase", "Uppercase", "Capitalize", "Strip whitespace"])
        if text_transform != "None":
            if text_transform == "Lowercase":
                cleaned_df[text_col] = cleaned_df[text_col].str.lower()
            elif text_transform == "Uppercase":
                cleaned_df[text_col] = cleaned_df[text_col].str.upper()
            elif text_transform == "Capitalize":
                cleaned_df[text_col] = cleaned_df[text_col].str.capitalize()
            elif text_transform == "Strip whitespace":
                cleaned_df[text_col] = cleaned_df[text_col].str.strip()
            st.success(f"Applied '{text_transform}' transformation on column '{text_col}'")

    st.markdown("---")

    # 6. Custom value replacement
    st.markdown("#### Custom Value Replacement")
    replace_col = st.selectbox("Select column for value replacement", options=[None] + selected_cols)
    if replace_col:
        old_value = st.text_input("Value to find")
        new_value = st.text_input("Value to replace with")
        if st.button("Replace values"):
            if old_value == "":
                st.error("Please enter a value to find.")
            else:
                cleaned_df[replace_col] = cleaned_df[replace_col].replace(old_value, new_value)
                st.success(f"Replaced '{old_value}' with '{new_value}' in column '{replace_col}'")

    st.markdown("---")

    # 7. Convert data types interactively
    st.markdown("#### Convert Data Types")
    col_type_convert = st.selectbox("Select column to convert type", options=[None] + selected_cols)
    if col_type_convert:
        target_type = st.selectbox("Convert to type", ["int", "float", "string", "category", "datetime"])
        if st.button(f"Convert '{col_type_convert}' to {target_type}"):
            try:
                if target_type == "int":
                    cleaned_df[col_type_convert] = pd.to_numeric(cleaned_df[col_type_convert], errors='coerce').astype('Int64')
                elif target_type == "float":
                    cleaned_df[col_type_convert] = pd.to_numeric(cleaned_df[col_type_convert], errors='coerce')
                elif target_type == "string":
                    cleaned_df[col_type_convert] = cleaned_df[col_type_convert].astype(str)
                elif target_type == "category":
                    cleaned_df[col_type_convert] = cleaned_df[col_type_convert].astype('category')
                elif target_type == "datetime":
                    cleaned_df[col_type_convert] = pd.to_datetime(cleaned_df[col_type_convert], errors='coerce')
                st.success(f"Converted column '{col_type_convert}' to {target_type}")
            except Exception as e:
                st.error(f"Error converting column: {e}")

    st.markdown("---")

    # 8. Rename columns
    st.markdown("#### Rename Columns")
    col_rename_old = st.selectbox("Select column to rename", options=[None] + selected_cols)
    if col_rename_old:
        new_name = st.text_input("New column name")
        if new_name:
            cleaned_df.rename(columns={col_rename_old: new_name}, inplace=True)
            st.success(f"Column '{col_rename_old}' renamed to '{new_name}'")

    st.markdown("---")

    # 9. Remove outliers (IQR) on numeric columns with adjustable multiplier
    st.markdown("#### Remove Outliers (IQR method)")
    outlier_cols = st.multiselect("Select numeric columns to check for outliers", options=cleaned_df.select_dtypes(include='number').columns.tolist())
    iqr_multiplier = st.slider("IQR Multiplier", min_value=1.0, max_value=3.0, step=0.1, value=1.5)
    if st.button("Remove outliers"):
        total_removed = 0
        for col in outlier_cols:
            Q1 = cleaned_df[col].quantile(0.25)
            Q3 = cleaned_df[col].quantile(0.75)
            IQR = Q3 - Q1
            before = cleaned_df.shape[0]
            cleaned_df = cleaned_df[~((cleaned_df[col] < (Q1 - iqr_multiplier * IQR)) | (cleaned_df[col] > (Q3 + iqr_multiplier * IQR)))]
            after = cleaned_df.shape[0]
            removed = before - after
            st.write(f"Outliers removed in '{col}': {removed}")
            total_removed += removed
        st.success(f"Total outliers removed: {total_removed}")

    st.markdown("---")

    # 10. Visualize missing data heatmap with selectable color map
    if st.checkbox("Show Missing Data Heatmap"):
        import matplotlib.pyplot as plt
        import seaborn as sns
        cmap_options = ["viridis", "magma", "plasma", "cividis", "coolwarm", "inferno"]
        selected_cmap = st.selectbox("Select heatmap color map", options=cmap_options, index=0)
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(cleaned_df[selected_cols].isna(), cbar=False, yticklabels=False, cmap=selected_cmap, ax=ax)
        st.pyplot(fig)

    st.markdown("---")

    # 11. Summary stats after cleaning
    st.markdown("#### Summary Statistics After Cleaning")
    st.dataframe(cleaned_df[selected_cols].describe(include='all').T)

    # 12. Preview cleaned data
    st.markdown("#### Preview Cleaned Data")
    st.dataframe(cleaned_df.head(20))

    # 13. Download cleaned dataset
    csv_clean = cleaned_df.to_csv(index=False).encode('utf-8')
    today = datetime.now().strftime("%Y-%m-%d")
    st.download_button("Download Cleaned Dataset (CSV)", data=csv_clean, file_name=f"cleaned_dataset_{today}.csv", mime='text/csv')

    st.markdown("---")

    # 14. Reset cleaning steps button
    if st.button("Reset Cleaning Steps (Reload Tab)"):
        st.experimental_rerun()