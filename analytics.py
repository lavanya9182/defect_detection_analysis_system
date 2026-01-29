'''import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io

def calculate_stats(df):
    """Calculate summary statistics from the dataframe."""
    if df.empty:
        return {
            "total": 0,
            "defect_count": 0,
            "good_count": 0,
            "defect_rate": 0.0,
            "most_frequent_defect": "None"
        }
    
    total = len(df)
    defect_count = len(df[df['status'] == 'Defect'])
    good_count = len(df[df['status'] == 'Good'])
    defect_rate = (defect_count / total) * 100 if total > 0 else 0
    
    # Most frequent defect (excluding 'None' or 'Normal')
    defects_only = df[df['status'] == 'Defect']
    if not defects_only.empty:
        most_frequent_defect = defects_only['defect_type'].mode()[0]
    else:
        most_frequent_defect = "None"
        
    return {
        "total": total,
        "defect_count": defect_count,
        "good_count": good_count,
        "defect_rate": defect_rate,
        "most_frequent_defect": most_frequent_defect
    }

def get_defect_distribution_chart(df):
    """Generate a pie chart for defect distribution."""
    if df.empty:
        return None
        
    defects_only = df[df['status'] == 'Defect']
    if defects_only.empty:
        return None
        
    counts = defects_only['defect_type'].value_counts()
    
    fig, ax = plt.subplots()
    ax.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("pastel"))
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title("Defect Type Distribution")
    return fig

def get_status_distribution_chart(df):
    """Generate a bar chart for Good vs Defect."""
    if df.empty:
        return None
        
    counts = df['status'].value_counts()
    
    fig, ax = plt.subplots()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, palette=['#FF9999', '#66B2FF'], legend=False)
    ax.set_title("Inspection Status (Good vs Defect)")
    ax.set_ylabel("Count")
    return fig

def get_trend_chart(df):
    """Generate a line chart for inspections over time."""
    if df.empty:
        return None
    
    # Resample by hour or day depending on data volume, for now simple plot
    # Let's plot inspections over time (index) or bucket by date
    df_sorted = df.sort_values('timestamp')
    
    fig, ax = plt.subplots(figsize=(10, 4))
    # Plot simply the cumulative count or just localized count?
    # Let's plot counts per day/hour
    # Group by date-hour
    
    # Just a scatter plot of scores over time is also interesting for QA
    sns.lineplot(data=df_sorted, x='timestamp', y='score', hue='status', marker="o", ax=ax)
    plt.xticks(rotation=45)
    ax.set_title("Anomaly Scores Over Time")
    plt.tight_layout()
    return fig
'''

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Required for Streamlit Cloud
import matplotlib.pyplot as plt


# ==============================
# STATISTICS
# ==============================
def calculate_stats(df):
    """Calculate summary statistics from the dataframe."""
    if df.empty:
        return {
            "total": 0,
            "defect_count": 0,
            "good_count": 0,
            "defect_rate": 0.0,
            "most_frequent_defect": "None"
        }

    total = len(df)
    defect_count = len(df[df["status"] == "Defect"])
    good_count = len(df[df["status"] == "Good"])
    defect_rate = (defect_count / total) * 100 if total > 0 else 0

    defects_only = df[df["status"] == "Defect"]
    most_frequent_defect = (
        defects_only["defect_type"].mode()[0]
        if not defects_only.empty
        else "None"
    )

    return {
        "total": total,
        "defect_count": defect_count,
        "good_count": good_count,
        "defect_rate": defect_rate,
        "most_frequent_defect": most_frequent_defect
    }


# ==============================
# DEFECT DISTRIBUTION (PIE)
# ==============================
def get_defect_distribution_chart(df):
    """Generate a pie chart for defect type distribution."""
    defects_only = df[df["status"] == "Defect"]

    if defects_only.empty:
        return None

    counts = defects_only["defect_type"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        counts.values,
        labels=counts.index,
        autopct="%1.1f%%",
        startangle=90
    )
    ax.axis("equal")
    ax.set_title("Defect Type Distribution")

    plt.tight_layout()
    return fig


# ==============================
# STATUS DISTRIBUTION (BAR)
# ==============================
def get_status_distribution_chart(df):
    """Generate a bar chart for Good vs Defect."""
    if df.empty:
        return None

    counts = df["status"].value_counts()

    fig, ax = plt.subplots(figsize=(5, 4))
    counts.plot(kind="bar", ax=ax, color=["#66B2FF", "#FF9999"])

    ax.set_title("Inspection Status (Good vs Defect)")
    ax.set_xlabel("Status")
    ax.set_ylabel("Count")

    plt.tight_layout()
    return fig


# ==============================
# TREND CHART (LINE)
# ==============================
def get_trend_chart(df):
    """Generate a line chart of anomaly scores over time."""
    if df.empty or "timestamp" not in df.columns:
        return None

    df_sorted = df.sort_values("timestamp")

    fig, ax = plt.subplots(figsize=(10, 4))

    for status in df_sorted["status"].unique():
        subset = df_sorted[df_sorted["status"] == status]
        ax.plot(
            subset["timestamp"],
            subset["score"],
            marker="o",
            label=status
        )

    ax.set_title("Anomaly Scores Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Anomaly Score")
    ax.legend()
    plt.xticks(rotation=45)

    plt.tight_layout()
    return fig



