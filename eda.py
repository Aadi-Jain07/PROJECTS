
"""
Exploratory Data Analysis (EDA) Module
Student Performance Prediction System

Generates publication-quality statistical charts and insights for:
1. Study hours vs. final marks (with regression trend)
2. Attendance percentage vs. performance
3. Performance category distribution and target distribution (KDE)
4. Correlation Heatmap
5. Sleep hours vs. marks (illustrating non-linear optimal rest impact)
"""

import os
import matplotlib
matplotlib.use('Agg') # Headless backend for reliable server-side rendering
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set consistent visual style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def ensure_plots_dir(base_dir: str) -> str:
    plots_dir = os.path.join(base_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    return plots_dir


def plot_study_vs_marks(df: pd.DataFrame, output_dir: str) -> str:
    """Plot Study Hours vs Final Marks with Regression Trend."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    sns.regplot(
        data=df,
        x='study_hours',
        y='final_marks',
        scatter_kws={'alpha': 0.55, 'color': '#2563EB', 's': 30},
        line_kws={'color': '#DC2626', 'linewidth': 2.5, 'label': 'Linear Trendline'},
        ax=ax
    )
    ax.set_title("Study Hours per Day vs. Final Marks", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Daily Study Hours (hrs)", fontsize=11, labelpad=8)
    ax.set_ylabel("Final Examination Marks", fontsize=11, labelpad=8)
    ax.set_ylim(15, 105)
    ax.legend(frameon=True, loc='upper left')
    plt.tight_layout()

    out_file = os.path.join(output_dir, 'study_vs_marks.png')
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def plot_attendance_vs_marks(df: pd.DataFrame, output_dir: str) -> str:
    """Plot Attendance Percentage vs Final Marks with Category Highlighting."""
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    scatter = sns.scatterplot(
        data=df,
        x='attendance_percentage',
        y='final_marks',
        hue='performance_category',
        palette={'Excellent': '#10B981', 'Good': '#3B82F6', 'Average': '#F59E0B', 'Needs Improvement': '#EF4444'},
        alpha=0.7,
        s=40,
        ax=ax
    )
    # Add institutional mandatory attendance threshold marker at 75%
    ax.axvline(75, color='#DC2626', linestyle='--', linewidth=1.5, label='75% Minimum Criteria')
    ax.set_title("Attendance Percentage vs. Final Marks", fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel("Classroom Attendance (%)", fontsize=11, labelpad=8)
    ax.set_ylabel("Final Examination Marks", fontsize=11, labelpad=8)
    ax.legend(title='Category', loc='lower right', frameon=True)
    plt.tight_layout()

    out_file = os.path.join(output_dir, 'attendance_vs_marks.png')
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def plot_performance_distribution(df: pd.DataFrame, output_dir: str) -> str:
    """Plot distribution of Final Marks and Academic Tiers."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=150)

    # 1. Continuous Distribution (Histogram + KDE)
    sns.histplot(df['final_marks'], kde=True, color='#4F46E5', bins=25, ax=ax1, edgecolor='white')
    ax1.set_title("Distribution of Final Examination Marks", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Final Marks", fontsize=11)
    ax1.set_ylabel("Student Count", fontsize=11)

    mean_marks = df['final_marks'].mean()
    median_marks = df['final_marks'].median()
    ax1.axvline(mean_marks, color='#EF4444', linestyle='-', linewidth=2, label=f'Mean ({mean_marks:.1f})')
    ax1.axvline(median_marks, color='#10B981', linestyle='--', linewidth=2, label=f'Median ({median_marks:.1f})')
    ax1.legend()

    # 2. Performance Category Count
    order = ['Needs Improvement', 'Average', 'Good', 'Excellent']
    palette = {'Needs Improvement': '#EF4444', 'Average': '#F59E0B', 'Good': '#3B82F6', 'Excellent': '#10B981'}
    sns.countplot(data=df, x='performance_category', order=order, hue='performance_category', palette=palette, legend=False, ax=ax2)
    ax2.set_title("Student Count by Academic Performance Tier", fontsize=13, fontweight='bold', pad=10)
    ax2.set_xlabel("Academic Tier", fontsize=11)
    ax2.set_ylabel("Number of Students", fontsize=11)

    # Add count labels on bars
    for p in ax2.patches:
        val = int(p.get_height())
        ax2.annotate(f"{val}", (p.get_x() + p.get_width() / 2., p.get_height()),
                     ha='center', va='bottom', fontsize=10, fontweight='bold', xytext=(0, 3),
                     textcoords='offset points')

    plt.tight_layout()
    out_file = os.path.join(output_dir, 'marks_distribution.png')
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: str) -> str:
    """Plot Pearson Correlation Heatmap for Numerical Features."""
    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=150)

    num_cols = ['study_hours', 'attendance_percentage', 'previous_marks', 'assignment_completion', 'sleep_hours', 'final_marks']
    corr_df = df[num_cols].rename(columns={
        'study_hours': 'Study Hours',
        'attendance_percentage': 'Attendance %',
        'previous_marks': 'Prev Marks',
        'assignment_completion': 'Assignments %',
        'sleep_hours': 'Sleep Hours',
        'final_marks': 'Final Marks'
    }).corr()

    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr_df,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        vmin=-0.2,
        vmax=1.0,
        square=True,
        linewidths=1.2,
        cbar_kws={"shrink": 0.8},
        ax=ax
    )
    ax.set_title("Correlation Heatmap: Features vs. Academic Marks", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()

    out_file = os.path.join(output_dir, 'correlation_heatmap.png')
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def run_full_eda(csv_path: str = None) -> dict:
    """
    Run full EDA pipeline and save charts.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if csv_path is None:
        csv_path = os.path.join(base_dir, 'student_data.csv')

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at: {csv_path}. Run generate_data.py first.")

    df = pd.read_csv(csv_path)
    plots_dir = ensure_plots_dir(base_dir)

    print(f"Loaded dataset with {len(df)} records. Generating EDA plots...")

    plot_study = plot_study_vs_marks(df, plots_dir)
    plot_att = plot_attendance_vs_marks(df, plots_dir)
    plot_dist = plot_performance_distribution(df, plots_dir)
    plot_corr = plot_correlation_heatmap(df, plots_dir)

    print(f"All EDA visualizations successfully generated in {plots_dir}")

    return {
        'study_vs_marks': plot_study,
        'attendance_vs_marks': plot_att,
        'marks_distribution': plot_dist,
        'correlation_heatmap': plot_corr
    }


if __name__ == '__main__':
    run_full_eda()
Displaying app.py.
