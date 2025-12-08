import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import argparse

# Import CEFR expectations from combined_scorer
from combine import CEFR_EXPECTED_FLESCH

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 10

# Output directory
OUTPUT_DIR = 'figures'
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def run_full_analysis(csv_path='eval.csv', baseline_path=None, output_dir=OUTPUT_DIR):
    # Load data
    df = pd.read_csv(csv_path)
    print(f"Main model: {len(df)} samples")
    
    baseline_df = None
    if baseline_path:
        baseline_df = pd.read_csv(baseline_path)
        print(f"Baseline model: {len(baseline_df)} samples")
    
    # Model comparison (if baseline provided)
    comparison_results = None
    if baseline_df is not None:
        comparison_results = compare_models(baseline_df, df, output_dir)
    
    # Descriptive Statistics
    stats_results = descriptive_stats(df, "Fine-tuned Model" if baseline_df is not None else "Model")
    
    # Generate Visualizations
    
    plot_score_overview(df, baseline_df, output_dir)
    plot_score_distribution(df, output_dir)
    plot_cefr_appropriateness(df, output_dir)
    
    # Additional model comparison plots if baseline provided
    if baseline_df is not None:
        plot_detailed_comparison(baseline_df, df, output_dir)
    
    # Generate Report
    generate_report(df, stats_results, comparison_results, baseline_df, output_dir)
    print(f"\nResults saved in: {output_dir}/")


def compare_models(baseline_df, finetuned_df, output_dir):
    """Compare baseline and fine-tuned models"""
    print("MODEL COMPARISON: BASELINE VS FINE-TUNED")
    
    levels = ['A2', 'B1', 'B2']
    metrics = ['cosine_score', 'appropriateness_score', 'total_avg', 'llm_score', 'flesch_reading_ease']
    
    comparison_results = {}
    
    for level in levels:
        print(f"\n{level} Level:")
        print("-" * 50)
        
        level_results = {}
        
        for metric in metrics:
            col_name = f'{level}_simplified_{metric}'
            
            if col_name in baseline_df.columns and col_name in finetuned_df.columns:
                baseline_vals = baseline_df[col_name].dropna()
                finetuned_vals = finetuned_df[col_name].dropna()
                
                if len(baseline_vals) > 0 and len(finetuned_vals) > 0:
                    baseline_mean = baseline_vals.mean()
                    finetuned_mean = finetuned_vals.mean()
                    improvement = ((finetuned_mean - baseline_mean) / baseline_mean) * 100
                    
                    # Statistical test
                    t_stat, p_value = stats.ttest_ind(baseline_vals, finetuned_vals)
                    
                    level_results[metric] = {
                        'baseline': baseline_mean,
                        'finetuned': finetuned_mean,
                        'improvement': improvement,
                        'p_value': p_value
                    }
                    
                    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else ""
                    print(f"  {metric:25s}: {baseline_mean:6.3f} → {finetuned_mean:6.3f} ({improvement:+6.1f}%) {sig}")
        
        comparison_results[level] = level_results
    
    # Generate comparison plot
    plot_model_comparison(baseline_df, finetuned_df, comparison_results, output_dir)
    
    return comparison_results


def plot_model_comparison(baseline_df, finetuned_df, comparison_results, output_dir):
    """Plot model comparison"""
    levels = ['A2', 'B1', 'B2']
    metrics = ['cosine_score', 'appropriateness_score', 'total_avg', 'llm_score']
    metric_labels = ['Cosine Score', 'Appropriateness', 'Total Average', 'LLM Score']
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        baseline_means = []
        finetuned_means = []
        
        for level in levels:
            col_name = f'{level}_simplified_{metric}'
            if col_name in baseline_df.columns and col_name in finetuned_df.columns:
                baseline_means.append(baseline_df[col_name].mean())
                finetuned_means.append(finetuned_df[col_name].mean())
        
        x = np.arange(len(levels))
        width = 0.35
        
        axes[i].bar(x - width/2, baseline_means, width, label='Baseline', color='#e74c3c', alpha=0.8)
        axes[i].bar(x + width/2, finetuned_means, width, label='Fine-tuned', color='#2ecc71', alpha=0.8)
        
        axes[i].set_xlabel('CEFR Level', fontweight='bold')
        axes[i].set_ylabel('Score', fontweight='bold')
        axes[i].set_title(label, fontweight='bold', fontsize=12)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(levels)
        axes[i].legend()
        axes[i].set_ylim(0, 1)
        axes[i].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Model Comparison: Baseline vs Fine-tuned', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/01_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/01_model_comparison.png")


def descriptive_stats(df, model_name="Model"):
    """Calculate and display descriptive statistics"""
    
    levels = ['A2', 'B1', 'B2']
    metrics = ['cosine_score', 'appropriateness_score', 'total_avg', 'llm_score', 'flesch_reading_ease']
    
    stats_results = {}
    
    for level in levels:
        print(f"\n{level} Level:")
        print("-" * 50)
        
        level_stats = {}
        
        for metric in metrics:
            col = f'{level}_simplified_{metric}'
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    level_stats[metric] = {
                        'mean': values.mean(),
                        'std': values.std(),
                        'min': values.min(),
                        'max': values.max(),
                        'median': values.median()
                    }
                    print(f"  {metric:25s}: μ={values.mean():.3f}, σ={values.std():.3f}, median={values.median():.3f}")
        
        # CEFR Appropriateness
        flesch_col = f'{level}_simplified_flesch_reading_ease'
        if flesch_col in df.columns:
            expected = CEFR_EXPECTED_FLESCH[level]['center']
            actual = df[flesch_col].mean()
            deviation = abs(actual - expected)
            print(f"  CEFR Appropriateness: Expected Flesch={expected}, Actual={actual:.1f}, Deviation={deviation:.1f}")
        
        stats_results[level] = level_stats
    
    return stats_results


def plot_score_overview(df, baseline_df=None, output_dir=OUTPUT_DIR):
    """Plot comprehensive score overview"""
    
    levels = ['A2', 'B1', 'B2']
    metrics = ['cosine_score', 'appropriateness_score', 'total_avg', 'llm_score']
    metric_labels = ['Cosine Score', 'Appropriateness', 'Total Average', 'LLM Score']
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Bar chart with all metrics
    means_by_metric = {metric: [] for metric in metrics}
    
    for level in levels:
        for metric in metrics:
            col = f'{level}_simplified_{metric}'
            if col in df.columns:
                means_by_metric[metric].append(df[col].mean())
    
    x = np.arange(len(levels))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        axes[0, 0].bar(x + i*width - 0.3, means_by_metric[metric], width, 
                      label=label, color=colors[i], alpha=0.8)
    
    axes[0, 0].set_xlabel('CEFR Level', fontweight='bold')
    axes[0, 0].set_ylabel('Score', fontweight='bold')
    axes[0, 0].set_title('Score Comparison Across Levels', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(levels)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Total Average boxplot
    total_avg_data = []
    for level in levels:
        col = f'{level}_simplified_total_avg'
        if col in df.columns:
            total_avg_data.append(df[col].dropna().values)
    
    bp = axes[0, 1].boxplot(total_avg_data, labels=levels, patch_artist=True)
    for patch, color in zip(bp['boxes'], ['#e74c3c', '#f39c12', '#3498db']):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0, 1].set_xlabel('CEFR Level', fontweight='bold')
    axes[0, 1].set_ylabel('Total Average Score', fontweight='bold')
    axes[0, 1].set_title('Total Average Score Distribution', fontweight='bold')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Plot 3: Flesch Reading Ease
    flesch_means = []
    flesch_expected = []
    
    for level in levels:
        col = f'{level}_simplified_flesch_reading_ease'
        if col in df.columns:
            flesch_means.append(df[col].mean())
            flesch_expected.append(CEFR_EXPECTED_FLESCH[level]['center'])
    
    x = np.arange(len(levels))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, flesch_expected, width, label='Expected', color='gray', alpha=0.6)
    axes[1, 0].bar(x + width/2, flesch_means, width, label='Actual', color='#16a085', alpha=0.8)
    
    axes[1, 0].set_xlabel('CEFR Level', fontweight='bold')
    axes[1, 0].set_ylabel('Flesch Reading Ease', fontweight='bold')
    axes[1, 0].set_title('Readability: Expected vs Actual', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(levels)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Score trend
    for metric, label, color in zip(metrics, metric_labels, colors):
        means = []
        for level in levels:
            col = f'{level}_simplified_{metric}'
            if col in df.columns:
                means.append(df[col].mean())
        axes[1, 1].plot(levels, means, marker='o', label=label, color=color, linewidth=2, markersize=8)
    
    axes[1, 1].set_xlabel('CEFR Level', fontweight='bold')
    axes[1, 1].set_ylabel('Score', fontweight='bold')
    axes[1, 1].set_title('Score Trends Across Levels', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/02_score_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/02_score_overview.png")


def plot_score_distribution(df, output_dir=OUTPUT_DIR):
    """Plot score distributions"""
    levels = ['A2', 'B1', 'B2']
    colors = ['#e74c3c', '#f39c12', '#3498db']
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    
    # Row 1: Total Average Score histograms
    for i, (level, color) in enumerate(zip(levels, colors)):
        col = f'{level}_simplified_total_avg'
        if col in df.columns:
            data = df[col].dropna()
            axes[0, i].hist(data, bins=20, color=color, alpha=0.7, edgecolor='black')
            axes[0, i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
            axes[0, i].set_xlabel('Score', fontweight='bold')
            axes[0, i].set_ylabel('Frequency', fontweight='bold')
            axes[0, i].set_title(f'{level} - Total Average Score', fontweight='bold')
            axes[0, i].legend()
            axes[0, i].grid(alpha=0.3)
    
    # Row 2: Appropriateness Score histograms
    for i, (level, color) in enumerate(zip(levels, colors)):
        col = f'{level}_simplified_appropriateness_score'
        if col in df.columns:
            data = df[col].dropna()
            axes[1, i].hist(data, bins=20, color=color, alpha=0.7, edgecolor='black')
            axes[1, i].axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {data.mean():.3f}')
            axes[1, i].set_xlabel('Score', fontweight='bold')
            axes[1, i].set_ylabel('Frequency', fontweight='bold')
            axes[1, i].set_title(f'{level} - Appropriateness Score', fontweight='bold')
            axes[1, i].legend()
            axes[1, i].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/03_score_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/03_score_distributions.png")


def plot_cefr_appropriateness(df, output_dir=OUTPUT_DIR):
    """Plot CEFR level appropriateness analysis"""
    
    levels = ['A2', 'B1', 'B2']
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    for i, level in enumerate(levels):
        flesch_col = f'{level}_simplified_flesch_reading_ease'
        approp_col = f'{level}_simplified_appropriateness_score'
        
        if flesch_col in df.columns and approp_col in df.columns:
            flesch_data = df[flesch_col].dropna()
            approp_data = df[approp_col].dropna()
            
            # Scatter plot
            axes[i].scatter(flesch_data, approp_data, alpha=0.5, s=50, color='#3498db')
            
            # Expected Flesch line
            expected = CEFR_EXPECTED_FLESCH[level]['center']
            tolerance = CEFR_EXPECTED_FLESCH[level]['tolerance']
            
            axes[i].axvline(expected, color='red', linestyle='--', linewidth=2, label=f'Expected: {expected}')
            axes[i].axvspan(expected - tolerance, expected + tolerance, alpha=0.2, color='green', label='Tolerance')
            
            axes[i].set_xlabel('Flesch Reading Ease', fontweight='bold')
            axes[i].set_ylabel('Appropriateness Score', fontweight='bold')
            axes[i].set_title(f'{level} Level', fontweight='bold')
            axes[i].legend()
            axes[i].grid(alpha=0.3)
    
    plt.suptitle('CEFR Level Appropriateness Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/04_cefr_appropriateness.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/04_cefr_appropriateness.png")


def plot_detailed_comparison(baseline_df, finetuned_df, output_dir=OUTPUT_DIR):
    """Plot detailed model comparison analysis"""
    levels = ['A2', 'B1', 'B2']
    metrics = ['cosine_score', 'appropriateness_score', 'total_avg', 'llm_score']
    metric_labels = ['Cosine Score', 'Appropriateness', 'Total Average', 'LLM Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Score improvement by level
    improvements = {metric: [] for metric in metrics}
    
    for level in levels:
        for metric in metrics:
            col = f'{level}_simplified_{metric}'
            if col in baseline_df.columns and col in finetuned_df.columns:
                baseline_mean = baseline_df[col].mean()
                finetuned_mean = finetuned_df[col].mean()
                improvement = ((finetuned_mean - baseline_mean) / baseline_mean) * 100
                improvements[metric].append(improvement)
            else:
                improvements[metric].append(0)
    
    x = np.arange(len(levels))
    width = 0.2
    colors = ['#3498db', '#2ecc71', '#9b59b6', '#e74c3c']
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        axes[0, 0].bar(x + i*width - 0.3, improvements[metric], width, 
                      label=label, color=colors[i], alpha=0.8)
    
    axes[0, 0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel('CEFR Level', fontweight='bold')
    axes[0, 0].set_ylabel('Improvement (%)', fontweight='bold')
    axes[0, 0].set_title('Score Improvement: Fine-tuned vs Baseline', fontweight='bold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(levels)
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Plot 2: Total Average comparison across levels
    baseline_total_avgs = []
    finetuned_total_avgs = []
    
    for level in levels:
        col = f'{level}_simplified_total_avg'
        if col in baseline_df.columns:
            baseline_total_avgs.append(baseline_df[col].mean())
        if col in finetuned_df.columns:
            finetuned_total_avgs.append(finetuned_df[col].mean())
    
    axes[0, 1].plot(levels, baseline_total_avgs, marker='o', linewidth=2, 
                   markersize=10, label='Baseline', color='#e74c3c')
    axes[0, 1].plot(levels, finetuned_total_avgs, marker='s', linewidth=2, 
                   markersize=10, label='Fine-tuned', color='#2ecc71')
    
    axes[0, 1].set_xlabel('CEFR Level', fontweight='bold')
    axes[0, 1].set_ylabel('Total Average Score', fontweight='bold')
    axes[0, 1].set_title('Total Average Score Trend', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Flesch Reading Ease comparison
    baseline_flesch = []
    finetuned_flesch = []
    expected_flesch = []
    
    for level in levels:
        col = f'{level}_simplified_flesch_reading_ease'
        if col in baseline_df.columns:
            baseline_flesch.append(baseline_df[col].mean())
        if col in finetuned_df.columns:
            finetuned_flesch.append(finetuned_df[col].mean())
        expected_flesch.append(CEFR_EXPECTED_FLESCH[level]['center'])
    
    x = np.arange(len(levels))
    width = 0.25
    
    axes[1, 0].bar(x - width, expected_flesch, width, label='Expected', color='gray', alpha=0.6)
    axes[1, 0].bar(x, baseline_flesch, width, label='Baseline', color='#e74c3c', alpha=0.8)
    axes[1, 0].bar(x + width, finetuned_flesch, width, label='Fine-tuned', color='#2ecc71', alpha=0.8)
    
    axes[1, 0].set_xlabel('CEFR Level', fontweight='bold')
    axes[1, 0].set_ylabel('Flesch Reading Ease', fontweight='bold')
    axes[1, 0].set_title('Readability Comparison', fontweight='bold')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(levels)
    axes[1, 0].legend()
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Plot 4: Score distribution comparison (box plots)
    data_baseline = []
    data_finetuned = []
    
    for level in levels:
        col = f'{level}_simplified_total_avg'
        if col in baseline_df.columns:
            data_baseline.append(baseline_df[col].dropna().values)
        if col in finetuned_df.columns:
            data_finetuned.append(finetuned_df[col].dropna().values)
    
    positions_baseline = [1, 4, 7]
    positions_finetuned = [2, 5, 8]
    
    bp1 = axes[1, 1].boxplot(data_baseline, positions=positions_baseline, 
                            widths=0.6, patch_artist=True,
                            boxprops=dict(facecolor='#e74c3c', alpha=0.7),
                            medianprops=dict(color='darkred', linewidth=2))
    
    bp2 = axes[1, 1].boxplot(data_finetuned, positions=positions_finetuned, 
                            widths=0.6, patch_artist=True,
                            boxprops=dict(facecolor='#2ecc71', alpha=0.7),
                            medianprops=dict(color='darkgreen', linewidth=2))
    
    axes[1, 1].set_xlabel('CEFR Level', fontweight='bold')
    axes[1, 1].set_ylabel('Total Average Score', fontweight='bold')
    axes[1, 1].set_title('Score Distribution Comparison', fontweight='bold')
    axes[1, 1].set_xticks([1.5, 4.5, 7.5])
    axes[1, 1].set_xticklabels(levels)
    axes[1, 1].legend([bp1["boxes"][0], bp2["boxes"][0]], 
                     ['Baseline', 'Fine-tuned'], loc='lower right')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.suptitle('Detailed Model Comparison Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {output_dir}/05_detailed_comparison.png")


def generate_report(df, stats_results, comparison_results, baseline_df, output_dir):
    """Generate text report"""
    print("\nGenerating analysis report...")
    
    report = []
    report.append("=" * 70)
    report.append("CEFR TEXT SIMPLIFICATION EVALUATION REPORT")
    report.append("=" * 70)
    report.append("")
    
    # Dataset info
    report.append("## DATASET INFORMATION")
    report.append(f"  Total samples: {len(df)}")
    if baseline_df is not None:
        report.append(f"Baseline samples: {len(baseline_df)}")
    report.append("")
    
    # Model comparison
    if comparison_results:
        report.append("## MODEL COMPARISON")
        report.append("")
        for level, metrics in comparison_results.items():
            report.append(f"{level} Level:")
            for metric, values in metrics.items():
                improvement = values['improvement']
                p_val = values['p_value']
                sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else "ns"
                report.append(f"  {metric:25s}: {values['baseline']:.3f} → {values['finetuned']:.3f} "
                            f"({improvement:+.1f}%) [{sig}]")
            report.append("")
    
    # Summary statistics
    report.append("## SUMMARY STATISTICS")
    report.append("")
    
    levels = ['A2', 'B1', 'B2']
    for level in levels:
        report.append(f"{level} Level:")
        
        for metric in ['cosine_score', 'appropriateness_score', 'total_avg', 'llm_score']:
            col = f'{level}_simplified_{metric}'
            if col in df.columns:
                values = df[col].dropna()
                if len(values) > 0:
                    report.append(f"  {metric:25s}: μ={values.mean():.3f}, σ={values.std():.3f}, "
                                f"median={values.median():.3f}")
        
        # Flesch
        flesch_col = f'{level}_simplified_flesch_reading_ease'
        if flesch_col in df.columns:
            expected = CEFR_EXPECTED_FLESCH[level]['center']
            actual = df[flesch_col].mean()
            report.append(f"  Flesch: Expected={expected}, Actual={actual:.1f}, "
                        f"Deviation={abs(actual-expected):.1f}")
        
        report.append("")
    
    # CEFR Expectations
    report.append("## CEFR LEVEL EXPECTATIONS")
    for level, exp in CEFR_EXPECTED_FLESCH.items():
        report.append(f"  {level}: Flesch = {exp['center']} (±{exp['tolerance']})")
    
    # Save report
    with open(f'{output_dir}/analysis_report.txt', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    
    print(f"  Saved: {output_dir}/analysis_report.txt")


def main():
#     """Command line interface"""
    parser = argparse.ArgumentParser(
        description='CEFR Text Simplification Evaluation Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Analyze single model
  python analysis.py eval.csv
  
  # Compare two models
  python analysis.py eval_finetuned.csv -b eval_baseline.csv
  
  # Specify output directory
  python analysis.py eval.csv -o results/
        """
    )
    
    parser.add_argument('csv_path', help='Path to evaluation CSV file')
    parser.add_argument('-b', '--baseline', help='Path to baseline CSV for comparison')
    parser.add_argument('-o', '--output', default='figures', help='Output directory for figures and reports')
    
    args = parser.parse_args()
    
    # Check files exist
    if not os.path.exists(args.csv_path):
        print(f"File not found: {args.csv_path}")
        return
    
    if args.baseline and not os.path.exists(args.baseline):
        print(f"Baseline file not found: {args.baseline}")
        return
    
    # Create output directory
    global OUTPUT_DIR
    OUTPUT_DIR = args.output
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # Run analysis
    run_full_analysis(args.csv_path, args.baseline, OUTPUT_DIR)


if __name__ == '__main__':
    main()