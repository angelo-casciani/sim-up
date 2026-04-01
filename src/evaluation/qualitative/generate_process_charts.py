"""Generate visualization charts for qualitative evaluation."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

def generate_charts():
    """Generate all charts for the qualitative evaluation."""

    # Create output directory
    output_dir = Path('reports/qualitative/charts')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read processed log
    df = pd.read_csv('logs/o2c/745347473/processed_745347473.csv')

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams['figure.dpi'] = 150
    plt.rcParams['font.size'] = 10

    # Color scheme for roles
    role_colors = {
        'BillingClerk': '#3498db',
        'WarehouseStaff': '#2ecc71',
        'SalesRep': '#e74c3c'
    }

    # ============================================
    # CHART 1: Activity Distribution (Horizontal Bar)
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 6))

    activity_counts = df['activity_name'].value_counts()
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(activity_counts)))[::-1]

    bars = ax.barh(range(len(activity_counts)), activity_counts.values, color=colors)
    ax.set_yticks(range(len(activity_counts)))
    ax.set_yticklabels(activity_counts.index, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Number of Events', fontsize=11)
    ax.set_title('Activity Frequency Distribution', fontsize=13, fontweight='bold')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, activity_counts.values)):
        ax.text(val + 50, bar.get_y() + bar.get_height()/2, f'{val:,}',
                va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'chart1_activity_distribution.png', bbox_inches='tight')
    plt.close()
    print('Created: chart1_activity_distribution.png')

    # ============================================
    # CHART 2: Resource Workload by Role (Grouped)
    # ============================================
    fig, ax = plt.subplots(figsize=(12, 5))

    resource_events = df.groupby('resource').size().sort_values(ascending=False)

    # Color by role
    colors = [role_colors.get(r.rsplit('_', 1)[0], '#95a5a6') for r in resource_events.index]

    bars = ax.bar(range(len(resource_events)), resource_events.values, color=colors)
    ax.set_xticks(range(len(resource_events)))
    ax.set_xticklabels(resource_events.index, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Number of Events', fontsize=11)
    ax.set_title('Workload Distribution Across Resources', fontsize=13, fontweight='bold')

    # Add legend
    legend_patches = [mpatches.Patch(color=c, label=r) for r, c in role_colors.items()]
    ax.legend(handles=legend_patches, loc='upper right')

    # Add average line
    avg_events = resource_events.mean()
    ax.axhline(y=avg_events, color='red', linestyle='--', linewidth=1.5)
    ax.text(len(resource_events)-1, avg_events + 30, f'Avg: {avg_events:.0f}', color='red', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'chart2_resource_workload.png', bbox_inches='tight')
    plt.close()
    print('Created: chart2_resource_workload.png')

    # ============================================
    # CHART 3: Role Workload Share (Pie Chart)
    # ============================================
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate events per role
    role_events = {}
    for resource in df['resource']:
        role = resource.rsplit('_', 1)[0]
        role_events[role] = role_events.get(role, 0) + 1

    roles = list(role_events.keys())
    events = list(role_events.values())
    colors = [role_colors.get(r, '#95a5a6') for r in roles]

    # Create pie chart
    wedges, texts, autotexts = ax.pie(events, labels=roles, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       explode=[0.02]*len(roles))

    # Style
    for autotext in autotexts:
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax.set_title('Event Distribution by Role', fontsize=13, fontweight='bold')

    # Add counts in legend
    legend_labels = [f'{r}: {role_events[r]:,} events' for r in roles]
    ax.legend(wedges, legend_labels, loc='lower right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'chart3_role_distribution.png', bbox_inches='tight')
    plt.close()
    print('Created: chart3_role_distribution.png')

    # ============================================
    # CHART 4: Case Length Distribution (Histogram)
    # ============================================
    fig, ax = plt.subplots(figsize=(10, 5))

    case_lengths = df.groupby('case_id').size()

    # Create histogram
    bins = range(1, min(case_lengths.max() + 2, 16))  # Cap at 15 for readability
    counts, edges, patches = ax.hist(case_lengths, bins=bins, color='#3498db',
                                      edgecolor='white', linewidth=1)

    ax.set_xlabel('Number of Events per Case', fontsize=11)
    ax.set_ylabel('Number of Cases', fontsize=11)
    ax.set_title('Case Complexity Distribution', fontsize=13, fontweight='bold')
    ax.set_xticks(range(1, 15))

    # Add statistics box
    stats_text = f'Median: {case_lengths.median():.0f} events\nMean: {case_lengths.mean():.1f} events'
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'chart4_case_length_distribution.png', bbox_inches='tight')
    plt.close()
    print('Created: chart4_case_length_distribution.png')

    print(f'\nAll charts saved to: {output_dir}')


if __name__ == '__main__':
    generate_charts()
