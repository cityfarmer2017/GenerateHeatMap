"""
Author: Dianchao Wang
Date: 2025-12-03
Modified: 2025-12-04
Description:
    Generate two 2D heatmaps (upper case and lower case) from CSV file columns based on pooling scores.
Prepparation Steps:
    1. Download the CSV file containing pooling scores from Insight.
    2. Ensure the CSV file is formatted correctly with appropriate column names.
    3. Use JMP to manipulate and clean the data if necessary, e.g. correct column names.
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')

def extract_coordinates(column_name, location_pattern):
    """Extract coordinates from column name using regex pattern."""
    match = re.search(location_pattern, column_name)
    if match:
        coord_str = match.group(1)
        # Determine if it's upper or lower case
        if coord_str[0].isupper():
            row = ord(coord_str[0]) - ord('A')
            col = int(coord_str.split('_')[1])
            return row, col, 'upper'
        else:
            row = ord(coord_str[0]) - ord('a')
            col = int(coord_str.split('_')[1]) - 16  # Adjust for lower case (16-19)
            return row, col, 'lower'
    return None

def generate_heatmaps_from_csv(csv_file: str, func: str, remote: bool, non_v3: bool):
    """
    Generate two 2D heatmaps (upper and lower case) from CSV file columns.
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
    func : str
        Function to aggregate values ('sum', 'mean')
    remote : bool
        Whether to use remote pooling scores
    non_v3 : bool
        Whether to use pooling score none v3 format
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize heatmap grids
    heatmap_upper = np.zeros((8, 16))
    heatmap_lower = np.zeros((3, 4))
    count_map_upper = np.zeros((8, 16))
    count_map_lower = np.zeros((3, 4))
    
    # Define location patterns for upper and lower case
    location_pattern_upper = r"([A-H]_\d{1,2})"
    location_pattern_lower = r"([abc]_\d{1,2})"
    location_pattern = rf"({location_pattern_upper}|{location_pattern_lower})"
    
    # Process each column
    for column in df.columns:
        # Check if column matches the pooling type and version
        if not remote and not non_v3:
            if not re.search(rf'.+{location_pattern}_PoolingScoreLocal_v3$', column):
                continue
        elif not remote and non_v3:
            if not re.search(rf'.+{location_pattern}_PoolingScoreLocal$', column):
                continue
        elif remote and not non_v3:
            if not re.search(rf'.+{location_pattern}_PoolingScoreRemote_v3$', column):
                continue
        else:
            if not re.search(rf'.+{location_pattern}_PoolingScoreRemote$', column):
                continue
            
        coords = extract_coordinates(column, location_pattern)
        if coords:
            row_idx, col_idx, case_type = coords
            
            # Get column values (handle non-numeric values)
            values = pd.to_numeric(df[column], errors='coerce').fillna(0)
            
            # Accumulate values based on case type
            if case_type == 'upper':
                if 0 <= row_idx < 8 and 0 <= col_idx < 16:
                    if func == 'sum':
                        heatmap_upper[row_idx, col_idx] += values.sum()
                    elif func == 'mean':
                        heatmap_upper[row_idx, col_idx] += values.sum()
                        count_map_upper[row_idx, col_idx] += values.count()
            else:  # lower case
                if 0 <= row_idx < 3 and 0 <= col_idx < 4:
                    if func == 'sum':
                        heatmap_lower[row_idx, col_idx] += values.sum()
                    elif func == 'mean':
                        heatmap_lower[row_idx, col_idx] += values.sum()
                        count_map_lower[row_idx, col_idx] += values.count()
    
    # Calculate mean if needed
    if func == 'mean':
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap_upper = np.divide(heatmap_upper, count_map_upper)
            heatmap_upper = np.nan_to_num(heatmap_upper)
            heatmap_lower = np.divide(heatmap_lower, count_map_lower)
            heatmap_lower = np.nan_to_num(heatmap_lower)
    
    return heatmap_upper, heatmap_lower

def plot_dual_heatmaps(heatmap_upper, heatmap_lower, func, title_prefix="Heatmap", cmap='viridis', show_values=False, save_path=None):
    """
    Plot two heatmaps (upper and lower case) side by side.
    """
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot upper case heatmap
    sns.heatmap(heatmap_upper, 
                annot=show_values,
                fmt='.2f' if show_values else '',
                cmap=cmap,
                cbar_kws={'label': 'Value'},
                ax=ax1,
                linewidths=0.5,
                linecolor='gray')
    
    ax1.set_xlabel('Column (0-15)')
    ax1.set_ylabel('Row (A-H)')
    ax1.set_title(f'{title_prefix} - (A-H, 0-15)')
    ax1.set_xticks(np.arange(16) + 0.5)
    ax1.set_xticklabels([str(i) for i in range(16)])
    ax1.set_yticks(np.arange(8) + 0.5)
    ax1.set_yticklabels([chr(ord('A') + i) for i in range(8)])
    
    # Plot lower case heatmap
    sns.heatmap(heatmap_lower, 
                annot=show_values,
                fmt='.2f' if show_values else '',
                cmap=cmap,
                cbar_kws={'label': 'Value'},
                ax=ax2,
                linewidths=0.5,
                linecolor='gray')
    
    ax2.set_xlabel('Column (16-19)')
    ax2.set_ylabel('Row (a-c)')
    ax2.set_title(f'{title_prefix} - (a-c, 16-19)')
    ax2.set_xticks(np.arange(4) + 0.5)
    ax2.set_xticklabels([str(i) for i in range(16, 20)])
    ax2.set_yticks(np.arange(3) + 0.5)
    ax2.set_yticklabels([chr(ord('a') + i) for i in range(3)])
    
    plt.suptitle(f'{func.capitalize()} of Values', fontsize=16, y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, (ax1, ax2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate dual heatmaps (upper and lower case) from CSV data.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    parser.add_argument('function', type=str, choices=['sum', 'mean'], help='Aggregation function to use for heatmap generation')
    parser.add_argument('--remote', action='store_true', help='Whether to use remote pooling (default: False)')
    parser.add_argument('--non_v3', action='store_true', help='Whether to use non_v3 format (default: False)')
    parser.add_argument('--save', type=str, help='Path to save the heatmap image (optional)')
    
    args = parser.parse_args()
    csv_filename = args.csv_file
    func = args.function
    remote = args.remote
    non_v3 = args.non_v3
    save_path = args.save

    print("\nGenerating dual heatmaps (upper and lower case)...")
    
    heatmap_upper, heatmap_lower = generate_heatmaps_from_csv(csv_filename, func, remote, non_v3)
    
    # Choose colormap based on function
    cmap = 'YlOrRd' if func == 'sum' else 'coolwarm'
    
    plot_dual_heatmaps(heatmap_upper, heatmap_lower, func,
                       title_prefix="Pooling Score",
                       cmap=cmap,
                       show_values=True,
                       save_path=save_path)