"""
Author: Dianchao Wang
Date: 2025-12-03
Description:
    Generate a 2D heatmap from CSV file columns based on pooling scores.
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
# from collections import defaultdict
import argparse
import warnings
warnings.filterwarnings('ignore')

def extract_coordinates(column_name, low_case):
    """Extract coordinates from column name using regex pattern."""
    pattern = rf".+{location}.+"
    match = re.search(pattern, column_name)
    if match:
        coord_str = match.group(1)
        row = ord(coord_str[0]) - ord('A' if not low_case else 'a')  # Convert A-H to 0-7 or a-c to 0-2
        col = int(coord_str.split('_')[1])  # Convert column # to integer
        col = col if not low_case else col - 15  # Adjust for low_case or upper_case
        return row, col
    return None

def generate_heatmap_from_csv(csv_file: str, func: str, remote: bool, non_v3: bool, low_case: bool):
    """
    Generate a 2D heatmap from CSV file columns.
    
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
    low_case : bool
        Whether to handle low_case letters of press location in column names
    """
    # Read CSV file
    df = pd.read_csv(csv_file)
    
    # Initialize heatmap grid (8 rows A-H, 16 columns 0-15) or (3 rows a-c, 4 columns 16-19)
    heatmap = np.zeros((8, 16)) if not low_case else np.zeros((3, 4))
    count_map = np.zeros((8, 16)) if not low_case else np.zeros((3, 4))  # For averaging
    
    # Process each column
    for column in df.columns:
        # Check if column matches the pooling type and version
        if not remote and not non_v3:
            if not re.search(rf'.+{location}_PoolingScoreLocal_v3$', column):
                continue
        elif not remote and non_v3:
            if not re.search(rf'.+{location}_PoolingScoreLocal$', column):
                continue
        elif remote and not non_v3:
            if not re.search(rf'.+{location}_PoolingScoreRemote_v3$', column):
                continue
        else:
            if not re.search(rf'.+{location}_PoolingScoreRemote$', column):
                continue

        # print(f"Processing column: {column}")
            
        coords = extract_coordinates(column, low_case)
        if coords:
            row_idx, col_idx = coords
            
            # Check if coordinates are within bounds
            if (not low_case and (0 <= row_idx < 8 and 0 <= col_idx < 16)) or (low_case and (0 <= row_idx < 3 and 0 <= col_idx < 4)):
                # Get column values (handle non-numeric values)
                values = pd.to_numeric(df[column], errors='coerce').fillna(0)
                # print(F"Column sum preview: {values.sum()}")
                # print(F"Column count preview: {values.count()}")
                
                # Accumulate values
                if func == 'sum':
                    heatmap[row_idx, col_idx] += values.sum()
                elif func == 'mean':
                    heatmap[row_idx, col_idx] += values.sum()
                    count_map[row_idx, col_idx] += values.count()
                # elif func == 'max':
                #     current_max = heatmap[row_idx, col_idx]
                #     new_max = values.max()
                #     if new_max > current_max or (row_idx == 0 and col_idx == 0 and current_max == 0):
                #         heatmap[row_idx, col_idx] = new_max
                # elif func == 'min':
                #     current_min = heatmap[row_idx, col_idx]
                #     new_min = values.min()
                #     if new_min < current_min or (row_idx == 0 and col_idx == 0 and current_min == 0):
                #         heatmap[row_idx, col_idx] = new_min
    
    # Calculate mean if needed
    if func == 'mean':
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = np.divide(heatmap, count_map)
            heatmap = np.nan_to_num(heatmap)
    
    return heatmap

def plot_heatmap(heatmap_data, cols, rows, title="Heatmap", cmap='viridis', show_values=False, save_path=None):
    """
    Plot the heatmap with proper labels.
    """
    fig, ax = plt.subplots(figsize=(12, 6)) if cols == 16 else plt.subplots(figsize=(3, 2.25))
    
    # Create heatmap
    sns.heatmap(heatmap_data, 
                annot=show_values,
                fmt='.2f' if show_values else '',
                cmap=cmap,
                cbar_kws={'label': 'Value'},
                ax=ax,
                linewidths=0.5,
                linecolor='gray')
    
    # Set labels
    ax.set_xlabel('Column ' + '(0-15)' if cols == 16 else f'(16-{cols+15})')
    ax.set_ylabel('Row (A-H)' if rows == 8 else 'Row (a-c)')
    ax.set_title(title)
    
    # Set proper ticks
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels([str(i) for i in (range(cols) if cols == 16 else range(16, 16 + cols))])
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels([chr(ord('A' if rows == 8 else 'a') + i) for i in range(rows)])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate heatmap from CSV data.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    parser.add_argument('function', type=str, choices=['sum', 'mean'], help='Aggregation function to use for heatmap generation')
    parser.add_argument('--remote', action='store_true', help='Whether to use remote pooling (default: True)')
    parser.add_argument('--non_v3', action='store_true', help='Whether to use non_v3 format (default: True)')
    parser.add_argument('--low_case', action='store_true', help='Whether to handle low_case letters in column names (default: True)')
    
    args = parser.parse_args()
    csv_filename = args.csv_file
    func = args.function
    remote = args.remote
    non_v3 = args.non_v3
    low_case = args.low_case

    print("\nGenerating heatmaps...")

    location = "([A-H]_\d{1,2})" if not low_case else "([abc]_\d{1,2})"
    
    heatmap = generate_heatmap_from_csv(csv_filename, func, remote, non_v3, low_case)
    plot_heatmap(heatmap, 16 if not low_case else 4, 8 if not low_case else 3,
                 title="Heatmap - " + func +" of Values", 
                 cmap='YlOrRd' if not func == 'mean' else 'coolwarm',
                 show_values=True)
