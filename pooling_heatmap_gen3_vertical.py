"""
Author: Dianchao Wang
Date: 2025-12-03
Modified: 2025-12-04
Description:
    Generate two 2D heatmaps { (A-H, 0-15) & (a-c, 0-3) } from CSV file columns based on pooling scores.
    Both heatmaps are displayed with proportional cell sizes and share a common color scale.
Prepparation Steps:
    1. Download the CSV file containing pooling scores from Insight.
    2. Ensure the CSV file is formatted correctly with appropriate column names.
    3. Use JMP to manipulate and clean the data if necessary, e.g. correct column names.
"""
import re
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
    Generate two 2D heatmaps { (A-H, 0-15) & (a-c, 0-3) } from CSV file columns.
    
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

def plot_proportional_heatmaps(heatmap_upper, heatmap_lower, func, title_prefix="Heatmap", cmap='viridis', show_values=False, save_path=None):
    """
    Plot two heatmaps { (A-H, 0-15) & (a-c, 0-3) } with proportional cell sizes and shared color scale.
    
    The cell sizes are proportional to the actual grid dimensions (8x16 vs 3x4).
    Both heatmaps share a common color scale for direct comparison.
    """
    # Create figure with specific aspect ratios
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [16, 4], 'wspace': 0.3})
    
    # Determine global vmin and vmax for shared color scale
    # vmin = min(np.nanmin(heatmap_upper), np.nanmin(heatmap_lower))
    vmax = max(np.nanmax(heatmap_upper), np.nanmax(heatmap_lower))
    vmin = 0
    # vmax = 122.07
    
    # Normalize for consistent colors across both heatmaps
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot upper case heatmap
    ax1 = axes[0]
    sns.heatmap(heatmap_upper, 
                annot=show_values,
                fmt='.2f' if show_values else '',
                cmap=cmap,
                cbar=False,  # We'll add a shared colorbar later
                ax=ax1,
                linewidths=0.5,
                linecolor='gray',
                norm=norm,
                square=True)  # Ensure cells are square
    
    # ax1.set_xlabel('Column (0-15)')
    # ax1.set_ylabel('Row (A-H)')
    ax1.set_title(f'{title_prefix} (A-H, 0-15)')
    ax1.set_xticks(np.arange(16) + 0.5)
    ax1.set_xticklabels([str(i) for i in range(16)])
    ax1.set_yticks(np.arange(8) + 0.5)
    ax1.set_yticklabels([chr(ord('A') + i) for i in range(8)])
    
    # Plot lower case heatmap
    ax2 = axes[1]
    sns.heatmap(heatmap_lower, 
                annot=show_values,
                fmt='.2f' if show_values else '',
                cmap=cmap,
                cbar=False,  # We'll add a shared colorbar later
                ax=ax2,
                linewidths=0.5,
                linecolor='gray',
                norm=norm,
                square=True)  # Ensure cells are square
    
    # ax2.set_xlabel('Column (16-19)')
    # ax2.set_ylabel('Row (a-c)')
    ax2.set_title(f'{title_prefix} (a-c, 16-19)')
    ax2.set_xticks(np.arange(4) + 0.5)
    ax2.set_xticklabels([str(i) for i in range(16, 20)])
    ax2.set_yticks(np.arange(3) + 0.5)
    ax2.set_yticklabels([chr(ord('a') + i) for i in range(3)])
    
    # Adjust y-axis positions to align at the top
    # Get the positions of both axes
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    
    # Calculate the vertical difference in their current positions
    # We want to align them at the top, so we adjust the lower one
    if pos1.height < pos2.height:
        # ax2 is higher, move ax1 up
        new_pos1 = [pos1.x0, pos2.y0 + pos2.height - pos1.height, pos1.width, pos1.height]
        ax1.set_position(new_pos1)
    elif pos2.height < pos1.height:
        # ax1 is higher, move ax2 up
        new_pos2 = [pos2.x0, pos1.y0 + pos1.height - pos2.height, pos2.width, pos2.height]
        ax2.set_position(new_pos2)
    
    # Create color bar
    # Position: [left, bottom, width, height]
    colorbar_left = pos2.x0 - 0.075
    colorbar_bottom = pos1.y0
    colorbar_width = 0.02
    colorbar_height = pos1.height
    
    # Create an axes for colorbar
    cbar_ax = fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height])
    
    # Create a ScalarMappable for the colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Add the colorbar
    cbar = fig.colorbar(sm, cax=cbar_ax)
    # cbar.set_label('Value', rotation=270, labelpad=15)
    
    # Adjust layout
    title = f'{func.capitalize()} of Pooling Scores'
    title += ' (Remote)' if remote else ' (Local)'
    title += ' v3' if not non_v3 else ''
    plt.suptitle(title, fontsize=16, y=0.93)
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for the colorbar
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    else:
        plt.show()
    
    return fig, axes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate proportional heatmaps (upper and lower case) from CSV data with shared color scale.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    parser.add_argument('function', type=str, choices=['sum', 'mean'], help='Aggregation function to use for heatmap generation')
    parser.add_argument('--remote', action='store_true', help='Whether to use remote pooling (default: False)')
    parser.add_argument('--non_v3', action='store_true', help='Whether to use non_v3 format (default: False)')
    parser.add_argument('--save', type=str, help='Path to save the heatmap image (optional)')
    parser.add_argument('--no_values', action='store_true', help='Hide cell values on heatmaps (default: False)')
    
    args = parser.parse_args()
    csv_filename = args.csv_file
    func = args.function
    remote = args.remote
    non_v3 = args.non_v3
    save_path = args.save
    show_values = not args.no_values

    print("\nGenerating proportional heatmaps with shared color scale...")
    print(f"  Function: {func}")
    print(f"  Remote: {remote}")
    print(f"  Non-V3: {non_v3}")
    print(f"  Show values: {show_values}\n")
    
    heatmap_upper, heatmap_lower = generate_heatmaps_from_csv(csv_filename, func, remote, non_v3)
    
    # Choose colormap based on function
    cmap = 'YlOrRd' if func == 'sum' else 'coolwarm'
    
    plot_proportional_heatmaps(heatmap_upper, heatmap_lower, func,
                               title_prefix="",
                               cmap=cmap,
                               show_values=show_values,
                               save_path=save_path)