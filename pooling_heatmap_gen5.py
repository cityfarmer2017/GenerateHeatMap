"""
Author: Dianchao Wang
Date: 2025-12-03
Modified: 2025-12-04
Description:
    Generate two 2D heatmaps (upper case and lower case) from CSV file columns based on pooling scores.
    Both heatmaps are displayed top-aligned with proportional cell sizes and share a common color scale.
    The color bar height matches the upper case heatmap height.
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
from matplotlib.colors import Normalize
from matplotlib.gridspec import GridSpec
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

def plot_top_aligned_heatmaps(heatmap_upper, heatmap_lower, func, title_prefix="Heatmap", cmap='viridis', show_values=False, save_path=None):
    """
    Plot two heatmaps (upper and lower case) top-aligned with proportional cell sizes and shared color scale.
    The color bar height matches the upper case heatmap height.
    
    The cell sizes are proportional to the actual grid dimensions (8x16 vs 3x4).
    Both heatmaps share a common color scale for direct comparison and are top-aligned.
    """
    # Create figure with GridSpec for precise control
    fig = plt.figure(figsize=(12, 8))
    
    # Create GridSpec: 1 row, 2 columns for heatmaps
    # Width ratios: 16 for upper case (8x16), 4 for lower case (3x4)
    gs = GridSpec(1, 2, width_ratios=[16, 4], wspace=0.3)
    
    # Determine global vmin and vmax for shared color scale
    vmin = min(np.nanmin(heatmap_upper), np.nanmin(heatmap_lower))
    vmax = max(np.nanmax(heatmap_upper), np.nanmax(heatmap_lower))
    
    # Normalize for consistent colors across both heatmaps
    norm = Normalize(vmin=vmin, vmax=vmax)
    
    # Plot upper case heatmap (left)
    ax1 = fig.add_subplot(gs[0])
    im1 = sns.heatmap(heatmap_upper, 
                annot=show_values,
                fmt='.2f' if show_values else '',
                cmap=cmap,
                cbar=False,
                ax=ax1,
                linewidths=0.5,
                linecolor='gray',
                norm=norm,
                square=True)
    
    ax1.set_xlabel('Column (0-15)')
    ax1.set_ylabel('Row (A-H)')
    ax1.set_title(f'Upper Case (A-H, 0-15)')
    ax1.set_xticks(np.arange(16) + 0.5)
    ax1.set_xticklabels([str(i) for i in range(16)])
    ax1.set_yticks(np.arange(8) + 0.5)
    ax1.set_yticklabels([chr(ord('A') + i) for i in range(8)])
    
    # Plot lower case heatmap (right)
    ax2 = fig.add_subplot(gs[1])
    im2 = sns.heatmap(heatmap_lower, 
                annot=show_values,
                fmt='.2f' if show_values else '',
                cmap=cmap,
                cbar=False,
                ax=ax2,
                linewidths=0.5,
                linecolor='gray',
                norm=norm,
                square=True)
    
    ax2.set_xlabel('Column (16-19)')
    ax2.set_ylabel('Row (a-c)')
    ax2.set_title(f'Lower Case (a-c, 16-19)')
    ax2.set_xticks(np.arange(4) + 0.5)
    ax2.set_xticklabels([str(i) for i in range(16, 20)])
    ax2.set_yticks(np.arange(3) + 0.5)
    ax2.set_yticklabels([chr(ord('a') + i) for i in range(3)])
    
    # Adjust y-axis positions to align at the top
    pos1 = ax1.get_position()
    pos2 = ax2.get_position()
    
    # Calculate the vertical difference in their current positions
    # We want to align them at the top, so we adjust the lower one
    if pos1.y0 < pos2.y0:
        # ax2 is higher, move ax1 up
        new_pos1 = [pos1.x0, pos2.y0, pos1.width, pos1.height]
        ax1.set_position(new_pos1)
        pos1 = new_pos1  # Update pos1
    elif pos2.y0 < pos1.y0:
        # ax1 is higher, move ax2 up
        new_pos2 = [pos2.x0, pos1.y0, pos2.width, pos2.height]
        ax2.set_position(new_pos2)
        pos2 = new_pos2  # Update pos2
    
    # Create color bar with height matching the upper case heatmap
    # Position: [left, bottom, width, height]
    # We'll place it to the right of both heatmaps
    colorbar_left = pos2.x0 + pos2.width + 0.02  # Right of the second heatmap
    colorbar_bottom = pos1.y0  # Same bottom as upper case heatmap
    colorbar_width = 0.02
    colorbar_height = pos1.height  # Same height as upper case heatmap
    
    # Create color bar axis
    cbar_ax = fig.add_axes([colorbar_left, colorbar_bottom, colorbar_width, colorbar_height])
    
    # Create ScalarMappable for colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Add the colorbar to the figure
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Value', rotation=270, labelpad=15)
    
    # Add overall title
    plt.suptitle(f'{func.capitalize()} of Pooling Scores', fontsize=16, y=0.98)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 0.95, 0.95])  # Make room for the suptitle and colorbar
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    
    plt.show()
    
    return fig, (ax1, ax2)

# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate top-aligned heatmaps (upper and lower case) from CSV data with shared color scale.')
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

    print("\nGenerating top-aligned heatmaps with shared color scale...")
    print(f"  Function: {func}")
    print(f"  Remote: {remote}")
    print(f"  Non-V3: {non_v3}")
    print(f"  Show values: {show_values}")
    
    heatmap_upper, heatmap_lower = generate_heatmaps_from_csv(csv_filename, func, remote, non_v3)
    
    # Choose colormap based on function
    cmap = 'YlOrRd' if func == 'sum' else 'coolwarm'
    
    plot_top_aligned_heatmaps(heatmap_upper, heatmap_lower, func,
                               title_prefix="Pooling Score",
                               cmap=cmap,
                               show_values=show_values,
                               save_path=save_path)