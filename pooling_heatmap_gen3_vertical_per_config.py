"""
Author: Dianchao Wang
Date: 2025-12-16
Modified: 2025-12-18
Description:
    Generate two 2D heatmaps { (A-H, 0-15) & (a-c, 0-3) } from CSV file columns based on pooling scores.
    Both heatmaps are displayed with proportional cell sizes and share a common color scale.
Prepparation Steps:
    1. Download the CSV file containing pooling scores from Insight.
    2. Ensure the CSV file is formatted correctly with appropriate column names.
    3. Use JMP to manipulate and clean the data if necessary, e.g. correct column names.
"""

import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import numpy as np
import argparse
import warnings
warnings.filterwarnings('ignore')


CONFIG_COLUMN = 'Special Build Description'


def extract_coordinates(column_name, location_pattern):
    """Extract coordinates from column name using regex pattern."""
    match = re.search(location_pattern, column_name, re.IGNORECASE)
    if match:
        row_char = match.group(1)
        col_str = match.group(2)
        # Determine if it's upper or lower case
        if row_char.isupper():
            row = ord(row_char) - ord('A')
            col = int(col_str)
            return row, col, 'upper'
        else:
            row = ord(row_char) - ord('a')
            col = int(col_str) - 16  # Adjust for lower case (16-19 -> 0-3)
            return row, col, 'lower'
    return None


def generate_heatmaps_from_csv_per_config(csv_file: str, func: str, remote: bool, non_v3: bool, config_filter: list = None):
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
    config_filter : list, optional
        List of special configurations to filter on (default: None, meaning no filtering, include all)
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"CSV file not found: {csv_file}")
        return {}
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return {}

    if CONFIG_COLUMN not in df.columns:
        print(f"Column '{CONFIG_COLUMN}' not found in CSV file.")
        exit(1)

    all_configs = df[CONFIG_COLUMN].unique()
    print(f"Found {len(all_configs)} unique configurations in the CSV file.")

    if config_filter:
        configs = [config for config in all_configs if config in config_filter]
        print(f"Filtered to {len(configs)} configurations based on provided filter.")
    else:
        print("No configuration filter provided; processing all configurations.")
        configs = all_configs

    print("Configurations to process:")
    for config in configs[:10]:
        print(f"  - {config}")
    if len(configs) > 10:
        print(f"  ... and {len(configs) - 10} more")
    print()
    
    # Define location patterns for upper and lower case of rows
    location_pattern_upper = r"([A-H])_(\d{1,2})"
    location_pattern_lower = r"([a-c])_(\d{1,2})"
    location_pattern = rf"{location_pattern_upper}|{location_pattern_lower}"

    # Create regex pattern per pooling type and version
    if remote:
        suffix = "_PoolingScoreRemote" if non_v3 else "_PoolingScoreRemote_v3"
    else:
        suffix = "_PoolingScoreLocal" if non_v3 else "_PoolingScoreLocal_v3"

    location_pattern = rf".*{location_pattern}{suffix}$"

    # Find all matching columns once (they're the same for all configurations)
    all_columns = df.columns.tolist()
    matching_columns = [col for col in all_columns if re.search(location_pattern, col)]
    if not matching_columns:
        print(f"No columns found matching pattern '{location_pattern}'.")
        return {}
    else:
        print(f"Found {len(matching_columns)} columns matching pattern '{location_pattern}'.")

    heatmaps = {}

    for config in configs:
        print(f"Processing configuration: {config}")
        df_config = df[df[CONFIG_COLUMN] == config].copy()
        if df_config.empty:
            print(f"  No data found for configuration '{config}'. Skipping.")
            continue

        count = len(df_config)
        print(f"  Number of rows for this configuration: {count}")
        
        # Initialize heatmap grids for this configuration
        heatmap_upper = np.zeros((8, 16))
        heatmap_lower = np.zeros((3, 4))
        count_map_upper = np.zeros((8, 16))
        count_map_lower = np.zeros((3, 4))
        
        # Process each matching column
        for column in matching_columns:
            coords = extract_coordinates(column, location_pattern)
            if not coords:
                continue

            row_idx, col_idx, case_type = coords
            
            # Get column values (handle non-numeric values)
            values = pd.to_numeric(df_config[column], errors='coerce').fillna(0)
            
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
        
        heatmaps[config] = (heatmap_upper, heatmap_lower, count)
    
    return heatmaps


def plot_proportional_heatmaps(heatmaps, func, cmap, title_prefix, show_values, save_path, remote, non_v3):
    """
    Plot two heatmaps { (A-H, 0-15) & (a-c, 0-3) } with proportional cell sizes and shared color scale.
    
    The cell sizes are proportional to the actual grid dimensions (8x16 vs 3x4).
    Both heatmaps share a common color scale for direct comparison.
    """
    # Determine global vmin and vmax for shared color scale
    # vmin = min(np.nanmin(heatmap_upper), np.nanmin(heatmap_lower))
    # vmax = max(np.nanmax(heatmap_upper), np.nanmax(heatmap_lower))
    vmin = 0
    vmax = 122.07
    
    # Normalize for consistent colors across both heatmaps
    norm = Normalize(vmin=vmin, vmax=vmax)

    # Create dir of save_path
    save_dir = os.path.dirname(os.path.abspath(__file__))
    save_path += "_remote" if remote else "_local"
    save_path += "_v3" if not non_v3 else ""
    save_dir = os.path.join(save_dir, save_path)
    os.makedirs(save_dir, exist_ok=True)

    for config_name, (heatmap_upper, heatmap_lower, count) in heatmaps.items():
        # Make the title shorter by removing prefix
        title = config_name[len(title_prefix):]

        # Create figure with specific aspect ratios
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [16, 4], 'wspace': 0.3})

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
        ax1.set_title(f'{title} ({count}-DUTs) (A-H, 0-15)')
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
        # ax2.set_title(f'{title} (a-c, 16-19)')
        ax2.set_title('(a-c, 16-19)')
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
        title_str = f'{func.capitalize()} of Pooling Scores'
        title_str += ' (Remote)' if remote else ' (Local)'
        title_str += ' v3' if not non_v3 else ''
        plt.suptitle(title_str, fontsize=16, y=0.93)
        plt.tight_layout(rect=[0, 0, 0.9, 1])  # Make room for the colorbar

        file_name = os.path.join(save_dir, f"{title}_heatmap.png")

        plt.savefig(file_name, dpi=300, bbox_inches='tight', pad_inches=0.1)
        plt.close(fig)
        
        print(f"Saved heatmap for configuration '{config_name}' to '{file_name}'")


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate proportional heatmaps (upper and lower case) from CSV data with shared color scale.')
    parser.add_argument('csv_file', type=str, help='Path to the input CSV file')
    parser.add_argument('function', type=str, choices=['sum', 'mean'], help='Aggregation function to use for heatmap generation')
    parser.add_argument('--remote', action='store_true', help='Whether to use remote pooling (default: False)')
    parser.add_argument('--non_v3', action='store_true', help='Whether to use non_v3 format (default: False)')
    parser.add_argument('--save_dir', type=str, default='heatmaps', help='Path to save the heatmap images (default: heatmaps)')
    parser.add_argument('--no_values', action='store_true', help='Hide cell values on heatmaps (default: False)')
    parser.add_argument('--configs', type=str, nargs='*', help='List of special configurations to filter on (optional)')
    parser.add_argument('--title_prefix', type=str, default='J91a-Drop_Drop-', help='Prefix to remove from config names in titles (default: J91a-Drop_Drop-)')
    
    args = parser.parse_args()
    csv_filename = args.csv_file
    func = args.function
    remote = args.remote
    non_v3 = args.non_v3
    save_path = args.save_dir
    show_values = not args.no_values
    config_filter = args.configs if args.configs else None
    title_prefix = args.title_prefix

    print("\nGenerating proportional heatmaps with shared color scale...")
    print(f"  CSV file: {csv_filename}")
    print(f"  Function: {func}")
    print(f"  Remote: {remote}")
    print(f"  Non-V3: {non_v3}")
    print(f"  Save path: {save_path if save_path else 'Default (script directory)'}")
    print(f"  Show values: {show_values}")
    print(f"  Config filter: {config_filter if config_filter else 'None (all configs)'}")
    print(f"  Title prefix to remove: '{title_prefix}'\n")
    
    heatmaps = generate_heatmaps_from_csv_per_config(csv_filename, func, remote, non_v3, config_filter)
    if not heatmaps:
        print("No heatmaps generated. Exiting.")
        exit(1)
    
    # Choose colormap based on function
    cmap = 'YlOrRd' if func == 'sum' else 'coolwarm'
    
    plot_proportional_heatmaps(heatmaps, func, cmap, title_prefix, show_values, save_path, remote, non_v3)

    print(f"\nGenerated {len(heatmaps)} heatmap(s) successfully!")
    

if __name__ == "__main__":
    main()