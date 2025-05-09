import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib.patches import Rectangle

# Set font configuration to support Chinese
matplotlib.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
# Fix minus sign display issue
matplotlib.rcParams['axes.unicode_minus'] = False  

def draw_refined_command_table():
    # Create figure and axis with appropriate size
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('off')
    
    # Define header layout and styling
    header_labels = ['命令类型', '示例命令', '噪音水平', '距语音传感器距离', 'WER(%)', '平均延时']
    col_widths = [0.1, 0.4, 0.15, 0.15, 0.1, 0.1]  # Column width proportions
    
    # Define data - keeping the same data structure
    data = [
        ['短语命令', 
         '抓住物体，\n放下物体，\n抬起手臂，\n旋转90度，\n前进，\n后退，\n停止等', 
         '安静（30-40dB）', 
         '50cm', 
         '6.02', 
         '894ms'],
        
        ['短语命令', 
         '抓住物体，\n放下物体，\n抬起手臂，\n旋转90度，\n前进，\n后退，\n停止等', 
         '中等（50-60dB）', 
         '50cm', 
         '8.73', 
         '903ms'],
        
        ['串行命令', 
         '抓住物体，然后放到托盘上；\n抓取物体后，转向左边；\n先打开夹爪，再抓取物体等', 
         '安静（30-40dB）', 
         '50cm', 
         '5.43', 
         '843ms'],
        
        ['串行命令', 
         '抓住物体，然后放到托盘上；\n抓取物体后，转向左边；\n先打开夹爪，再抓取物体等', 
         '中等（50-60dB）', 
         '50cm', 
         '7.89', 
         '923ms']
    ]
    
    # Create a custom table layout
    # First, add the main title
    title_y = 0.9
    ax.text(0.5, title_y, '语音命令性能数据表', fontsize=18, ha='center', va='center', fontweight='bold')
    
    # Add "示例命令" subtitle at the appropriate position
    # Calculate the x position for the 示例命令 column
    x_start = col_widths[0]
    x_center = x_start + col_widths[1]/2
    # ax.text(x_center, 0.85, '示例命令', fontsize=12, ha='center', va='center')
    
    # Draw the table borders and cells
    y_top = 0.8  # Starting y-position for the table
    y_step = 0.15  # Height of each row
    x_positions = [0]  # Starting x positions for each column
    
    # Calculate x positions for all columns
    total_width = sum(col_widths)
    for width in col_widths:
        x_positions.append(x_positions[-1] + width/total_width)
    
    # Draw outer frame
    rect = Rectangle((0, 0.05), 1, y_top, fill=False, edgecolor='black', linewidth=1.5)
    ax.add_patch(rect)
    
    # Draw header row
    y_header = y_top - y_step
    for i in range(len(x_positions)-1):
        # Draw header cell
        rect = Rectangle((x_positions[i], y_header), 
                         x_positions[i+1]-x_positions[i], y_step, 
                         fill=False, edgecolor='black', linewidth=1.2)
        ax.add_patch(rect)
        
        # Add header text
        ax.text((x_positions[i]+x_positions[i+1])/2, 
                y_header + y_step/2, 
                header_labels[i], 
                fontsize=11, ha='center', va='center', fontweight='bold')
    
    # Draw data rows
    for row_idx, row_data in enumerate(data):
        y_row = y_header - (row_idx+1) * y_step
        
        # Draw command type cell (first column)
        rect = Rectangle((x_positions[0], y_row), 
                        x_positions[1]-x_positions[0], y_step, 
                        fill=False, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text((x_positions[0]+x_positions[1])/2, 
                y_row + y_step/2, 
                row_data[0], 
                fontsize=10, ha='center', va='center')
        
        # Draw example command cell (second column)
        rect = Rectangle((x_positions[1], y_row), 
                        x_positions[2]-x_positions[1], y_step, 
                        fill=False, edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text((x_positions[1]+x_positions[2])/2, 
                y_row + y_step/2, 
                row_data[1], 
                fontsize=10, ha='left', va='center', linespacing=1.3)
        
        # Draw remaining data cells
        for i in range(2, len(row_data)):
            rect = Rectangle((x_positions[i], y_row), 
                            x_positions[i+1]-x_positions[i], y_step, 
                            fill=False, edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            ax.text((x_positions[i]+x_positions[i+1])/2, 
                    y_row + y_step/2, 
                    row_data[i], 
                    fontsize=10, ha='center', va='center')
    
    # Set the aspect ratio to be equal
    plt.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.02)
    
    # Save and show
    plt.savefig('refined_voice_command_table.png', dpi=300, bbox_inches='tight')
    plt.show()

# Call the function
draw_refined_command_table()