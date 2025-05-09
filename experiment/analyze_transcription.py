import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def analyze_transcription_times():
    # Path to the CSV file
    csv_file = Path("../essay/Data/transcription_results.csv")
    
    # Check if the file exists
    if not csv_file.exists():
        print(f"CSV file not found: {csv_file}")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"成功加载数据，共 {len(df)} 个条目")
    except Exception as e:
        print(f"读取CSV文件错误: {e}")
        return
    
    # Calculate average response time
    avg_response_time = df['response_time'].mean()
    print(f"平均响应时间: {avg_response_time:.3f} 秒")
    
    # Create a directory for saving the figure if it doesn't exist
    output_dir = Path("../essay/Data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set font to support Chinese characters
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # Properly display minus sign
    
    # Create a figure for response times
    plt.figure(figsize=(12, 6))
    
    # Sort by file name for better visualization
    df = df.sort_values('file_name')
    
    # Create new x labels in Chinese
    command_labels = [f"指令 {i+1}" for i in range(len(df))]
    
    # Create a mapping of command labels to file names for the legend
    file_mapping = {label: filename for label, filename in zip(command_labels, df['file_name'])}
    
    # Create bar chart of response times with new labels
    bars = plt.bar(command_labels, df['response_time'], color='skyblue')
    
    # Add a horizontal line for the average
    plt.axhline(y=avg_response_time, color='red', linestyle='--', 
                label=f'平均值: {avg_response_time:.3f}秒')
    
    # Add labels and title in Chinese
    plt.xlabel('指令')
    plt.ylabel('响应时间（秒）')
    plt.title('OpenAI Whisper API 指令响应时间')
    plt.xticks(rotation=45)  # Less rotation needed for shorter labels
    plt.tight_layout()  # Adjust layout to make room for rotated labels
    
    # Add legend with file mapping
    plt.legend([f"平均值: {avg_response_time:.3f}秒"], loc='upper right')
    
    # Add values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}秒', ha='center', va='bottom', rotation=0)
    
    # Save the figure
    output_file = output_dir / "response_time_analysis_chinese.png"
    plt.savefig(output_file)
    print(f"图表已保存至: {output_file}")
    
    # Create and save a mapping file so we know which command corresponds to which file
    mapping_file = output_dir / "command_file_mapping.txt"
    with open(mapping_file, 'w', encoding='utf-8') as f:
        f.write("指令与文件名映射\n")
        f.write("="*30 + "\n")
        for label, filename in file_mapping.items():
            f.write(f"{label}: {filename}\n")
    
    print(f"指令映射已保存至: {mapping_file}")
    
    # Display the figure (if running in an environment with a display)
    plt.show()
    
    # Additional statistics
    min_time = df['response_time'].min()
    max_time = df['response_time'].max()
    median_time = df['response_time'].median()
    std_dev = df['response_time'].std()
    
    print("\n响应时间统计:")
    print(f"最小值: {min_time:.3f} 秒")
    print(f"最大值: {max_time:.3f} 秒")
    print(f"中位数: {median_time:.3f} 秒")
    print(f"标准差: {std_dev:.3f} 秒")
    
    # Save statistics to a text file
    stats_file = output_dir / "response_time_stats_chinese.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("OpenAI Whisper 转录响应时间统计\n")
        f.write("="*60 + "\n")
        f.write(f"处理文件数量: {len(df)}\n")
        f.write(f"平均响应时间: {avg_response_time:.3f} 秒\n")
        f.write(f"最小响应时间: {min_time:.3f} 秒\n")
        f.write(f"最大响应时间: {max_time:.3f} 秒\n")
        f.write(f"中位数响应时间: {median_time:.3f} 秒\n")
        f.write(f"标准差: {std_dev:.3f} 秒\n")
    
    print(f"统计数据已保存至: {stats_file}")

if __name__ == "__main__":
    analyze_transcription_times() 