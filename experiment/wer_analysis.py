import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import jieba
from jiwer import wer

def calculate_wer():
    # Path to the CSV file
    csv_file = Path("../essay/Data/transcription_results.csv")
    
    # Check if the file exists
    if not csv_file.exists():
        print(f"CSV file not found: {csv_file}")
        return
    
    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(f"Successfully loaded data with {len(df)} entries")
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Correct transcriptions provided by the user
    correct_transcriptions = [
        "小车后退",
        "机械臂停止",
        "机械臂复位",
        "机械臂放下",
        "小车暂停",
        "机械臂复位",
        "机械臂旋转到左侧",
        "小车继续",
        "机械臂夹具打开",
        "机械臂松开",
        "小车左转",
        "小车前进",
        "机械臂夹具关闭",
        "机械臂抓取",
        "小车右转",
        "小车停止",
        "机械臂抬起",
        "机械臂旋转到右侧",
        "小车加速",
        "小车减速"
    ]
    
    # Check if we have the correct number of reference transcriptions
    if len(df) > len(correct_transcriptions):
        print(f"Warning: More transcriptions in CSV ({len(df)}) than reference transcriptions ({len(correct_transcriptions)})")
        print("Will only calculate WER for the first", len(correct_transcriptions), "entries")
        df = df.iloc[:len(correct_transcriptions)]
    elif len(df) < len(correct_transcriptions):
        print(f"Warning: Fewer transcriptions in CSV ({len(df)}) than reference transcriptions ({len(correct_transcriptions)})")
        print("Will only use the first", len(df), "reference transcriptions")
        correct_transcriptions = correct_transcriptions[:len(df)]
    
    # Add the correct transcriptions to the dataframe
    df['correct_transcription'] = correct_transcriptions
    
    # Calculate WER for each transcription
    wer_scores = []
    for i, row in df.iterrows():
        # Use jieba for word segmentation in Chinese
        reference_words = list(jieba.cut(row['correct_transcription']))
        hypothesis_words = list(jieba.cut(row['transcription']))
        
        # Calculate WER using jiwer
        error_rate = wer(' '.join(reference_words), ' '.join(hypothesis_words))
        wer_scores.append(error_rate)
    
    # Add WER to the dataframe
    df['wer'] = wer_scores
    
    # Calculate average WER
    avg_wer = np.mean(wer_scores)
    print(f"Average Word Error Rate: {avg_wer:.3f}")
    
    # Create a directory for saving the figure if it doesn't exist
    output_dir = Path("../essay/Data/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a figure for WER visualization
    plt.figure(figsize=(14, 7))
    
    # Set font to support Chinese characters
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False  # Properly display minus sign
    
    # Create command labels in Chinese
    command_labels = [f"指令 {i+1}" for i in range(len(df))]
    
    # Create bar chart of WER scores
    bars = plt.bar(command_labels, df['wer'], color='lightcoral')
    
    # Add a horizontal line for the average WER
    plt.axhline(y=avg_wer, color='red', linestyle='--', 
                label=f'平均错误率: {avg_wer:.3f}')
    
    # Add labels and title in Chinese
    plt.xlabel('指令')
    plt.ylabel('词错误率 (WER)')
    plt.title('中文指令转录的词错误率分析')
    plt.xticks(rotation=45)
    
    # Increase y-axis height by setting a higher upper limit
    # Increase the upper limit more significantly
    plt.ylim(0, min(1.5, max(df['wer']) * 1.5))  # Increased from 1.2 to 1.5
    
    plt.tight_layout()
    plt.legend()
    
    # Add values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}', ha='center', va='bottom', rotation=0)
    
    # Save the figure
    output_file = output_dir / "wer_analysis_chinese.png"
    plt.savefig(output_file)
    print(f"Figure with Chinese labels saved to: {output_file}")
    
    # Save detailed results to a CSV
    detailed_output = output_dir / "wer_detailed_results.csv"
    df[['file_name', 'transcription', 'correct_transcription', 'wer']].to_csv(detailed_output, index=False)
    print(f"Detailed WER results saved to: {detailed_output}")
    
    # Create a text file with summary statistics
    stats_file = output_dir / "wer_stats.txt"
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write("Word Error Rate Statistics for Chinese Command Transcriptions\n")
        f.write("="*60 + "\n")
        f.write(f"Number of files analyzed: {len(df)}\n")
        f.write(f"Average WER: {avg_wer:.3f}\n")
        f.write(f"Minimum WER: {min(wer_scores):.3f}\n")
        f.write(f"Maximum WER: {max(wer_scores):.3f}\n")
        f.write("\nCommands with highest error rates:\n")
        
        # Get the 3 commands with highest WER
        worst_indices = np.argsort(wer_scores)[-3:]
        for idx in worst_indices:
            f.write(f"  Command {idx+1}: WER={wer_scores[idx]:.3f}\n")
            f.write(f"    Reference: {correct_transcriptions[idx]}\n")
            f.write(f"    Whisper  : {df.iloc[idx]['transcription']}\n\n")
            
        f.write("\nCommands with lowest error rates:\n")
        
        # Get the 3 commands with lowest WER
        best_indices = np.argsort(wer_scores)[:3]
        for idx in best_indices:
            f.write(f"  Command {idx+1}: WER={wer_scores[idx]:.3f}\n")
            f.write(f"    Reference: {correct_transcriptions[idx]}\n")
            f.write(f"    Whisper  : {df.iloc[idx]['transcription']}\n\n")
    
    print(f"WER statistics saved to: {stats_file}")
    
    # Display the figure
    plt.show()

if __name__ == "__main__":
    calculate_wer() 