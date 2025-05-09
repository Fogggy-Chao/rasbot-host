import os
import time
import csv
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def transcribe_audio(file_path):
    """
    Transcribe an audio file using OpenAI's Whisper model and measure response time.
    
    Args:
        file_path (str): Path to the audio file
    
    Returns:
        tuple: (transcription text, response time in seconds)
    """
    start_time = time.time()
    
    with open(file_path, "rb") as audio_file:
        try:
            # Call OpenAI API to transcribe audio
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            
            end_time = time.time()
            response_time = end_time - start_time
            
            return transcript.text, response_time
        except Exception as e:
            print(f"Error transcribing {file_path}: {e}")
            return None, None

def main():
    # Define paths
    commands_dir = Path("../essay/Data/commands/simple_commands")
    output_file = Path("../essay/Data/transcription_results.csv")
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if the commands directory exists
    if not commands_dir.exists():
        print(f"Directory not found: {commands_dir}")
        return
    
    # Collect results
    results = []
    
    # Process all MP3 files in the directory
    for file_path in commands_dir.glob("*.mp3"):
        print(f"Processing: {file_path.name}")
        transcription, response_time = transcribe_audio(str(file_path))
        
        if transcription is not None:
            results.append({
                "file_name": file_path.name,
                "transcription": transcription,
                "response_time": round(response_time, 3)
            })
    
    # Write results to CSV file
    if results:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ["file_name", "transcription", "response_time"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for result in results:
                writer.writerow(result)
        
        print(f"Transcription results saved to: {output_file}")
    else:
        print("No files were successfully transcribed.")

if __name__ == "__main__":
    main()
