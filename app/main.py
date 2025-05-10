import os
from dotenv import load_dotenv
import threading
import pyaudio
import time
import utils
from client import oai_init, rpi_init

# Load environment variables
load_dotenv()

# Extract environment variables
MODEL_NAME = os.getenv('MODEL_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
SYSTEM_PROMPT = str(os.getenv('SYSTEM_PROMPT'))
RPI_URL = os.getenv('RPI_URL')

# Verify environment variables are loaded
if not all([MODEL_NAME, OPENAI_API_KEY, SYSTEM_PROMPT]):
    raise ValueError("Missing required environment variables")

# Set up websocket connection
oai_url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-12-17"
headers = [
    "Authorization: Bearer " + OPENAI_API_KEY,
    "OpenAI-Beta: realtime=v1"
]

# Set up audio stream
CHUNK = 1024 * 2  # Number of frames per buffer
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000  # Sample rate expected by whisper

# Initialize websocket clients
ws_oai = oai_init(oai_url, headers)
ws_rpi = rpi_init(RPI_URL)

# Main function
if __name__ == "__main__":

    # Start WebSocket in a separate thread
    ws_oai_thread = threading.Thread(target=ws_oai.run_forever)
    ws_oai_thread.daemon = True
    ws_oai_thread.start()

    ws_rpi_thread = threading.Thread(target=lambda: ws_rpi.run_forever())
    ws_rpi_thread.daemon = True
    ws_rpi_thread.start()
    
    try:
        while True:
            time.sleep(1)
            if ws_oai.sock and ws_oai.sock.connected:
                utils.stream_audio(ws_oai, FORMAT, CHANNELS, RATE, CHUNK)
            else:
                print("Connection lost. Reconnecting...")
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping application...")
        ws_oai.close()
        ws_rpi.close()