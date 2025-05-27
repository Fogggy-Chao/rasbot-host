import os
from dotenv import load_dotenv
import threading
import pyaudio
import time
import utils
from client import oai_init, rpi_init
from executor import MockExecutor, RPIExecutor


# Load environment variables
load_dotenv()


# Extract environment variables
MODEL_NAME = os.getenv('MODEL_NAME')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
RPI_URL = os.getenv('RPI_URL')

# Verify environment variables are loaded (excluding SYSTEM_PROMPT now)
if not all([MODEL_NAME, OPENAI_API_KEY]): # <-- Check only required env vars
    raise ValueError("Missing required environment variables (MODEL_NAME, OPENAI_API_KEY)")

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

# Initialize websocket RPI client first as it's needed by the RPIExecutor
ws_rpi = rpi_init(RPI_URL)

# Initialize executor
EXECUTOR = RPIExecutor(ws_rpi)# <-- switch to RPIExecutor(ws_rpi) later

# Initialize websocket OAI client
ws_oai = oai_init(oai_url, headers, EXECUTOR)

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
                print("Connecting...")
                time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping application...")
        ws_oai.close()
        ws_rpi.close()