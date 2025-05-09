import websocket
import json
import os
from dotenv import load_dotenv
from utils import message_to_rpi
import time

# Load environment variables
load_dotenv()

# Get Raspberry Pi connection details
RPI_URL = os.getenv('RPI_URL')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT')

# Initialize OpenAI client

def oai_init(url, headers):
    global ws_oai
    ws_oai = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=oai_on_open,
        on_message=oai_on_message,
        on_error=oai_on_error,
        on_close=oai_on_close)
    return ws_oai

# Initialize Raspberry Pi client
def rpi_init(url):
    global ws_rpi
    ws_rpi = websocket.WebSocketApp(
        url,
        on_open=rpi_on_open,
        on_message=rpi_on_message,
        on_error=rpi_on_error,
        on_close=rpi_on_close)
    return ws_rpi

# OpenAI on open handler
def oai_on_open(ws):
    print("Connected to oai wss server.")
    try:
        event = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": SYSTEM_PROMPT,
                "input_audio_transcription": {
                    "model": "whisper-1"
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 400,
                    "create_response": True
                },
                "tools":[
                    {
                        "type": "function",
                        "name": "motor_driver_control",
                        "description": "Control the motor driver by defining the signals of the motors.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "motor1": {
                                    "type": "boolean",
                                    "description": "The signal of motor 1. True means the motor is on, False means the motor is off."
                                },
                                "motor2": {
                                    "type": "boolean",
                                    "description": "The signal of motor 2. True means the motor is on, False means the motor is off."
                                },  
                                "motor3": {
                                    "type": "boolean",
                                    "description": "The signal of motor 3. True means the motor is on, False means the motor is off."
                                },
                                "motor4": {
                                    "type": "boolean",
                                    "description": "The signal of motor 4. True means the motor is on, False means the motor is off."
                                },
                                "m1_speed": {
                                    "type": "number",
                                    "description": "The speed of motor 1. The value should be between 0 and 1."
                                },
                                "m2_speed": {
                                    "type": "number",
                                    "description": "The speed of motor 2. The value should be between 0 and 1."
                                },
                                "m3_speed": {
                                    "type": "number",
                                    "description": "The speed of motor 3. The value should be between 0 and 1."
                                },
                                "m4_speed": {
                                    "type": "number",
                                    "description": "The speed of motor 4. The value should be between 0 and 1."
                                }
                            },
                            "required": ["motor1", "motor2", "motor3", "motor4", "m1_speed", "m2_speed", "m3_speed", "m4_speed"]
                        }
                    }
                ]
            }
        }
        ws.send(json.dumps(event))
    except Exception as e:
        print(f"Error sending event: {e}")

# OpenAI on message handler
def oai_on_message(ws, message):
    try:
        # Parse the JSON message from server
        server_response = json.loads(message)
        
        # Handle server response
        if server_response['type'] == "response.function_call_arguments.done":
            args = json.loads(server_response['arguments'])
            print(f"Function call arguments: {args}")
            try:
                global ws_rpi
                # Check if the Raspberry Pi connection is established
                if ws_rpi and ws_rpi.sock and ws_rpi.sock.connected:
                    message_to_rpi(
                        bool(args.get('motor1', False)),
                        bool(args.get('motor2', False)),
                        bool(args.get('motor3', False)),
                        bool(args.get('motor4', False)), 
                        float(args.get('m1_speed')),
                        float(args.get('m2_speed')),
                        float(args.get('m3_speed')),
                        float(args.get('m4_speed')),
                        ws_rpi
                    )
            except Exception as e:
                print(f"Error executing motor control: {e}")
                print(f"Received arguments: {args}")

        elif server_response['type'] == "response.audio_transcript.done":
            print(f"Audio transcript: {server_response['transcript']}")

        elif server_response['type'] == 'conversation.item.input_audio_transcription.completed':
            elapsed = time.perf_counter() - start
            print(f"Input audio transcript: {server_response['transcript']}")
            print(f"Latency: {elapsed}")

        elif server_response['type'] == 'input_audio_buffer.speech_started':
            start = time.perf_counter()
        
        elif server_response['type'] == 'input_audio_buffer.speech_stopped':
            input_time = time.perf_counter() - start
            print(f"User input time: {input_time}") 

        else:
            print(f"Response: {server_response['type']}")
            # None
            
    except Exception as e:
        print(f" Error processing message: {e}")

# OpenAI on error handler
def oai_on_error(ws, error):
    print("Error:", error)

# OpenAI on close handler
def oai_on_close(ws, close_status_code, close_msg):
    print("Connection closed")

# Raspberry Pi on open handler
def rpi_on_open(ws):
    print("Connected to Raspberry Pi.")

# Raspberry Pi on message handler
def rpi_on_message(ws, message):
    response = json.loads(message)
    print("Raspberry Pi message:", response)

# Raspberry Pi on error handler
def rpi_on_error(ws, error):
    print("Raspberry Pi error:", error)

# Raspberry Pi on close handler
def rpi_on_close(ws, close_status_code, close_msg):
    print("Raspberry Pi connection closed")
