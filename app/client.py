import websocket
import json
import os
from dotenv import load_dotenv
from utils import message_to_rpi
import time
from tool import TOOLS

# Global variable to hold the executor instance
_executor_instance = None

# Load environment variables
load_dotenv()

# Get Raspberry Pi connection details
RPI_URL = os.getenv('RPI_URL')

# Load system prompt from file
PROMPT_FILE = "system_prompt.txt" # Or .md if you used that
try:
    with open(os.path.join(os.path.dirname(__file__), PROMPT_FILE), 'r') as f:
        SYSTEM_PROMPT = f.read()
except FileNotFoundError:
    raise FileNotFoundError(f"Error: System prompt file not found at {PROMPT_FILE}")

model_tools = TOOLS

# Initialize OpenAI websocket client
def oai_init(url, headers, executor):
    global ws_oai, _executor_instance
    _executor_instance = executor
    ws_oai = websocket.WebSocketApp(
        url,
        header=headers,
        on_open=oai_on_open,
        on_message=oai_on_message,
        on_error=oai_on_error,
        on_close=oai_on_close)
    return ws_oai

# Initialize Raspberry Pi websocket client
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
                "tools":model_tools,
                "tool_choice": "auto"
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
        response_type = server_response.get('type')

        # Handle function call
        if response_type == "response.done" and server_response.get("response", {}).get("output", [])[0].get("type") == "function_call":
            tool_call = server_response["response"]["output"][0]
            tool_call_id = tool_call.get("call_id")
            name = tool_call.get("name")
            args = json.loads(tool_call.get("arguments", "{}"))
            print(f"Function: {name}\nArguments: {args}")

            if not tool_call_id or not name:
                print("Error: Missing tool_call_id or function name in response.")
                return

            try:
                # Execute the tool using the executor
                result = _executor_instance(name, args)
                print(f"Tool Result: {result}")

                # Format the result message for OpenAI
                tool_result_message = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": json.dumps(result)
                    }
                }
                ws.send(json.dumps(tool_result_message))
                create_response_message = {
                    "type": "response.create"
                }
                ws.send(json.dumps(create_response_message))
                print("Tool result sent back to OpenAI.")

            except Exception as e:
                print(f"Error executing tool {name} or sending result: {e}")
                # Optionally send an error result back to OpenAI
                error_result_message = {
                    "type": "conversation.item.create",
                    "item": {
                        "type": "function_call_output",
                        "call_id": tool_call_id,
                        "output": json.dumps({"error": str(e)})
                    }
                }
                ws.send(json.dumps(error_result_message))


        # Handle text response (check for DONE/ERROR)
        elif response_type == "response.done" and server_response.get("response", {}).get("output", [])[0].get("type") == "text":
            text_output = server_response["response"]["output"][0]["value"]
            print(f"Final Text Response: {text_output}")
            if text_output.startswith("DONE:"):
                print("Goal achieved.")
                # Potentially add logic here to stop or reset
            elif text_output.startswith("ERROR:"):
                print(f"Error reported by LLM: {text_output}")
                # Potentially add logic here to stop or handle the error

        elif response_type == "response.audio_transcript.done":
            print(f"Audio transcript: {server_response.get('transcript')}")

        elif response_type == 'conversation.item.input_audio_transcription.completed':
            # elapsed = time.perf_counter() - start # 'start' is not defined here, remove for now
            print(f"Input audio transcript: {server_response.get('transcript')}")
            # print(f"Latency: {elapsed}") # Remove latency calculation for now

        elif response_type == 'input_audio_buffer.speech_started':
            # start = time.perf_counter() # 'start' is not defined here, remove for now
            print("Speech started.")

        elif response_type == 'input_audio_buffer.speech_stopped':
            # input_time = time.perf_counter() - start # 'start' is not defined here, remove for now
            # print(f"User input time: {input_time}") # Remove time calculation for now
            print("Speech stopped.")

        # elif response_type == 'audio.done':

        elif response_type == 'error':
            print(f"Error: {server_response.get('error', 'Unknown error')}")

        else:
            print(f"Other Response Type: {response_type}")
            # print(f"Full message: {server_response}") # Optional: print full message for debugging

    except json.JSONDecodeError:
        print(f"Received non-JSON message: {message}")
    except Exception as e:
        import traceback
        print(f" Error processing message: {e}")
        traceback.print_exc() # Print full traceback for debugging

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
