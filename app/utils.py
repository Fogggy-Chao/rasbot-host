import numpy as np
import struct
import base64
import pyaudio
import websocket
import json

# Float to 16-bit PCM
def float_to_16bit_pcm(float32_array):
    clipped = np.clip(float32_array, -1.0, 1.0)
    pcm16 = struct.pack('<%dh' % len(clipped), *(np.array(clipped * 32767, dtype=np.int16)))
    return pcm16

# Base64 encode audio
def base64_encode_audio(float32_array):
    pcm_bytes = float_to_16bit_pcm(float32_array)
    encoded = base64.b64encode(pcm_bytes).decode('ascii')
    return encoded

# Stream audio
def stream_audio(ws, FORMAT, CHANNELS, RATE, CHUNK):
    p = pyaudio.PyAudio()
    recording = True
    
    # Check if websocket is still connected
    if not ws.sock or not ws.sock.connected:
        print("WebSocket is not connected. Reconnecting...")
        ws.run_forever()
        return

    # Open audio stream
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK
    )
    
    try:
        while recording and ws.sock and ws.sock.connected:
            # Read audio chunk
            audio_data = stream.read(CHUNK)
            # Convert to numpy array
            numpy_data = np.frombuffer(audio_data, dtype=np.float32)
                
            # Encode and send
            base64_chunk = base64_encode_audio(numpy_data)
            event = {
                "type": "input_audio_buffer.append",
                "audio": base64_chunk
            }
            ws.send(json.dumps(event))
            
    except websocket.WebSocketConnectionClosedException:
        print("WebSocket connection closed. Attempting to reconnect...")
        ws.run_forever()
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        # Stop and close the stream
        stream.stop_stream()
        stream.close()
        p.terminate()

# Motor driver control
def message_to_rpi(motor1: bool, motor2: bool, motor3: bool, motor4: bool, m1_speed: float, m2_speed: float, m3_speed: float, m4_speed: float, ws):

    # Create motor signals dictionary
    motor_signals = {
        "motor1": motor1,
        "motor2": motor2,
        "motor3": motor3,
        "motor4": motor4,
        "m1_speed": m1_speed,
        "m2_speed": m2_speed,
        "m3_speed": m3_speed,
        "m4_speed": m4_speed
    }

    # Send the motor signals to the Raspberry Pi
    try:
        message_data = json.dumps(motor_signals)
        ws.send(message_data)
    
    except Exception as e:
        print(f"Error sending signals to RPI: {e}")
