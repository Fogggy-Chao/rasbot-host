# import numpy as np
# import struct
# import base64
# import soundfile as sf
# def float_to_16bit_pcm(float32_array):
#     clipped = [max(-1.0, min(1.0, x)) for x in float32_array]
#     pcm16 = b''.join(struct.pack('<h', int(x * 32767)) for x in clipped)
#     return pcm16

# def base64_encode_audio(float32_array):
#     pcm_bytes = float_to_16bit_pcm(float32_array)
#     encoded = base64.b64encode(pcm_bytes).decode('ascii')
#     return encoded

# files = [
#     '../media/sys_audio.wav',
# ]

# for filename in files:
#     data, samplerate = sf.read(filename, dtype='float32')  
#     channel_data = data[:, 0] if data.ndim > 1 else data
#     base64_chunk = base64_encode_audio(channel_data)
    
#     # Save to file
#     save_path = "../media/sys_audio_base64.txt"
#     with open(save_path, "w") as f:
#         f.write(base64_chunk)
        
#     print("Successfully converted and saved audio buffer")
