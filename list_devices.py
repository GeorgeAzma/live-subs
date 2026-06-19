import sounddevice as sd

devices = sd.query_devices()
for i, d in enumerate(devices):
    if d['max_input_channels'] > 0:
        print(f"  {i}: {d['name']}  in={d['max_input_channels']}  rate={d['default_samplerate']}")
