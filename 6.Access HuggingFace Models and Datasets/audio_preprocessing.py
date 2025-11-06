# Resample the audio to a frequency of 16,000 Hz
dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

# Load the audio processor
processor = AutoProcessor.from_pretrained('openai/whisper-small')

# Preprocess the audio data of the 0th dataset element
audio_pp = processor(dataset[0]["audio"]["array"], sampling_rate=16000, padding=True, return_tensors="pt")
make_spectrogram(audio_pp["input_features"][0])