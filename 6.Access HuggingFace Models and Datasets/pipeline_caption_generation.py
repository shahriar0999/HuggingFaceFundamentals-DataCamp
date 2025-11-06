# Load the image-to-text pipeline
pipe = pipeline(task="image-to-text", model="Salesforce/blip-image-captioning-base")

# Use the pipeline to generate a caption with the image of datapoint 3
pred = pipe(dataset[3]["image"])

print(pred)

# Passing keyword arguments
# Load a text-to-audio pipeline
musicgen = pipeline(task="text-to-audio", model="facebook/musicgen-small", framework="pt")

# Make a dictionary to set the generation temperature to 0.8 and max_new_tokens to 1
generate_kwargs = {"temperature": 0.8, "max_new_tokens": 1}

# Generate an audio array passing the arguments
outputs = musicgen("Classic rock riff", generate_kwargs=generate_kwargs)
sf.write("output.wav", outputs["audio"][0][0], outputs["sampling_rate"])