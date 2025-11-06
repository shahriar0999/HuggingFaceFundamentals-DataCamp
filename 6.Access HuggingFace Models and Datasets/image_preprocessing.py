# Load the image from index 5 of the dataset
image = dataset[5]["image"]

# Load the image processor of the pretrained model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# Preprocess the image
inputs = processor(images=image, return_tensors="pt")

# Generate a caption using the model
output = model.generate(**inputs)
print(f'Generated caption: {processor.decode(output[0])}')
print(f'Original caption: {dataset[5]["caption"][0]}')