# Load the image-classification pipeline
pipe = pipeline('image-classification', 'google/mobilenet_v2_1.0_224')

# Use the pipeline to predict the class of the sample image
pred = pipe(image)

# Print the first (highest probability) label
print("Predicted class:", pred[0]['label'])