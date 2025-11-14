# Load the image-segmentation pipeline
pipe = pipeline('image-segmentation', model='briaai/RMBG-1.4', trust_remote_code=True)

# Use the pipeline to predict the class of the sample image
outputs = pipe(image)

plt.imshow(outputs)
plt.show()