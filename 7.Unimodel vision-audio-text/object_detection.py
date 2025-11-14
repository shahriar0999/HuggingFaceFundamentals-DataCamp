# Load the object-detection pipeline
pipe = pipeline("object-detection", "facebook/detr-resnet-50", revision="no_timm")
pred = pipe(image)
outputs = pipe(image)

for n, obj in enumerate(outputs):
    # Find the detected label
    label = obj['label']
    # Find the confidence score of the prediction
    confidence = obj['score']
    # Obtain the bounding box coordinates
    box = obj['box']
    
    plot_args = {"linewidth": 1, "edgecolor": colors[n], "facecolor": 'none'}
    rect = patches.Rectangle((box['xmin'], box['ymin']), box['xmax']-box['xmin'], box['ymax']-box['ymin'], **plot_args)
    ax.add_patch(rect)
    print(f"Detected {label} with confidence {confidence:.2f} at ({box['xmin']}, {box['ymin']}) to ({box['xmax']}, {box['ymax']})")

plt.show()