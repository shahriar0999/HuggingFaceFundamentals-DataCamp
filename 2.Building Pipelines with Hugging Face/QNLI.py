# for QNLI pipeline
# Create the pipeline
classifier = pipeline(task='text-classification', model="cross-encoder/qnli-electra-base")

# Predict the output
output = classifier("Where is the capital of France?, Brittany is known for its stunning coastline.")

print(output)