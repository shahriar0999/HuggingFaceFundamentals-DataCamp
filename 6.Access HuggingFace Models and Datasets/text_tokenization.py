# Load the first caption from the image at index 5
text = dataset[5]["caption"][0]
print(text)

# Load the first caption from the image at index 5
text = dataset[5]["caption"][0]
print(text)

# Load the tokenizer of the pretrained model
tokenizer = AutoTokenizer.from_pretrained('distilbert/distilbert-base-uncased')

# Perform full preprocessing of the text
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)