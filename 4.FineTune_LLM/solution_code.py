#Tokenizing Text
# Load the model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

# Tokenize the data
tokenized_training_data = tokenizer(train_data["interaction"], return_tensors="pt", padding=True, truncation=True, max_length=20)

tokenized_test_data = tokenizer(test_data["interaction"], return_tensors="pt", padding=True, truncation=True, max_length=20)

print(tokenized_training_data)
