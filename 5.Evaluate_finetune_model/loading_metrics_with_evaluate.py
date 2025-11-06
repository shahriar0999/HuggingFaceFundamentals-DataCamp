# Load the metrics
accuracy = evaluate.load("accuracy")
precision = evaluate.load('precision')
recall = evaluate.load('recall')
f1 = evaluate.load('f1')


"""
Describing metrics
It's never a bad time to revise the definitions of some popular metrics.

The evaluate library has been loaded for you, along with the four classification metrics: accuracy, precision, recall, and f1.
"""
# Obtain a description of each metric
print(accuracy.description)
print(precision.description)
print(recall.description)
print(f1.description)


# See the required data types
print(f"The required data types for accuracy are: {accuracy.features}.")
print(f"The required data types for precision are: {precision.features}.")
print(f"The required data types for recall are: {recall.features}.")
print(f"The required data types for f1 are: {f1.features}.")


"""
Using evaluate metrics
It's time to evaluate your LLM that classifies customer support interactions. Picking up from where you left your fine-tuned model, you'll now use a new validation dataset to assess the performance of your model.

Some interactions and their corresponding labels have been loaded for you as validate_text and validate_labels. The model and tokenizer are also loaded.
"""

accuracy = evaluate.load("accuracy")
precision = evaluate.load("precision")
recall = evaluate.load("recall")
f1 = evaluate.load("f1")

# Extract the new predictions
predicted_labels = torch.argmax(outputs.logits, dim=1).tolist()

# Compute the metrics by comparing real and predicted labels
print(accuracy.compute(references=validate_labels, predictions=predicted_labels))
print(precision.compute(references=validate_labels, predictions=predicted_labels))
print(recall.compute(references=validate_labels, predictions=predicted_labels))
print(f1.compute(references=validate_labels, predictions=predicted_labels))


"""
Evaluating perplexity
Try your had at generating text and evaluating the perplexity score.

You've been provided some input_text that is the start of a sentence: "Current trends show that by 2030 ".

Use an LLM to generate the rest of the sentence.

An AutoModelForCausalLM model and its tokenizer have been loaded for you as model and tokenizer variables.
"""


# Encode the input text, generate and decode it
input_text_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_text_ids, max_length=20)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("Generated Text: ", generated_text)

# Load and compute the perplexity score
perplexity = evaluate.load("perplexity", module_type="metric")
results = perplexity.compute(model_id="gpt2", predictions=generated_text)
print("Perplexity: ", results['mean_perplexity'])