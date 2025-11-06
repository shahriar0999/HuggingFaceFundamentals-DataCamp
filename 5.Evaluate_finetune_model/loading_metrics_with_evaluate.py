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


"""
BLEU translations
Let's get familiar with the BLEU metric.

A pipeline based on the Helsinki-NLP Spanish-English translation model and the BLEU metric has been loaded for you, using evaluate.load("bleu") from the evaluate library.

Given the following inputs and references for evaluation:

input_sentence_1 = "Hola, ¿cómo estás?"

reference_1 = [
     ["Hello, how are you?", "Hi, how are you?"]
     ]

input_sentences_2 = ["Hola, ¿cómo estás?", "Estoy genial, gracias."]

references_2 = [
     ["Hello, how are you?", "Hi, how are you?"],
     ["I'm great, thanks.", "I'm great, thank you."]
     ]
"""

translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

# Translate the first input sentence then calucate the BLEU metric for translation quality
translated_output = translator(input_sentence_1)

translated_sentence = translated_output[0]['translation_text']

print("Translated:", translated_sentence)

results = bleu.compute(predictions=[translated_sentence], references=reference_1)
print(results)



# Translate the input sentences, extract the translated text, and compute BLEU score
translator = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en")

translated_outputs = translator(input_sentences_2)

predictions = [translated_output['translation_text'] for translated_output in translated_outputs]
print(predictions)

results = bleu.compute(predictions=predictions, references=references_2)
print(results)


"""
Evaluating with ROUGE
ROUGE is commonly used to evaluate summarization tasks as it checks for similarities between predictions and references. You have been provided with a model-generated summary, predictions, and a references summary for validate. Calculate the scores to see how well the model performed.

The evaluate library has been loaded for you."""

# Load the rouge metric
rouge = evaluate.load('rouge')

predictions = ["""Pluto is a dwarf planet in our solar system, located in the Kuiper Belt beyond Neptune, and was formerly considered the ninth planet until its reclassification in 2006."""]
references = ["""Pluto is a dwarf planet in the solar system, located in the Kuiper Belt beyond Neptune, and was previously deemed as a planet until it was reclassified in 2006."""]

# Calculate the rouge scores between the predicted and reference summaries
results = rouge.compute(predictions=predictions, references=references)
print("ROUGE results: ", results)


"""
Evaluating with METEOR
METEOR excels at evaluating some of the more semantic features in text. It works similar to ROUGE by comparing a model-generated output to a reference output. You've been provided these texts as generated and reference; it's over to you to evaluate the score.

The evaluate library has been loaded for you."""

meteor = evaluate.load("meteor")

generated = ["The burrow stretched forward like a narrow corridor for a while, then plunged abruptly downward, so quickly that Alice had no chance to stop herself before she was tumbling into an extremely deep shaft."]
reference = ["The rabbit-hole went straight on like a tunnel for some way, and then dipped suddenly down, so suddenly that Alice had not a moment to think about stopping herself before she found herself falling down a very deep well."]

# Compute and print the METEOR score
results = meteor.compute(predictions=generated, references=reference)
print("Meteor: ", results['meteor'])


"""
Evaluating with EM
Exact Match helps us evaluate models when it comes to extractive question and answering but looking for, you guessed it, exact matches! Once again, you have been provided some predictions and references for evaluation. The evaluate library has been loaded for you.
"""


# Load the metric
exact_match = evaluate.load('exact_match')

predictions = ["It's a wonderful day", "I love dogs", "DataCamp has great AI courses", "Sunshine and flowers"]
references = ["What a wonderful day", "I love cats", "DataCamp has great AI courses", "Sunsets and flowers"]

# Compute the exact match and print the results
results = exact_match.compute(predictions=predictions, references=references)
print("EM results: ", results)