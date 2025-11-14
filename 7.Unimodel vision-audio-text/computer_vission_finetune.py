# Create a train/test split within the HF dataset
data_splits = dataset.train_test_split(test_size=0.2, seed=42)

# Apply the transformations
dataset_transformed = data_splits.with_transform(transforms)

# Plot the transformed image
plt.imshow(dataset_transformed["train"][0]["pixel_values"].permute(1, 2, 0))
plt.show()


# model class
# Obtain the new label names from the dataset
labels = dataset["train"].features["label"].names

label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

model = AutoModelForImageClassification.from_pretrained(
    "google/mobilenet_v2_1.0_224",
    num_labels=len(labels),
    # Add the id2label mapping
    id2label=id2label,
    # Add the corresponding label2id mapping
    label2id=label2id,
    # Add the required flag to change the number of classes
    ignore_mismatched_sizes=True
)

#Training Configuration
training_args = TrainingArguments(
    output_dir="dataset_finetune",
    # Adjust the learning rate
    learning_rate=6e-5,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    push_to_hub=False
)

trainer = Trainer(
    # Provide the model and datasets
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    processing_class=image_processor,
    compute_metrics=compute_metrics,
)