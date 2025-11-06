# Find the list of models for a task
models = api.list_models(task="text-to-image")
print(f"Task: text-to-image, Models: {len(list(models))}")