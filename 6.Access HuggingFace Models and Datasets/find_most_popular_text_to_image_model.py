models = api.list_models(
    # Filter for text-to-image tasks
    task="text-to-image",
    author="CompVis",
    # Filter for models that can be loaded by the diffusers library
    tags="diffusers:StableDiffusionPipeline",
    # Sort according to the most popular
    sort="likes"
)

models = list(models)

# Load the most popular model from models
pipe = StableDiffusionPipeline.from_pretrained(models[0].id)