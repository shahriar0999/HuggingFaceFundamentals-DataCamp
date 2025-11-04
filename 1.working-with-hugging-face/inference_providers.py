import os
from huggingface_hub import InferenceClient


client = InferenceClient(
    provider="together",    
    api_key=os.environ["HF_TOKEN"])

completion = client.chat.completions.create(
    model="deepseek-ai/DeepSeek-V3",    
    messages=[{"role": "user","content": "What is the capital of France?"}]) 



print(completion.choices[0].message)