# Create the summarization pipeline
summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum")

# Summarize the text
summary_text = summarizer(original_text)

# Compare the length
print(f"Original text length: {len(original_text)}")
print(f"Summary length: {len(summary_text[0]['summary_text'])}")


#Adjusting the summary length
# Generate a summary of original_text between 1 and 10 tokens
short_summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum", min_new_tokens=1, max_new_tokens=10)

short_summary_text = short_summarizer(original_text)

print(short_summary_text[0]["summary_text"])

# for longer summary
# Repeat for a summary between 50 and 150 tokens
long_summarizer = pipeline(task="summarization", model="cnicu/t5-small-booksum", min_new_tokens=50, max_new_tokens=150)

long_summary_text = long_summarizer(original_text)

print(long_summary_text[0]["summary_text"])