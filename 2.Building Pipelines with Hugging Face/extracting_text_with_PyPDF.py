from pypdf import PdfReader

# Extract text from the PDF
reader = PdfReader("US_Employee_Policy.pdf")

# Extract text from all pages
document_text = ""
for page in reader.pages: 
    document_text += page.extract_text()

print(document_text)


# Load the question-answering pipeline
qa_pipeline = pipeline(task="question-answering", model="distilbert-base-cased-distilled-squad")

question = "What is the notice period for resignation?"

# Get the answer from the QA pipeline
result = qa_pipeline(question=question, context=document_text)

# Print the answer
print(f"Answer: {result['answer']}")