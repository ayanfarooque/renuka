# Import Libraries
try:
    from transformers import AutoTokenizer, AutoModelForQuestionAnswering
    import torch
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Required packages not installed. Run: python -m pip install transformers torch"
    ) from exc

# Load Pretrained Model
MODEL_NAME = "distilbert-base-cased-distilled-squad"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

model.eval()  # Set model to evaluation mode

# Question Answering Function
def answer_question(question: str, context: str) -> str:
    inputs = tokenizer(
        question,
        context,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.no_grad():
        outputs = model(**inputs)

    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1

    answer = tokenizer.decode(
        inputs["input_ids"][0][start_idx:end_idx],
        skip_special_tokens=True
    )

    return answer

# Sample Context
context = """
Nowadays banking has become highly advanced due to modern technology.
Customers can transfer money online, check account balances using mobile apps,
and pay bills instantly. Banks use artificial intelligence for fraud detection
and customer support. Internet banking allows users to access their accounts
anytime and anywhere. Digital payments and cashless transactions have made
banking faster, safer, and easier for everyone.
"""

# Example Questions
question1 = "What can customers do using mobile banking apps?"
print("Question:", question1)
print("Answer:", answer_question(question1, context))

print("\n")

question2 = "Why do banks use artificial intelligence?"
print("Question:", question2)
print("Answer:", answer_question(question2, context))