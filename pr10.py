try:
    from transformers import T5Tokenizer, T5ForConditionalGeneration
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Required packages not installed. Run: python -m pip install transformers sentencepiece"
    ) from exc

model_name = "t5-small"

tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

text = """
Artificial Intelligence is transforming industries across the world.
It enables machines to learn from data and make decisions.
AI is used in healthcare, finance, education, and many other fields.
However, it also raises ethical concerns regarding privacy and job displacement.
"""

input_text = "summarize: " + text.strip()

inputs = tokenizer.encode(
    input_text,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

summary_ids = model.generate(
    inputs,
    max_length=60,
    min_length=20,
    length_penalty=2.0,
    num_beams=4,
    early_stopping=True
)

summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("\n===== ORIGINAL TEXT =====\n")
print(text.strip())
print("\n===== SUMMARY =====\n")
print(summary)