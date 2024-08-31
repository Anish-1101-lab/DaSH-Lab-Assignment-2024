from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch


model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")


input_text = "Once upon a time,"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)

# Manual inference with sampling techniques
temperature = 0.7
top_k = 50
max_length = 50  # Max length of the generated text

# Generate text
with torch.no_grad():
    for _ in range(max_length):
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :] / temperature  # Apply temperature scaling

        # Apply Top-K sampling
        top_k_logits, top_k_indices = torch.topk(logits, top_k)
        top_k_probs = torch.nn.functional.softmax(top_k_logits, dim=-1)

        # Sample from the top-K distribution
        next_token = torch.multinomial(top_k_probs, 1)

        # Update input_ids with the new token
        input_ids = torch.cat([input_ids, top_k_indices[0, next_token].unsqueeze(0)], dim=-1)

# Decode and print the output
generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
print(generated_text)
