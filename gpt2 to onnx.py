from transformers import GPT2Model, GPT2Tokenizer
import torch
import onnx

# Load GPT-2 model and tokenizer
model = GPT2Model.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Define a function to convert PyTorch model to ONNX
def convert_to_onnx(model, input_tensor, output_path="gpt2.onnx"):
    torch.onnx.export(model,               # PyTorch model
                      input_tensor,       # Model input (dummy input in this case)
                      output_path,        # Output file path
                      export_params=True, # Store parameters inside the model file
                      opset_version=12,   # ONNX version
                      do_constant_folding=True,  # Optimize the model
                      input_names=['input_ids'],   # Input name
                      output_names=['last_hidden_state']  # Output name
                     )

# Create dummy input tensor
input_text = "Hello, how are you?"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Convert the model
convert_to_onnx(model, input_ids)
