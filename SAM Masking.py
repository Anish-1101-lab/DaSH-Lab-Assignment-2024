import torch
import torch.nn.functional as F


attention_scores = torch.tensor([[0.1, 0.2, 0.7],
                                 [0.5, 0.3, 0.2],
                                 [0.3, 0.3, 0.4]])


attention_mask = torch.tensor([[1, 1, 0],
                               [1, 1, 1],
                               [1, 0, 1]])


masked_attention_scores = attention_scores * attention_mask


attention_probs = F.softmax(masked_attention_scores, dim=-1)
print(attention_probs)
