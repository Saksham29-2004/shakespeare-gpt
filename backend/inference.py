
import torch
import torch.nn.functional as F


@torch.no_grad()
def generate(
    model,
    input_ids,
    max_new_tokens=100,
    temperature=1.0,
):
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    model.eval()

    for _ in range(max_new_tokens):
        context = input_ids[:, -model.block_size:]

        logits = model(context)
        logits = logits[:, -1, :]
        logits = logits / temperature

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat(
            [input_ids, next_token], dim=1
        )

    return input_ids
