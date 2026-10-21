
import torch

from loader import load_model
from tokenizer import BPETokenizer
from inference import generate


class TextGenerator:
    def __init__(
        self,
        checkpoint_path="shakespeare_gpt.pth",
        config_path="config.json",
        tokenizer_path="tokenizer.json"
    ):
        self.model = load_model(
            checkpoint_path=checkpoint_path,
            config_path=config_path
        )

        self.tokenizer = BPETokenizer(tokenizer_path)

    def generate_text(
        self,
        prompt,
        max_new_tokens=100,
        temperature=0.8
    ):
        token_ids = self.tokenizer.encode(prompt)

        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long
        )

        output_ids = generate(
            self.model,
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )

        return self.tokenizer.decode(
            output_ids[0].tolist()
        )
