
import json
import torch
from model import GPTLanguageModel


def load_model(
    checkpoint_path="shakespeare_gpt.pth",
    config_path="config.json"
):
    # Read model configuration.
    with open(config_path, "r") as f:
        config = json.load(f)

    # Recreate the model architecture.
    model = GPTLanguageModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        block_size=config["block_size"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        hidden_dim=config["hidden_dim"]
    )

    # Load the trained weights.
    state_dict = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model
