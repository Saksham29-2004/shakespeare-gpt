
import json
import re


class BPETokenizer:
    def __init__(self, tokenizer_path="tokenizer.json"):
        with open(tokenizer_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.token_to_id = data["token_to_id"]
        self.id_to_token = {
            int(token_id): token
            for token_id, token in data["id_to_token"].items()
        }
        self.merges = data["merges"]

    def encode_piece(self, piece):
        tokens = list(piece)

        for left, right in self.merges:
            merged = []
            i = 0

            while i < len(tokens):
                if (
                    i + 1 < len(tokens)
                    and tokens[i] == left
                    and tokens[i + 1] == right
                ):
                    merged.append(left + right)
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1

            tokens = merged

        return tokens

    def encode(self, text):
        pieces = re.findall(r"\s+|[^\w\s]+|\w+", text)
        tokens = []

        for piece in pieces:
            tokens.extend(self.encode_piece(piece))

        return [self.token_to_id[token] for token in tokens]

    def decode(self, token_ids):
        return "".join(
            self.id_to_token[int(token_id)]
            for token_id in token_ids
        )
