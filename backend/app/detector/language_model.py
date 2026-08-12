from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_NAME = "EleutherAI/pythia-160m"


@dataclass
class TokenScore:
    token: str
    token_id: int
    probability: float
    log_probability: float
    negative_log_likelihood: float


@dataclass
class RawTextScore:
    text: str
    token_count: int
    tokens: list[TokenScore]


class LanguageModelAnalyzer:
    """
    Uses a causal language model as a measurement instrument.

    This class is responsible only for:
    - loading the language model
    - running inference
    - extracting token-level probabilities

    It does NOT calculate detector features.
    It does NOT decide whether text is AI-generated.
    """

    def __init__(
        self,
        model_name: str = MODEL_NAME,
        device: str | None = None,
    ) -> None:

        self.model_name = model_name

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        print(f"Loading tokenizer: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        print(f"Loading model: {model_name}")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

        self.model.to(self.device)
        self.model.eval()

        print(f"Model loaded on: {self.device}")

    @torch.inference_mode()
    def score_text(self, text: str) -> RawTextScore:
        """
        Run the text through the language model and return
        token-level measurements.
        """

        if not text or not text.strip():
            raise ValueError("Text cannot be empty.")

        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            add_special_tokens=True,
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits

        # A causal language model predicts the next token.
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = torch.log_softmax(
            shift_logits,
            dim=-1,
        )

        token_log_probs = log_probs.gather(
            dim=-1,
            index=shift_labels.unsqueeze(-1),
        ).squeeze(-1)

        token_nll = -token_log_probs
        probabilities = torch.exp(token_log_probs)

        token_ids = shift_labels[0].tolist()
        nll_values = token_nll[0].tolist()
        probability_values = probabilities[0].tolist()
        log_probability_values = token_log_probs[0].tolist()

        token_scores: list[TokenScore] = []

        for (
            token_id,
            nll,
            probability,
            log_probability,
        ) in zip(
            token_ids,
            nll_values,
            probability_values,
            log_probability_values,
        ):

            token = self.tokenizer.decode(
                [token_id],
                clean_up_tokenization_spaces=False,
            )

            token_scores.append(
                TokenScore(
                    token=token,
                    token_id=token_id,
                    probability=float(probability),
                    log_probability=float(log_probability),
                    negative_log_likelihood=float(nll),
                )
            )

        return RawTextScore(
            text=text,
            token_count=len(token_scores),
            tokens=token_scores,
        )