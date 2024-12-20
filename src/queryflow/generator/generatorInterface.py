"""
Base RAG interface for language model interactions.
"""
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import torch
from abc import ABC, abstractmethod
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, LlamaForCausalLM


@dataclass
class GenerationOutput:
    """Container for generation outputs"""
    text: str
    tokens: Optional[List[str]] = None
    logprobs: Optional[List[float]] = None
    attention_weights: Optional[torch.Tensor] = None


@dataclass
class GenerationConfig:
    """Configuration for generation"""
    max_length: int = 100
    return_logprobs: bool = False
    return_attention: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0


class BaseGenerator(ABC):
    """Abstract base class for text generation."""

    TRUST_REMOTE_MODELS = {
        "falcon",
        "mpt",
        "codegen",
    }

    def __init__(self, model_name_or_path: str):
        """Initialize the rag.

        Args:
            model_name_or_path: Path or name of the model to load
        """
        self.model_name = model_name_or_path
        self._setup_model()

    def _setup_model(self):
        """Set up the model and tokenizer."""
        # Handle special cases like Llama
        if "llama" in self.model_name.lower():
            self.tokenizer = LlamaTokenizer.from_pretrained(self.model_name, legacy=True)
            self.model = LlamaForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto"
            )
            self.space_token = "▁"
        else:
            needs_trust_remote = any(
                model_type in self.model_name.lower()
                for model_type in self.TRUST_REMOTE_MODELS
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=needs_trust_remote
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                trust_remote_code=needs_trust_remote
            )
            self.space_token = self.tokenizer.tokenize(' ')[0]

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def generate(
            self,
            prompt: str,
            config: Optional[GenerationConfig] = None
    ) -> GenerationOutput:
        """Generate text from a prompt.

        Args:
            prompt: Input text to generate from
            config: Generation configuration

        Returns:
            GenerationOutput containing generated text and optional metadata
        """
        if config is None:
            config = GenerationConfig()

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        attention_mask = torch.ones_like(input_ids)
        input_length = input_ids.shape[1]

        # Set up generation parameters
        generation_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": config.max_length,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "top_k": config.top_k,
            "return_dict_in_generate": True,
            "output_scores": config.return_logprobs or config.return_attention,
            "output_attentions": config.return_attention,
        }

        # Generate
        outputs = self.model.generate(**generation_kwargs)

        # Process outputs
        generated_tokens = outputs.sequences[:, input_length:]
        text = self.tokenizer.decode(generated_tokens[0])

        # Get logprobs if requested
        logprobs = None
        if config.return_logprobs:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = [p.cpu().numpy() for p in transition_scores[0]]

        # Get tokens if needed for logprobs or attention
        tokens = None
        if config.return_logprobs or config.return_attention:
            tokens = [self.tokenizer.decode(t) for t in generated_tokens[0]]

        # Get attention weights if requested
        attention_weights = None
        if config.return_attention and hasattr(outputs, 'attentions'):
            attention_weights = outputs.attentions[-1][0]  # Last layer attention

        return GenerationOutput(
            text=text,
            tokens=tokens,
            logprobs=logprobs,
            attention_weights=attention_weights
        )

    @abstractmethod
    def get_reasoning_chain(self, question: str) -> List[str]:
        """Generate a chain of reasoning steps for a question.

        Args:
            question: The question to reason about

        Returns:
            List of reasoning steps
        """
        pass

    @abstractmethod
    def get_final_answer(self, reasoning_chain: List[str]) -> str:
        """Generate final answer from reasoning chain.

        Args:
            reasoning_chain: List of reasoning steps

        Returns:
            Final answer
        """
        pass