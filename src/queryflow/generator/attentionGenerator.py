"""
Generation component with attention-based hallucination detection.
Separates generation logic from retrieval while maintaining original functionality.
"""
from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple, Any
import torch
from src.queryflow.generator.generatorInterface import BaseGenerator, GenerationConfig
from src.queryflow.attention.attention import AttentionAnalyzer


@dataclass
class GenerationResult:
    """Container for generation results including hallucination info"""
    text: str
    is_hallucination: bool
    previous_text: str = ""  # Valid text before hallucination
    current_tokens: Optional[List[str]] = None  # Tokens in current generation
    current_hits: Optional[List[int]] = None  # Hallucination markers
    attention_weights: Optional[torch.Tensor] = None


class AttentionGenerator(BaseGenerator):
    """
    Generator with attention-based hallucination detection.
    Maps the generation portion of the original AttnWeightRAG implementation.
    """

    def __init__(
            self,
            model_name_or_path: str,
            hallucination_threshold: float = 0.5,
            generate_max_length: int = 100,
            method: str = "attn_prob"  # or "dragin" for entropy
    ):
        super().__init__(model_name_or_path)
        self.hallucination_threshold = hallucination_threshold
        self.generate_max_length = generate_max_length
        self.method = method
        self.attention_analyzer = AttentionAnalyzer(
            hallucination_threshold=hallucination_threshold,
            space_token=self.space_token
        )

    def generate_with_monitoring(
            self,
            prompt: str,
            current_text: str = ""
    ) -> GenerationResult:
        """
        Generate text while monitoring for hallucinations.
        Maps the core generation logic from the original implementation.
        """
        # Configure generation parameters based on method
        config = GenerationConfig(
            max_length=self.generate_max_length,
            return_attention=True,
            return_logprobs=self.method == "attn_prob",
            use_entropy=self.method == "dragin"
        )

        # Generate text
        output = self.generate(
            prompt + " " + current_text if current_text else prompt,
            config=config
        )

        # Process generation
        tokens, attns = self.attention_analyzer.process_attention_patterns(
            self.tokenizer.encode(output.text, return_tensors="pt"),
            self.tokenizer.convert_ids_to_tokens(
                self.tokenizer.encode(output.text)
            ),
            output.attention_weights[-1][0] if output.attention_weights else None,
            None  # generated_tokens not needed for processing
        )

        # Calculate weights based on method
        if self.method == "dragin":
            weights = output.entropies if output.entropies else []
        else:
            weights = [-v for v in output.logprobs] if output.logprobs else []

        # Check for hallucination
        hallucination, previous_text, curr_tokens, curr_hits = (
            self.attention_analyzer.analyze_generation(
                output.text,
                tokens,
                attns,
                weights
            )
        )

        return GenerationResult(
            text=output.text,
            is_hallucination=hallucination,
            previous_text=previous_text if hallucination else output.text,
            current_tokens=curr_tokens if hallucination else tokens,
            current_hits=curr_hits,
            attention_weights=output.attention_weights[-1][0] if output.attention_weights else None
        )


class MonitoredGenerator:
    """
    High-level generator that maintains generation state and handles text accumulation.
    This separates the text building logic from the RAG implementation.
    """

    def __init__(
            self,
            generator: AttentionGenerator,
            max_steps: int = 5
    ):
        self.generator = generator
        self.max_steps = max_steps

    def generate_until_complete(
            self,
            prompt: str,
            stop_phrases: List[str] = None,
    ) -> Tuple[str, List[GenerationResult]]:
        """
        Generate text until completion or hallucination.

        Args:
            prompt: Initial prompt
            stop_phrases: Optional phrases that signal completion

        Returns:
            Tuple of:
            - Final generated text
            - List of generation results for each step
        """
        if stop_phrases is None:
            stop_phrases = ["the answer is"]

        text = ""
        results = []

        for _ in range(self.max_steps):
            # Generate next piece
            result = self.generator.generate_with_monitoring(prompt, text)
            results.append(result)

            if result.is_hallucination:
                break

            # Accumulate text
            new_text = result.text.strip()
            if not new_text:
                break

            text = (text.strip() + " " + new_text).strip()

            # Check completion
            tokens_count = len(self.generator.tokenizer.encode(text))
            if (tokens_count > self.generator.generate_max_length or
                    len(text) <= len(text) or
                    any(phrase in text.lower() for phrase in stop_phrases)):
                break

        return text, results