"""
Reasoning chain implementations for different RAG strategies.
"""
from typing import List, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
import torch
import numpy as np

from src.queryflow.generator.generatorInterface import BaseGenerator, GenerationConfig


@dataclass
class ReasoningStep:
    """A single step in a reasoning chain"""
    text: str
    confidence: float
    evidence: Optional[str] = None
    attention_weights: Optional[torch.Tensor] = None
    needs_verification: bool = False


@dataclass
class ReasoningChain:
    """Complete reasoning chain with metadata"""
    steps: List[ReasoningStep]
    final_answer: str
    confidence: float
    hallucination_detected: bool = False
    retrieved_context: Optional[List[str]] = None


class BaseReasoner(ABC):
    """Base class for reasoning implementations"""

    def __init__(self, generator: BaseGenerator):
        self.generator = generator

    @abstractmethod
    def generate_reasoning_chain(
            self,
            question: str,
            context: Optional[List[str]] = None
    ) -> ReasoningChain:
        """Generate a reasoning chain for a question"""
        pass

    @abstractmethod
    def detect_hallucination(self, step: ReasoningStep) -> bool:
        """Detect if a reasoning step contains hallucination"""
        pass

    def _build_prompt(
            self,
            question: str,
            context: Optional[List[str]] = None,
            current_text: str = ""
    ) -> str:
        """Build prompt with optional context and current reasoning."""
        parts = []

        if context:
            context_str = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(context))
            parts.append(f"Context:\n{context_str}")

        parts.append(f"Question: {question}")

        if current_text:
            parts.append(f"Current reasoning: {current_text}")

        parts.append("Let's solve this step by step:")
        return "\n\n".join(parts)

class SimpleReasoner(BaseReasoner):
    """Basic reasoning implementation - generate once with context"""

    def generate_reasoning_chain(
            self,
            question: str,
            context: Optional[List[str]] = None
    ) -> ReasoningChain:
        # Build prompt with context if provided
        prompt = self._build_prompt(question, context)

        # Generate single response
        output = self.generator.generate(
            prompt,
            config=GenerationConfig(return_logprobs=True)
        )

        # Convert to reasoning step
        step = ReasoningStep(
            text=output.text,
            confidence=np.mean([1 - np.exp(p) for p in output.logprobs]) if output.logprobs else 0.0
        )

        return ReasoningChain(
            steps=[step],
            final_answer=self._extract_answer(output.text),
            confidence=step.confidence
        )

    def detect_hallucination(self, step: ReasoningStep) -> bool:
        # Simple confidence threshold
        return step.confidence < 0.5

    def _extract_answer(self, text: str) -> str:
        if "the answer is" in text.lower():
            answer = text.lower().split("the answer is")[-1].strip()
            return "yes" if answer.startswith("yes") else "no"
        return ""


class TokenBasedReasoner(BaseReasoner):
    """Token-level hallucination detection and reasoning"""

    def __init__(self, generator: BaseGenerator, threshold: float = 0.5):
        super().__init__(generator)
        self.threshold = threshold

    def generate_reasoning_chain(
            self,
            question: str,
            context: Optional[List[str]] = None
    ) -> ReasoningChain:
        steps = []
        hallucination_detected = False
        current_text = ""

        while True:
            prompt = self._build_prompt(question, context, current_text)
            output = self.generator.generate(
                prompt,
                config=GenerationConfig(return_logprobs=True)
            )

            step = ReasoningStep(
                text=output.text,
                confidence=self._calculate_confidence(output.logprobs)
            )

            if self.detect_hallucination(step):
                hallucination_detected = True
                break

            steps.append(step)
            current_text += " " + output.text

            if "therefore" in output.text.lower() or len(steps) >= 5:
                break

        return ReasoningChain(
            steps=steps,
            final_answer=self._extract_answer(current_text),
            confidence=np.mean([s.confidence for s in steps]),
            hallucination_detected=hallucination_detected
        )

    def detect_hallucination(self, step: ReasoningStep) -> bool:
        return step.confidence < self.threshold

    def _calculate_confidence(self, logprobs: Optional[List[float]]) -> float:
        if not logprobs:
            return 0.0
        return np.mean([1 - np.exp(p) for p in logprobs])


class AttentionBasedReasoner(BaseReasoner):
    """Attention-weight based reasoning with sophisticated hallucination detection"""

    def __init__(
            self,
            generator: BaseGenerator,
            attention_threshold: float = 0.5,
            entropy_threshold: float = 0.5
    ):
        super().__init__(generator)
        self.attention_threshold = attention_threshold
        self.entropy_threshold = entropy_threshold

    def generate_reasoning_chain(
            self,
            question: str,
            context: Optional[List[str]] = None
    ) -> ReasoningChain:
        steps = []
        current_text = ""
        hallucination_detected = False

        while True:
            prompt = self._build_prompt(question, context, current_text)
            output = self.generator.generate(
                prompt,
                config=GenerationConfig(
                    return_logprobs=True,
                    return_attention=True
                )
            )

            # Calculate attention-based confidence
            attention_confidence = self._calculate_attention_confidence(
                output.attention_weights,
                output.tokens
            )

            # Calculate entropy-based confidence
            entropy_confidence = self._calculate_entropy_confidence(
                output.logprobs
            ) if output.logprobs else 0.0

            step = ReasoningStep(
                text=output.text,
                confidence=(attention_confidence + entropy_confidence) / 2,
                attention_weights=output.attention_weights
            )

            if self.detect_hallucination(step):
                hallucination_detected = True
                if context:
                    # Try to recover with context
                    recovery_step = self._generate_recovery_step(
                        question,
                        context,
                        current_text,
                        step
                    )
                    if recovery_step and not self.detect_hallucination(recovery_step):
                        steps.append(recovery_step)
                        current_text += " " + recovery_step.text
                        continue
                break

            steps.append(step)
            current_text += " " + output.text

            if "therefore" in output.text.lower() or len(steps) >= 5:
                break

        return ReasoningChain(
            steps=steps,
            final_answer=self._extract_answer(current_text),
            confidence=np.mean([s.confidence for s in steps]),
            hallucination_detected=hallucination_detected,
            retrieved_context=context
        )

    def detect_hallucination(self, step: ReasoningStep) -> bool:
        if not step.attention_weights:
            return step.confidence < self.attention_threshold

        attention_scores = self._analyze_attention_patterns(
            step.attention_weights,
            step.text
        )
        return attention_scores.max() > self.attention_threshold

    def _calculate_attention_confidence(
            self,
            attention_weights: Optional[torch.Tensor],
            tokens: Optional[List[str]]
    ) -> float:
        if not attention_weights or not tokens:
            return 0.0

        # Analyze attention flow and patterns
        attention_matrix = attention_weights.mean(dim=0)
        scores = []

        for i, token in enumerate(tokens):
            if i == 0:
                continue
            # Look at attention to previous tokens
            prev_attention = attention_matrix[i, :i]
            if prev_attention.sum() > 0:
                # Normalize and calculate confidence
                prev_attention = prev_attention / prev_attention.sum()
                # High entropy = low confidence
                entropy = -(prev_attention * torch.log(prev_attention + 1e-10)).sum()
                scores.append(1 - entropy / np.log(i + 1))

        return np.mean(scores) if scores else 0.0

    def _calculate_entropy_confidence(self, logprobs: List[float]) -> float:
        probs = np.exp(logprobs)
        entropy = -np.sum(probs * logprobs)
        return 1 - entropy / np.log(len(logprobs))

    def _find_relevant_context(
            self,
            text: str,
            context: List[str],
            attention_weights: Optional[torch.Tensor]
    ) -> Optional[str]:
        """Find most relevant context based on attention patterns.

        Args:
            text: Current text being analyzed
            context: Available context passages
            attention_weights: Attention weights from model

        Returns:
            Most relevant context passage or None
        """
        if not attention_weights or not context:
            return None

        # Convert text and context to tokens for comparison
        text_tokens = self.generator.tokenizer.tokenize(text)
        context_tokens = [
            self.generator.tokenizer.tokenize(c) for c in context
        ]

        # Get attention patterns for text tokens
        text_attention = attention_weights.mean(dim=0)[-len(text_tokens):]

        # Score each context passage
        scores = []
        for ctx_tokens in context_tokens:
            # Look at attention flow between text and context tokens
            overlap_score = self._calculate_token_overlap(
                text_tokens,
                ctx_tokens,
                text_attention
            )
            scores.append(overlap_score)

        # Return most relevant context if score is high enough
        max_score_idx = np.argmax(scores)
        if scores[max_score_idx] > self.attention_threshold:
            return context[max_score_idx]

        return None

    def _calculate_token_overlap(
            self,
            text_tokens: List[str],
            context_tokens: List[str],
            attention_weights: torch.Tensor
    ) -> float:
        """Calculate attention-weighted token overlap score."""
        score = 0.0
        for i, text_token in enumerate(text_tokens):
            if text_token in context_tokens:
                # Weight by attention
                score += attention_weights[i].item()
        return score / len(text_tokens)

    def _build_recovery_prompt(
            self,
            question: str,
            relevant_context: str,
            current_text: str,
            failed_text: str
    ) -> str:
        """Build prompt for recovery from hallucination.

        Args:
            question: Original question
            relevant_context: Retrieved relevant context
            current_text: Current valid reasoning
            failed_text: Text that triggered hallucination

        Returns:
            Recovery prompt
        """
        return f"""Context: {relevant_context}
                   Question: {question}
                   Current reasoning: {current_text}
                   The following reasoning may be incorrect: {failed_text}
                   Let's try to reason about this again using the context:"""


    def _analyze_attention_patterns(
            self,
            attention_weights: torch.Tensor,
            text: str
    ) -> torch.Tensor:
        # Sophisticated attention pattern analysis
        # This is where the magic happens for detecting hallucination
        attention_matrix = attention_weights.mean(dim=0)

        # Look for suspicious patterns like:
        # - Lack of attention to context
        # - Too much self-attention
        # - Inconsistent attention flow

        scores = []
        for i in range(1, attention_matrix.shape[0]):
            # Calculate attention flow metrics
            backward_attention = attention_matrix[i, :i]
            forward_attention = attention_matrix[i:, i]

            # High confidence = consistent attention to relevant context
            # Low confidence = scattered or inconsistent attention

            if backward_attention.sum() > 0:
                backward_attention = backward_attention / backward_attention.sum()
                entropy = -(backward_attention * torch.log(backward_attention + 1e-10)).sum()
                scores.append(1 - entropy / np.log(i + 1))

        return torch.tensor(scores) if scores else torch.tensor([0.0])