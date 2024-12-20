"""
Evaluation framework for comparing reasoning approaches.
"""
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np
from collections import defaultdict
import json
import logging
from src.queryflow.rag.reasoning import ReasoningChain, BaseReasoner


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics"""
    accuracy: float
    hallucination_rate: float
    avg_confidence: float
    avg_reasoning_steps: float
    recovery_rate: Optional[float] = None  # Only for attention-based
    context_usage: Optional[float] = None  # How often context was used effectively

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class EvaluationResult:
    """Complete evaluation results"""
    metrics: EvaluationMetrics
    example_outputs: List[Dict[str, Any]]  # Sample reasoning chains
    error_cases: List[Dict[str, Any]]  # Cases where reasoning failed
    metadata: Dict[str, Any]  # Additional info (model, dataset, etc)


class ReasoningEvaluator:
    """Evaluates reasoning approaches on question-answering tasks."""

    def __init__(
            self,
            dataset_name: str,
            answer_extractor: Callable[[str], str],
            correct_answer_checker: Callable[[str, str], bool],
    ):
        """Initialize evaluator.

        Args:
            dataset_name: Name of dataset being evaluated
            answer_extractor: Function to extract answer from reasoning chain
            correct_answer_checker: Function to check if answer is correct
        """
        self.dataset_name = dataset_name
        self.answer_extractor = answer_extractor
        self.correct_answer_checker = correct_answer_checker
        self.logger = logging.getLogger(__name__)

    def evaluate(
            self,
            reasoner: BaseReasoner,
            questions: List[Dict[str, Any]],
            sample_size: Optional[int] = None
    ) -> EvaluationResult:
        """Evaluate a reasoner on a set of questions.

        Args:
            reasoner: Reasoning implementation to evaluate
            questions: List of question dictionaries
            sample_size: Optional number of examples to sample

        Returns:
            Complete evaluation results
        """
        if sample_size:
            questions = np.random.choice(questions, sample_size, replace=False)

        results = []
        errors = []
        correct_count = 0
        hallucination_count = 0
        total_confidence = 0
        total_steps = 0
        recovery_count = 0
        context_usage_count = 0

        for question in questions:
            try:
                # Generate reasoning chain
                chain = reasoner.generate_reasoning_chain(
                    question["question"],
                    context=question.get("context")
                )

                # Extract and validate answer
                predicted_answer = self.answer_extractor(chain.final_answer)
                is_correct = self.correct_answer_checker(
                    predicted_answer,
                    question["answer"]
                )

                # Collect metrics
                if is_correct:
                    correct_count += 1
                if chain.hallucination_detected:
                    hallucination_count += 1
                if chain.retrieved_context:
                    context_usage_count += 1

                total_confidence += chain.confidence
                total_steps += len(chain.steps)

                # Track successful recovery from hallucination
                if chain.hallucination_detected and is_correct:
                    recovery_count += 1

                # Store example output
                result = {
                    "question": question["question"],
                    "predicted_answer": predicted_answer,
                    "correct_answer": question["answer"],
                    "is_correct": is_correct,
                    "confidence": chain.confidence,
                    "num_steps": len(chain.steps),
                    "hallucination_detected": chain.hallucination_detected,
                    "reasoning_steps": [step.text for step in chain.steps]
                }

                results.append(result)

                if not is_correct:
                    errors.append(result)

            except Exception as e:
                self.logger.error(f"Error evaluating question: {question['question']}")
                self.logger.error(str(e))
                errors.append({
                    "question": question["question"],
                    "error": str(e)
                })

        # Calculate metrics
        total = len(questions)
        metrics = EvaluationMetrics(
            accuracy=correct_count / total,
            hallucination_rate=hallucination_count / total,
            avg_confidence=total_confidence / total,
            avg_reasoning_steps=total_steps / total,
            recovery_rate=recovery_count / hallucination_count if hallucination_count > 0 else None,
            context_usage=context_usage_count / total if hasattr(reasoner, '_find_relevant_context') else None
        )

        return EvaluationResult(
            metrics=metrics,
            example_outputs=results[:10],  # Store first 10 examples
            error_cases=errors,
            metadata={
                "dataset": self.dataset_name,
                "reasoner_type": reasoner.__class__.__name__,
                "total_examples": total
            }
        )

    def compare_reasoners(
            self,
            reasoners: List[BaseReasoner],
            questions: List[Dict[str, Any]],
            sample_size: Optional[int] = None
    ) -> Dict[str, EvaluationResult]:
        """Compare multiple reasoning approaches.

        Args:
            reasoners: List of reasoners to compare
            questions: Evaluation questions
            sample_size: Optional sample size

        Returns:
            Dictionary mapping reasoner names to their evaluation results
        """
        results = {}
        for reasoner in reasoners:
            reasoner_name = reasoner.__class__.__name__
            self.logger.info(f"Evaluating {reasoner_name}...")
            results[reasoner_name] = self.evaluate(
                reasoner,
                questions,
                sample_size
            )
        return results

    def save_results(
            self,
            results: Dict[str, EvaluationResult],
            output_path: str
    ):
        """Save evaluation results to file.

        Args:
            results: Evaluation results
            output_path: Path to save results
        """
        output = {
            name: {
                "metrics": result.metrics.to_dict(),
                "examples": result.example_outputs,
                "errors": result.error_cases,
                "metadata": result.metadata
            }
            for name, result in results.items()
        }

        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)


class StrategyQAEvaluator(ReasoningEvaluator):
    """Specific evaluator for StrategyQA yes/no questions."""

    def __init__(self):
        super().__init__(
            dataset_name="StrategyQA",
            answer_extractor=self._extract_yes_no,
            correct_answer_checker=self._check_yes_no
        )

    def _extract_yes_no(self, text: str) -> str:
        """Extract yes/no answer from text."""
        text = text.lower().strip()
        if "yes" in text:
            return "yes"
        if "no" in text:
            return "no"
        return ""

    def _check_yes_no(self, predicted: str, actual: str) -> bool:
        """Check if yes/no answer is correct."""
        return predicted.lower() == actual.lower()