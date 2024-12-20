"""
This module implements attention-based hallucination detection for language models.

The key idea is to use attention patterns to determine when a model is "making things up"
vs when it's properly grounding its output in given context. This is done by:
1. Analyzing how tokens attend to context and each other
2. Looking for suspicious patterns in attention weights
3. Using entropy and logprobs as additional signals
4. Validating against known real-world entities
"""
from typing import List, Tuple, Optional, Dict, Any
import torch
import numpy as np
from math import exp
from scipy.special import softmax


class AttentionAnalyzer:
    """
    Analyzes attention patterns to detect hallucination in generated text.

    The analyzer uses multiple signals:
    - Token-level attention patterns
    - Word-level attention aggregation
    - Entropy of token predictions
    - Log probabilities of sequences
    - Entity and real word validation
    """

    def __init__(
            self,
            hallucination_threshold: float,
            space_token: str
    ):
        """
        Args:
            hallucination_threshold: Threshold for flagging suspicious attention patterns
            space_token: Token used to represent spaces for the specific tokenizer
        """
        self.hallucination_threshold = hallucination_threshold
        self.space_token = space_token

    def analyze_generation(
            self,
            text: str,
            tokens: List[str],
            attentions: List[float],
            weight: List[float]
    ) -> Tuple[bool, str, Optional[List[str]], Optional[List[int]]]:
        """
        Analyze generated text for signs of hallucination.

        This works by:
        1. Breaking text into sentences
        2. For each sentence:
            - Map tokens to sentence spans
            - Calculate normalized attention values
            - Apply weighting (entropy or logprobs)
            - Check against threshold
            - Optionally validate against known entities
        3. Return hallucination status and details

        Args:
            text: Generated text to analyze
            tokens: Individual tokens from generation
            attentions: Attention weights for tokens
            weight: Additional weights (entropy or -logprob)

        Returns:
            Tuple of:
            - Whether hallucination was detected
            - Valid text up to hallucination point
            - Tokens in suspicious span
            - Binary flags for suspicious tokens
        """
        # Split into sentences for analysis
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]

        tid = 0  # Track current token position
        for sid, sent in enumerate(sentences):
            # Find token span for this sentence
            tl, tr = tid, tid  # Token span (left, right)
            if sid == len(sentences) - 1:
                # Last sentence gets all remaining tokens
                tl, tr = tid, len(tokens)
            else:
                # Find token span that matches this sentence
                for i in range(tid + 1, len(tokens)):
                    seq = " ".join(tokens[tl:i])
                    if sent in seq:
                        tr = i
                        break
                tid = tr

            # Calculate attention-based hallucination score
            attns = attentions[tl:tr]
            attns = np.array(attns) / sum(attns)  # Normalize
            # Multiply by weight and span length for final score
            value = [attns[i - tl] * weight[i] * (tr - tl) for i in range(tl, tr)]
            # Flag tokens that exceed threshold
            thres = [1 if v > self.hallucination_threshold else 0 for v in value]

            if 1 in thres:  # Hallucination detected
                # Optionally validate against known entities
                if hasattr(self, 'check_real_words') and self.check_real_words:
                    doc = nlp(sent)
                    # Get known entity types
                    real_words = set(token.text for token in doc if token.pos_ in
                                     ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

                    def match(tok):
                        """Check if token appears in known entities"""
                        return any(word in tok for word in real_words)

                    # Clear flag for tokens that match known entities
                    for i in range(len(thres)):
                        if not match(tokens[tl + i]):
                            thres[i] = 0

                # Return valid text up to this point
                prev = "" if sid == 0 else " ".join(sentences[:sid])
                return True, prev, tokens[tl:tr], thres

        return False, text, None, None

    def process_attention_patterns(
            self,
            input_ids: torch.Tensor,
            tokens_tmp: List[str],
            atten_tmp: torch.Tensor,
            generated_tokens: torch.Tensor,
    ) -> Tuple[List[str], List[float]]:
        """
        Process raw attention patterns into analyzable token sequences.

        This involves:
        1. Merging subword tokens back into words
        2. Aggregating attention weights for merged tokens
        3. Handling special tokens and spaces

        Args:
            input_ids: Token IDs from generation
            tokens_tmp: Raw tokens
            atten_tmp: Raw attention weights
            generated_tokens: Tokens produced in generation

        Returns:
            Lists of:
            - Merged token sequences
            - Aggregated attention values
        """
        # Track token merge ranges
        range_ = []
        for i, t in enumerate(tokens_tmp):
            # Start new range if:
            # - First token
            # - Token starts with space
            # - Token is carriage return
            # - Previous token was end of sequence
            if (i == 0 or
                    t.startswith(self.space_token) or
                    input_ids[0][i] == 13 or
                    tokens_tmp[i - 1] == '</s>'):
                range_.append([i, i])
            else:
                # Extend current merge range
                range_[-1][-1] += 1

        # Build sequence list and attention values
        seqlist = []
        attns = []
        mean_atten = torch.mean(atten_tmp, dim=0)

        for r in range_:
            # Merge tokens and clean up spaces
            tokenseq = "".join(tokens_tmp[r[0]: r[1] + 1]).replace(self.space_token, "")
            # Sum attention over merged span
            value = sum(mean_atten[r[0]: r[1] + 1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        return seqlist, attns

    def calculate_entropy(
            self,
            outputs_scores: List[torch.Tensor],
            range_: List[List[int]]
    ) -> List[float]:
        """
        Calculate entropy for token sequences.

        Higher entropy indicates more uncertainty in the model's predictions.

        Args:
            outputs_scores: Raw scores from model output
            range_: Token merge ranges

        Returns:
            Entropy values for each merged sequence
        """
        # Move to CPU and convert to probabilities
        tmp = [v.cpu() for v in outputs_scores]
        softmax_probs = softmax(tmp, axis=-1)

        # Calculate entropy
        entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
        entropies = [v[0] for v in entropies]

        # Average entropy over merged sequences
        seqentropies = []
        for r in range_:
            entropyseq = sum(entropies[r[0]:r[1] + 1]) / (r[1] - r[0] + 1)
            seqentropies.append(entropyseq)

        return seqentropies

    def calculate_logprobs(
            self,
            logprobs: List[float],
            range_: List[List[int]]
    ) -> List[float]:
        """
        Calculate average log probabilities for merged token sequences.

        Lower values indicate the model is less confident about the generation.

        Args:
            logprobs: Token log probabilities
            range_: Token merge ranges

        Returns:
            Average log prob for each merged sequence
        """
        seqlogprobs = []
        for r in range_:
            # Average log prob over merge range
            logprobseq = sum(logprobs[r[0]:r[1] + 1]) / (r[1] - r[0] + 1)
            seqlogprobs.append(logprobseq)
        return seqlogprobs

    def get_attention_stats(
            self,
            attention: torch.Tensor,
            solver: str = "max"
    ) -> torch.Tensor:
        """
        Calculate aggregate attention statistics.

        Supports different aggregation methods:
        - max: Maximum attention per position (most attended token)
        - avg: Average attention, normalized by position
        - last_token: Attention pattern of final token

        Args:
            attention: Raw attention weights
            solver: Aggregation method

        Returns:
            Aggregated attention weights
        """
        if solver == "max":
            # Take max attention per position
            mean_attention, _ = torch.max(attention, dim=1)
            mean_attention = torch.mean(mean_attention, dim=0)
        elif solver == "avg":
            # Average attention, normalized by distance
            mean_attention = torch.sum(attention, dim=1)
            mean_attention = torch.mean(mean_attention, dim=0)
            for i in range(mean_attention.shape[0]):
                mean_attention[i] /= (mean_attention.shape[0] - i)
        elif solver == "last_token":
            # Just use final token's attention
            mean_attention = torch.mean(attention[:, -1], dim=0)
        else:
            raise NotImplementedError

        return mean_attention