"""
Exact mapping of existing prompt construction and query formulation strategies.
"""
from typing import List, Optional, Dict, Any
from enum import Enum
import spacy

nlp = spacy.load("en_core_web_sm")

class QueryFormulation(Enum):
    DIRECT = "direct"
    CURRENT = "current"
    CURRENT_WO_WRONG = "current_wo_wrong"
    FORWARD_ALL = "forward_all"
    LAST_SENTENCE = "last_sentence"
    LAST_N_TOKENS = "last_n_tokens"
    REAL_WORDS = "real_words"


class BasePromptBuilder:
    """Maps exact prompt building from original implementation."""

    def build_demo_prompt(self, demo: List[Dict]) -> str:
        """Build prompt from demonstration cases."""
        return "".join([d["case"] + "\n" for d in demo])

    def get_last_sentence(self, text: str) -> str:
        """Extract last sentence from text."""
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else ""

    def get_top_sentence(self, text: str) -> str:
        """Extract first sentence from text."""
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""


class SingleRAGPrompts(BasePromptBuilder):

    def build_prompt(
            self,
            question: str,
            demo: List[Dict],
            case: str,
            docs: Optional[List[str]] = None
    ) -> str:
        prompt = self.build_demo_prompt(demo)

        if docs:
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i + 1}] {doc}\n"

        prompt += "Answer in the same format as before.\n"
        prompt += case
        return prompt


class AttentionWeightRAGPrompts(BasePromptBuilder):

    def build_prompt(
            self,
            question: str,
            demo: List[Dict],
            case: str,
            text: str = "",
            docs: Optional[List[str]] = None
    ) -> str:
        prompt = self.build_demo_prompt(demo)

        if docs:
            prompt += "Context:\n"
            for i, doc in enumerate(docs):
                prompt += f"[{i + 1}] {doc}\n"
            prompt += "Answer in the same format as before.\n"

        tmp_li = [case, text]
        prompt += " ".join(s for s in tmp_li if len(s) > 0)
        return prompt

    def formulate_query(
            self,
            formulation: QueryFormulation,
            question: str,
            text: str,
            ptext: str,
            curr_tokens: List[str],
            curr_hit: List[int],
            tokenizer: Any,
            retrieve_keep_top_k: Optional[int] = None
    ) -> str:
        """Exact mapping of query formulation strategies."""

        if formulation == QueryFormulation.CURRENT:
            return " ".join(curr_tokens)

        elif formulation == QueryFormulation.CURRENT_WO_WRONG:
            return " ".join(
                curr_tokens[i] if curr_hit[i] == 0 else ""
                for i in range(len(curr_tokens))
            )

        elif formulation == QueryFormulation.FORWARD_ALL:
            forward_all = [question, text, ptext]
            return " ".join(s for s in forward_all if len(s) > 0)

        elif formulation == QueryFormulation.LAST_SENTENCE:
            forward_all = [question, text, ptext]
            combined = " ".join(s for s in forward_all if len(s) > 0)
            return self.get_last_sentence(combined)

        elif formulation == QueryFormulation.LAST_N_TOKENS:
            assert retrieve_keep_top_k is not None
            forward_all = [question, text, ptext]
            combined = " ".join(s for s in forward_all if len(s) > 0)
            return self.fetch_last_n_tokens(combined, retrieve_keep_top_k, tokenizer)

        elif formulation == QueryFormulation.REAL_WORDS:
            return self.real_words_query(
                prev_text=question + " " + text + " " + ptext,
                curr_tokens=curr_tokens,
                curr_hit=curr_hit
            )

        elif formulation == QueryFormulation.DIRECT:
            return question

        else:
            raise NotImplementedError(f"Unknown query formulation: {formulation}")

    def fetch_last_n_tokens(
            self,
            text: str,
            num: int,
            tokenizer: Any
    ) -> str:
        """Get last N tokens from text."""
        tokens = tokenizer.tokenize(text)
        if num >= len(tokens):
            return text
        last_n_tokens = tokens[-num:]
        return " ".join(last_n_tokens)

    def real_words_query(
            self,
            prev_text: str,
            curr_tokens: List[str],
            curr_hit: List[int]
    ) -> str:
        """Exact mapping of real_words query formulation."""
        doc = nlp(prev_text)
        real_words = set(token.text for token in doc if token.pos_ in
                         ['NOUN', 'ADJ', 'VERB', 'PROPN', 'NUM'])

        def match(token: str) -> bool:
            return any(word in token for word in real_words)

        retrieved_words = []
        for i, (token, hit) in enumerate(zip(curr_tokens, curr_hit)):
            if not hit and match(token):
                retrieved_words.append(token)

        return " ".join(retrieved_words)