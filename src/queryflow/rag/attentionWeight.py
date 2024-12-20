"""
RAG implementation using attention-based generation and retrieval.
"""
from typing import List, Optional, Dict, Any
from src.queryflow.rag.ragInterface import BaseRAG
from src.queryflow.rag.prompt import AttentionWeightRAGPrompts, QueryFormulation
from src.queryflow.generator.attentionGenerator import MonitoredGenerator, GenerationResult


class AttentionWeightRAG(BaseRAG):
    """
    Attention-weighted RAG implementation.
    Separates retrieval logic from generation while maintaining original functionality.
    """


    def __init__(
            self,
            generator: MonitoredGenerator,
            retriever: Any,  # Your retriever interface
            prompt_builder: Optional[AttentionWeightRAGPrompts] = None,
            retriever_topk: int = 3,
            query_formulation: str = "current",
    ):
        self.generator = generator
        self.retriever = retriever
        self.prompt_builder = prompt_builder or AttentionWeightRAGPrompts()
        self.retriever_topk = retriever_topk
        self.query_formulation = QueryFormulation(query_formulation)

    def generate(
            self,
            question: str,
            demos: List[Dict],
            case: str,
    ) -> str:
        """
        Generate answer using attention-based RAG.
        Maps the core logic from the original implementation.
        """
        text = ""

        while True:
            # Build initial prompt
            prompt = self.prompt_builder.build_prompt(
                question=question,
                demo=demos,
                case=case,
                text=text
            )

            # Generate with monitoring
            generation, all_results = self.generator.generate_until_complete(prompt)

            # Early exit if no hallucination
            if not any(r.is_hallucination for r in all_results):
                return generation

            # Get last result with hallucination
            result = next(r for r in reversed(all_results) if r.is_hallucination)

            # Formulate retrieval query
            query = self.prompt_builder.formulate_query(
                formulation=self.query_formulation,
                question=question,
                text=text,
                ptext=result.previous_text,
                curr_tokens=result.current_tokens,
                curr_hit=result.current_hits,
                tokenizer=self.generator.generator.tokenizer
            )

            # Retrieve relevant documents
            docs = self.retriever.retrieve(
                query,
                topk=self.retriever_topk
            )

            # Build prompt with retrieved context
            prompt = self.prompt_builder.build_prompt(
                question=question,
                demo=demos,
                case=case,
                text=text,
                docs=docs
            )

            # Generate recovery
            recovery_result = self.generator.generator.generate_with_monitoring(prompt)
            if not recovery_result.is_hallucination:
                text = " ".join([
                    t.strip() for t in [text, result.previous_text, recovery_result.text]
                    if t.strip()
                ])
                continue

            # If recovery failed, just use what we have
            return text