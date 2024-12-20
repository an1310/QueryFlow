"""
Base interfaces for RAG and retrieval components.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    documents: List[str]
    document_ids: Optional[List[str]] = None
    scores: Optional[List[float]] = None


class BaseRetriever(ABC):
    """Base interface for retrieval implementations."""

    @abstractmethod
    def retrieve(
            self,
            query: str,
            topk: int = 1,
            **kwargs
    ) -> RetrievalResult:
        """Retrieve relevant documents for query.

        Args:
            query: Query string
            topk: Number of documents to retrieve
            **kwargs: Implementation-specific parameters
        """
        pass

    @abstractmethod
    def index(self, documents: Dict[str, str]) -> None:
        """Index documents for retrieval.

        Args:
            documents: Dictionary mapping document IDs to text
        """
        pass


class BaseRAG(ABC):
    """Base interface for RAG implementations."""

    @abstractmethod
    def generate(
            self,
            question: str,
            demos: List[Dict],
            case: str,
    ) -> str:
        """Generate answer using retrieval-augmented generation.

        Args:
            question: Input question
            demos: List of demonstration examples
            case: Current case/context

        Returns:
            Generated answer
        """
        pass

    @abstractmethod
    def retrieve(
            self,
            query: str,
            topk: Optional[int] = None,
            **kwargs
    ) -> List[str]:
        """Retrieve documents for query.

        Args:
            query: Query string
            topk: Optional override for number of documents
            **kwargs: Additional retrieval parameters
        """
        pass