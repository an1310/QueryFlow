"""
Retriever implementations for question answering system.
"""
from typing import List, Dict, Tuple, Optional, Union
from abc import ABC, abstractmethod
import os
import time
import uuid
import logging
import numpy as np
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from dataclasses import dataclass
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search
from beir.retrieval.search.lexical.elastic_search import ElasticSearch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Container for retrieval results"""
    document_ids: np.ndarray  # Shape: (batch_size, top_k)
    documents: np.ndarray  # Shape: (batch_size, top_k)
    scores: Optional[np.ndarray] = None  # Shape: (batch_size, top_k)


class BaseRetriever(ABC):
    """Base class for all retriever implementations."""

    def __init__(self, tokenizer: Optional[AutoTokenizer] = None):
        self.tokenizer = tokenizer

    @abstractmethod
    def retrieve(
            self,
            queries: List[str],
            topk: int = 1,
            **kwargs
    ) -> RetrievalResult:
        """Retrieve relevant documents for given queries.

        Args:
            queries: List of query strings
            topk: Number of documents to retrieve per query
            **kwargs: Additional retriever-specific arguments

        Returns:
            RetrievalResult containing document IDs and contents
        """
        pass

    @staticmethod
    def _get_random_doc_id() -> str:
        """Generate a random document ID."""
        return f'_{uuid.uuid4()}'


class BM25Retriever(BaseRetriever):
    """BM25-based retrieval using Elasticsearch backend."""

    def __init__(
            self,
            tokenizer: Optional[AutoTokenizer] = None,
            index_name: Optional[str] = None,
            hostname: str = 'localhost',
            initialize: bool = False,
            number_of_shards: int = 1,
    ):
        super().__init__(tokenizer)
        self.max_ret_topk = 1000

        # Initialize Elasticsearch retriever
        self.retriever = EvaluateRetrieval(
            BM25Search(
                index_name=index_name,
                hostname=hostname,
                initialize=initialize,
                number_of_shards=number_of_shards
            ),
            k_values=[self.max_ret_topk]
        )

    def retrieve(
            self,
            queries: List[str],
            topk: int = 1,
            max_query_length: Optional[int] = None,
            **kwargs
    ) -> RetrievalResult:
        """Retrieve documents using BM25 ranking.

        Args:
            queries: List of query strings
            topk: Number of documents to retrieve per query
            max_query_length: Maximum length to truncate queries
            **kwargs: Additional arguments passed to underlying retriever

        Returns:
            RetrievalResult containing document IDs and contents
        """
        if topk > self.max_ret_topk:
            raise ValueError(f"topk ({topk}) cannot exceed max_ret_topk ({self.max_ret_topk})")

        batch_size = len(queries)

        # Truncate queries if needed
        if max_query_length and self.tokenizer:
            queries = self._truncate_queries(queries, max_query_length)

        # Retrieve documents
        results = self.retriever.retrieve(
            corpus=None,
            queries=dict(enumerate(queries)),
            **kwargs
        )

        # Process results
        docids, docs = [], []
        for qid in range(batch_size):
            query_docids, query_docs = [], []

            if qid in results:
                for did, (score, text) in results[qid].items():
                    query_docids.append(did)
                    query_docs.append(text)
                    if len(query_docids) >= topk:
                        break

            # Pad with dummy docs if needed
            if len(query_docids) < topk:
                query_docids.extend(self._get_random_doc_id() for _ in range(topk - len(query_docids)))
                query_docs.extend([''] * (topk - len(query_docs)))

            docids.extend(query_docids)
            docs.extend(query_docs)

        return RetrievalResult(
            document_ids=np.array(docids).reshape(batch_size, topk),
            documents=np.array(docs).reshape(batch_size, topk)
        )

    def _truncate_queries(self, queries: List[str], max_length: int) -> List[str]:
        """Truncate queries to specified maximum length."""
        original_padding_side = self.tokenizer.padding_side
        original_truncation_side = self.tokenizer.truncation_side

        try:
            # Truncate/pad on the left side
            self.tokenizer.padding_side = 'left'
            self.tokenizer.truncation_side = 'left'

            tokenized = self.tokenizer(
                queries,
                truncation=True,
                padding=True,
                max_length=max_length,
                add_special_tokens=False,
                return_tensors='pt'
            )['input_ids']

            return self.tokenizer.batch_decode(tokenized, skip_special_tokens=True)

        finally:
            # Restore original settings
            self.tokenizer.padding_side = original_padding_side
            self.tokenizer.truncation_side = original_truncation_side


class SGPTRetriever(BaseRetriever):
    """Dense retrieval using SGPT embeddings."""

    # IDs that SGPT cannot encode
    CANNOT_ENCODE_IDS = {
        6799132, 6799133, 6799134, 6799135, 6799136, 6799137, 6799138, 6799139,
        8374206, 8374223, 9411956, 9885952, 11795988, 11893344, 12988125,
        14919659, 16890347, 16898508
    }

    def __init__(
            self,
            model_name_or_path: str,
            sgpt_encode_file_path: str,
            passage_file: str,
            **kwargs
    ):
        super().__init__()
        self.model_name = model_name_or_path

        logger.info(f"Loading SGPT model from {model_name_or_path}")
        self._initialize_model(model_name_or_path)

        logger.info("Building SGPT indexes")
        self.p_reps = self._load_passage_representations(sgpt_encode_file_path)
        self.docs = self._load_passages(passage_file)

    def _initialize_model(self, model_name: str):
        """Initialize SGPT model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name, device_map="auto")
        self.model.eval()

        # Cache special tokens
        self.special_tokens = {
            'que_bos': self.tokenizer.encode("[", add_special_tokens=False)[0],
            'que_eos': self.tokenizer.encode("]", add_special_tokens=False)[0],
            'doc_bos': self.tokenizer.encode("{", add_special_tokens=False)[0],
            'doc_eos': self.tokenizer.encode("}", add_special_tokens=False)[0]
        }

    def _load_passage_representations(self, encode_file_path: str) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """Load pre-computed passage representations."""
        p_reps = []
        dir_names = sorted(os.listdir(encode_file_path))

        # Count number of split parts
        split_parts = sum(1 for i in range(1, len(dir_names) + 1)
                          if any(d.startswith(f'{i}_') for d in dir_names))

        for i in range(split_parts):
            filenames = [f"{i}_{j}.pt" for j in range(len(dir_names))
                         if f"{i}_{j}.pt" in dir_names]

            for filename in filenames:
                tp = torch.load(os.path.join(encode_file_path, filename))
                sz = tp.shape[0] // 2

                # Split tensor and compute norms
                tp1, tp2 = tp[:sz, :], tp[sz:, :]
                p_reps.extend([
                    (tp1.cuda(i), self._compute_norm(tp1).cuda(i)),
                    (tp2.cuda(i), self._compute_norm(tp2).cuda(i))
                ])

        return p_reps

    @staticmethod
    def _compute_norm(matrix: torch.Tensor) -> torch.Tensor:
        """Compute L2 norm, replacing zeros with ones to avoid division by zero."""
        norm = matrix.norm(dim=1)
        return torch.where(norm == 0, torch.tensor(1.0), norm).view(-1, 1)

    def _load_passages(self, passage_file: str) -> List[str]:
        """Load passages from TSV file."""
        df = pd.read_csv(passage_file, delimiter='\t')
        return list(df['text'])

    def retrieve(
            self,
            queries: List[str],
            topk: int = 1,
            **kwargs
    ) -> RetrievalResult:
        """Retrieve passages using SGPT embeddings.

        Args:
            queries: List of query strings
            topk: Number of passages to retrieve per query
            **kwargs: Additional retriever-specific arguments

        Returns:
            RetrievalResult containing retrieved passages
        """
        # Encode queries
        q_reps = self._encode_queries(queries)
        q_reps_trans = torch.transpose(q_reps, 0, 1)

        # Compute similarities and get top-k
        topk_values_list, topk_indices_list = [], []
        prev_count = 0

        for p_rep, p_rep_norm in self.p_reps:
            # Compute normalized similarities
            sim = p_rep @ q_reps_trans.to(p_rep.device)
            sim = sim / p_rep_norm

            # Get top-k for this batch
            topk_values, topk_indices = torch.topk(sim, k=topk, dim=0)

            topk_values_list.append(topk_values.cpu())
            topk_indices_list.append(topk_indices.cpu() + prev_count)
            prev_count += p_rep.shape[0]

        # Get global top-k
        all_topk_values = torch.cat(topk_values_list, dim=0)
        global_topk_values, global_topk_indices = torch.topk(all_topk_values, k=topk, dim=0)

        # Gather passages
        passages = []
        for qid in range(len(queries)):
            query_passages = []
            for j in range(topk):
                idx = global_topk_indices[j][qid].item()
                fid, rk = idx // topk, idx % topk
                passage = self.docs[topk_indices_list[fid][rk][qid]]
                query_passages.append(passage)
            passages.append(query_passages)

        return RetrievalResult(
            document_ids=np.array([[self._get_random_doc_id() for _ in range(topk)]
                                   for _ in range(len(queries))]),
            documents=np.array(passages),
            scores=global_topk_values.numpy()
        )

    def _encode_queries(self, queries: List[str]) -> torch.Tensor:
        """Encode queries using SGPT model."""
        batch_tokens = self._tokenize_with_special_tokens(queries, is_query=True)

        with torch.no_grad():
            last_hidden_state = self.model(
                **batch_tokens,
                output_hidden_states=True,
                return_dict=True
            ).last_hidden_state

        return self._weighted_mean_pooling(last_hidden_state, batch_tokens["attention_mask"])

    def _tokenize_with_special_tokens(
            self,
            texts: List[str],
            is_query: bool
    ) -> Dict[str, torch.Tensor]:
        """Tokenize texts with special tokens for queries or documents."""
        # Initial tokenization
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)

        # Add special brackets
        bos = self.special_tokens['que_bos' if is_query else 'doc_bos']
        eos = self.special_tokens['que_eos' if is_query else 'doc_eos']

        for seq, att in zip(batch_tokens["input_ids"], batch_tokens["attention_mask"]):
            seq.insert(0, bos)
            seq.append(eos)
            att.extend([1, 1])

        # Add padding
        return self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")

    def _weighted_mean_pooling(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute weighted mean of token embeddings."""
        # Create position-based weights
        weights = (torch.arange(start=1, end=hidden_states.shape[1] + 1)
                   .unsqueeze(0)
                   .unsqueeze(-1)
                   .expand(hidden_states.size())
                   .float()
                   .to(hidden_states.device))

        # Expand attention mask
        input_mask_expanded = (attention_mask
                               .unsqueeze(-1)
                               .expand(hidden_states.size())
                               .float())

        # Compute weighted sum
        sum_embeddings = torch.sum(hidden_states * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        return sum_embeddings / sum_mask


# For backward compatibility
BM25 = BM25Retriever
SGPT = SGPTRetriever