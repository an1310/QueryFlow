# QueryFlow: A Dynamic Retrieval Augmented Generation Framework

QueryFlow is an experimental Python framework designed to enhance the text generation capabilities of Large Language Models (LLMs) by intelligently and dynamically deciding **when** and **what** to retrieve during the generation process. It aims to improve the relevance, accuracy, and coherence of generated text by addressing the information needs of the LLM in real-time.

## Overview

Traditional Retrieval Augmented Generation (RAG) methods often retrieve information at fixed points (e.g., only before generation starts). QueryFlow explores more dynamic approaches, where the generation process itself is monitored, and retrieval is triggered only when the model shows signs of needing external information (e.g., uncertainty, potential hallucination). Furthermore, the queries for retrieval are formulated dynamically based on the current generation context.

This project was developed as a work-in-progress (WIP) to investigate these advanced RAG techniques.

## Core Components & Approach

QueryFlow is built upon two main conceptual components, with supporting modules for generation and reasoning:

1.  **RIND (Real-time Information Needs Detection):**
    * **Goal:** To determine the optimal moment to activate the retrieval module by assessing the LLM's uncertainty or the quality of its current generation.
    * **Implementation:**
        * The core logic resides in `src/queryflow/attention/attention.py` within the `AttentionAnalyzer` class.
        * This analyzer evaluates generated tokens using attention patterns, token entropy, and log probabilities to identify segments that might be hallucinations or indicate low model confidence.
        * Generation is monitored in real-time by `src/queryflow/generator/attentionGenerator.py`, which uses the `AttentionAnalyzer` to flag problematic parts of the text as it's being generated. If a hallucination is detected above a certain `hallucination_threshold`, this signals a need for retrieval.

2.  **QFS (Query Formulation based on Self-attention):**
    * **Goal:** To craft effective and contextually relevant retrieval queries dynamically when RIND signals a need for information.
    * **Implementation:**
        * When RIND detects an issue, the `src/queryflow/rag/attentionWeight.py` module takes over.
        * It uses a `prompt_builder` (along with `QueryFormulation` strategies) to dynamically create a new search query. This query is informed by the context of the problematic generation, including the text generated so far, the specific tokens flagged by RIND, and the original question.
        * The system leverages attention weights and contextual cues to formulate a targeted query for the retriever.

3.  **Iterative Refinement & Reasoning:**
    * The framework supports iterative generation, where text is built incrementally, monitored at each step, and refined through retrieval if necessary.
    * The concepts are extended in `src/queryflow/rag/reasoning.py`, which introduces `AttentionBasedReasoner` capable of constructing multi-step reasoning chains. Each step in the chain can be evaluated for hallucinations using attention-based mechanisms, and the system can attempt recovery or targeted context retrieval to improve the reasoning process.

## Key Features

* **Dynamic Retrieval Triggering:** Actively decides *when* to retrieve based on the LLM's real-time information needs, signaled by attention patterns, entropy, and token probabilities.
* **Context-Aware Query Formulation:** Dynamically crafts retrieval queries based on the current generation context and points of uncertainty/hallucination.
* **Attention-Based Hallucination Detection:** Leverages attention mechanisms, alongside other signals, to identify and potentially mitigate model fabrications.
* **Iterative Generation & Recovery:** Supports step-by-step text generation with the ability to retrieve and attempt to recover from detected issues.
* **Support for Multi-step Reasoning:** Includes modules for building reasoning chains where each step is monitored and can trigger dynamic information retrieval.
* **Lightweight Integration Philosophy:** Designed to be incorporated with Transformer-based LLMs without requiring model retraining or fine-tuning for the core LLM.

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers (for LLMs and tokenizers)
* Elasticsearch (for indexing and retrieving from knowledge bases like Wikipedia)
* NumPy, SciPy
