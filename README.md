🚀 Enterprise Policy Hybrid Search Assistant
-----------------------------------------

A hybrid search system that combines keyword precision and semantic similarity using transformer-based embeddings.

This project demonstrates a production-style retrieval architecture with chunk-level semantic search, evaluation modes, metadata filtering, and explainable ranking.

🔍 Features
--------
	•	Hybrid search (Keyword + Semantic)
	•	Chunk-level embeddings
	•	SentenceTransformer (all-mpnet-base-v2)
	•	Evaluation modes:
	•	Keyword Only
	•	Semantic Only
	•	Hybrid
	•	Adjustable keyword weight
	•	Metadata filtering (Region)
	•	Score visualization (stacked contribution chart)
	•	Chunk preview for matched content
	•	Fully local (no API dependency)

🏗 Architecture Overview
---------------------

Startup Phase
	•	Load policies.json
	•	Chunk policy content
	•	Compute embedding per chunk
	•	Cache tokens for keyword search

Query Phase
	1.	Tokenize query (lexical layer)
	2.	Generate query embedding (semantic layer)
	3.	For each policy:
	•	Compute keyword overlap
	•	Compute max similarity across chunks
	4.	Apply weighted fusion
	5.	Rank and display results

📊 Hybrid Scoring
----------------

Final score is computed as:

final_score =
    keyword_weight * normalized_keyword_score
  + vector_weight  * semantic_similarity

  Where:
	•	Keyword score = token overlap
	•	Semantic score = cosine similarity
	•	Weights are user-controlled

🧠 Embedding Model

Uses:
SentenceTransformer("all-mpnet-base-v2")

Local transformer model trained for semantic similarity.

🧩 Chunking Strategy

	•	Character-based chunking
	•	500 character chunks
	•	100 character overlap
	•	Max chunk similarity determines policy relevance

🛠 Running the App

1.	Install dependencies:
   pip install -r requirements.txt

2.	Run:
   streamlit run app.py

🎯 What This Demonstrates

	•	Hybrid retrieval architecture
	•	Semantic embeddings in search
	•	Chunk-level precision
	•	Evaluation of lexical vs semantic search
	•	Explainable ranking behavior

🏗 Executive Architecture – Hybrid Search System

## Architecture Diagram

```mermaid
flowchart TD

A[User Interface] --> B[UI Layer (app.py)]
B --> C[Hybrid Search Engine]

C --> D[Keyword Matching]
C --> E[Semantic Embedding Search]

E --> F[Chunk-Level Similarity]
D --> G[Fusion Layer]
F --> G

G --> H[Ranked Results]
H --> I[Chunk Preview + Chart]

subgraph Startup Phase
S1[Load Policies]
S2[Chunk Content]
S3[Compute Chunk Embeddings]
S1 --> S2 --> S3
end

Why Hybrid Search?
------------------

Keyword search provides precision.
Semantic search provides contextual understanding.
Hybrid search combines both to balance exact matching with conceptual relevance.

This project demonstrates how enterprise search systems are architected beyond simple full-text search.
