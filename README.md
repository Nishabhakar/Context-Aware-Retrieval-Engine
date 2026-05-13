# Context-Aware Retrieval Engine

This project demonstrates a simple local RAG (Retrieval-Augmented Generation) pipeline using:

- sentence-transformers for embeddings
- FAISS for vector similarity search
- Query Expansion for improved retrieval
- Python-based benchmarking

The project compares:

1. Raw Vector Search
2. AI-Enhanced Query Expansion Search

---

# Project Structure

```bash
project-folder/
│
├── main.py
├── test.py
├── data.txt
├── requirements.txt
└── README.md
```

---

# Step 1: Create Virtual Environment

## Windows

```bash
python -m venv venv
```

Activate virtual environment:

```bash
venv\Scripts\activate
```

---

## Mac/Linux

```bash
python3 -m venv venv
```

Activate virtual environment:

```bash
source venv/bin/activate
```

---

# Step 2: Install Dependencies

Run:

```bash
pip install -r requirements.txt
```

---

# Step 3: Add/modify Dataset

Create a file named:

```bash
data.txt
```

Add technical paragraphs like:

```txt
The platform uses Kubernetes autoscaling to manage sudden spikes in traffic.

Redis caching reduces database pressure during high concurrency.

Circuit breakers prevent cascading failures across microservices.

Kafka buffers asynchronous workloads during heavy ingestion bursts.

Load balancers distribute requests across stateless nodes.
```

Each line is treated as one searchable document chunk.

---

# Step 4: Run Manual Query Testing

Run:

```bash
python main.py
```

You will see:

```bash
Enter your query (type 'exit' to quit):
```

Example:

```bash
Enter your query:
How does the system handle peak load?
```

The application will show:

- Strategy A results (Raw Search)
- Strategy B results (Expanded Query Search)

---

# Example Output

```bash
QUERY:
How does the system handle peak load?

STRATEGY A : RAW VECTOR SEARCH

Top Result 1
Similarity Score: 0.7123
Load balancers distribute requests across nodes.

STRATEGY B : AI-ENHANCED RETRIEVAL

Expanded Query:
traffic spikes autoscaling load balancing scalability

Top Result 1
Similarity Score: 0.8432
Kubernetes autoscaling manages traffic spikes.
```

---

# Step 5: Run Automated Benchmark Testing

Run:

```bash
python test.py
```

This file automatically tests predefined benchmark queries.

---

# What test_retrieval.py Does

It:

- Loads the documents
- Creates embeddings
- Stores vectors in FAISS
- Runs multiple benchmark queries
- Compares:
  - Raw Search
  - Query Expansion Search
- Saves output to timestamped JSON files

Example generated file:

```bash
retrieval_results_20260513_183000.json
```

---

# Example Benchmark Queries

```python
queries = [

    "How does the system handle peak load?",

    "How are failures isolated in distributed systems?",

    "How is database latency reduced?"
]
```

---

# Output JSON Format

Example:

```json
[
    {
        "query": "How does the system handle peak load?",
        "strategy_a_raw_search": [...],
        "expanded_query": "...",
        "strategy_b_expanded_search": [...]
    }
]
```

---

# Similarity Metric Used

This project uses:

## Cosine Similarity

Why?

Because semantic embeddings work better when comparing vector direction instead of vector distance.

FAISS uses:

```python
faiss.IndexFlatIP
```

with normalized embeddings to simulate cosine similarity.

---

# Query Expansion Logic

The project improves retrieval by expanding user queries.

Example:

Input Query:

```text
How does the system handle peak load?
```

Expanded Query:

```text
traffic spikes autoscaling load balancing high concurrency scalability
```

This helps retrieve more semantically relevant documents.

---

# Technologies Used

| Technology | Purpose |
|---|---|
| sentence-transformers | Generate embeddings |
| FAISS | Vector similarity search |
| NumPy | Numerical operations |
| Python | Main implementation |

---

# Install Requirements Manually (Optional)

If requirements.txt is not used:

```bash
pip install sentence-transformers
pip install faiss-cpu
pip install numpy
pip install torch
```

---

# Exit Manual Query Mode

Type:

```bash
exit
```

to stop the application.

---

# Notes

- Fully local implementation
- No cloud dependency
- No OpenAI API required
- No Vertex AI account required
- Query expansion is mocked for demonstration purposes

---

# Future Improvements

Possible enhancements:

- Chunking large documents
- Hybrid BM25 + Vector Search
- Reranking models
- Real Vertex AI Embeddings
- ChromaDB integration
- REST API using FastAPI
- Web UI using Streamlit

---


# Production Migration to Vertex AI

This project currently uses:

- sentence-transformers for embeddings
- FAISS for vector storage

In production, this can be migrated to Google Cloud Vertex AI.

## Local → Production Mapping

| Local Component | Vertex AI Equivalent |
|---|---|
| sentence-transformers | Vertex AI TextEmbeddingModel |
| FAISS | Vertex AI Matching Engine |
| Mock Query Expansion | Gemini / GenerativeModel |
| Local files | Cloud Storage / BigQuery |

---

## Production Flow

1. Documents are stored in Cloud Storage or BigQuery.
2. Vertex AI generates embeddings.
3. Embeddings are indexed in Vertex AI Matching Engine.
4. User queries are expanded using Gemini.
5. Expanded query embeddings are searched against the vector index.
6. Top matching documents are returned.

---

## Why Vertex AI Matching Engine?

Benefits:

- Managed vector database
- Scalable to billions of vectors
- Faster approximate nearest neighbor search
- Automatic infrastructure scaling
- Integrated Google Cloud security and IAM