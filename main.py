import numpy as np
import faiss

from sentence_transformers import SentenceTransformer


# LOAD DOCUMENTS
def load_documents(file_path):

    with open(file_path, "r", encoding="utf-8") as file:

        documents = [
            line.strip()
            for line in file.readlines()
            if line.strip()
        ]

    return documents


# QUERY EXPANSION
def expand_query(query):

    query_expansions = {
        "peak load":
            "traffic spikes autoscaling load balancing high concurrency scalability",
        "failures":
            "fault tolerance retry mechanisms circuit breakers resilience service isolation",
        "database latency":
            "redis caching query optimization response time read performance"
    }

    expanded_query = query.lower()

    for key, value in query_expansions.items():
        if key in query.lower():
            expanded_query += " " + value
    return expanded_query


# LOAD EMBEDDING MODEL
print("Loading embedding model...")

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

print("Embedding model loaded successfully.")


# GENERATE EMBEDDINGS
def generate_embeddings(texts):

    embeddings = model.encode(
        texts,
        normalize_embeddings=True
    )

    return embeddings


# CREATE VECTOR STORE
def create_vector_store(embeddings):

    dimension = len(embeddings[0])

    index = faiss.IndexFlatIP(dimension)

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    index.add(embeddings)

    return index


# =========================================================
# SEARCH DOCUMENTS
# =========================================================

def search_documents(
    query,
    index,
    documents,
    k=3
):

    query_embedding = generate_embeddings(
        [query]
    )

    query_embedding = np.array(
        query_embedding,
        dtype=np.float32
    )

    scores, indices = index.search(
        query_embedding,
        k
    )

    results = []

    for i, idx in enumerate(indices[0]):
        results.append({
            "score": float(scores[0][i]),
            "document": documents[idx]
        })

    return results


def run_retrieval(query, index, documents):

    # =====================================================
    # STRATEGY A
    # =====================================================

    raw_results = search_documents(
        query,
        index,
        documents
    )

    # =====================================================
    # STRATEGY B
    # =====================================================

    expanded_query = expand_query(query)

    expanded_results = search_documents(
        expanded_query,
        index,
        documents
    )

    final_result = {
        "query": query,
        "strategy_a_raw_search": raw_results,
        "expanded_query": expanded_query,
        "strategy_b_expanded_search": expanded_results
    }
    return final_result


# =========================================================
# PRINT RESULTS
# =========================================================

def print_results(results):

    print("\n================================================")
    print(f"QUERY: {results['query']}")
    print("================================================")

    # STRATEGY A
    print("\nSTRATEGY A : RAW VECTOR SEARCH")

    for i, result in enumerate(results["strategy_a_raw_search"],start=1):
        print(f"\nTop Result {i}")
        print(f"Similarity Score: {result['score']:.4f}")
        print(result["document"])

    # STRATEGY B ----- 
    print("\nSTRATEGY B : AI-ENHANCED RETRIEVAL")
    print("\nExpanded Query:")
    print(results["expanded_query"])

    for i, result in enumerate(results["strategy_b_expanded_search"],start=1):
        print(f"\nTop Result {i}")
        print(f"Similarity Score: {result['score']:.4f}")
        print(result["document"])



def main():

    print("\n================================================")
    print("CONTEXT-AWARE RETRIEVAL ENGINE")
    print("================================================")

    # LOAD DOCUMENTS
    documents = load_documents("data.txt")

    print(f"\nLoaded {len(documents)} documents.")

    # GENERATE DOCUMENT EMBEDDINGS
    print("\nGenerating embeddings...")

    document_embeddings = generate_embeddings(
        documents
    )

    print("Embeddings generated successfully.")

    # CREATE VECTOR DATABASE
    index = create_vector_store(
        document_embeddings
    )

    print("FAISS vector database created.")


    while True:

        print("\n")

        user_query = input(
            "Enter your query (type 'exit' to quit): "
        )

        if user_query.lower() == "exit":

            print("\nExiting retrieval engine.")

            break

        results = run_retrieval(
            user_query,
            index,
            documents
        )

        print_results(results)



# --- entry point -- 
if __name__ == "__main__":
    main()