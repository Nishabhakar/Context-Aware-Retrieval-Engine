import json
from datetime import datetime

from main import (
    load_documents,
    generate_embeddings,
    create_vector_store,
    run_retrieval
)


# =========================================================
# TEST QUERIES
# =========================================================

queries = [
    "How does the system handle peak load?",
    "How are failures isolated in distributed systems?",
    "How is database latency reduced?"
]


# LOAD DATA

documents = load_documents("data.txt")
document_embeddings = generate_embeddings(documents)

index = create_vector_store(document_embeddings)


# RUN TESTS

all_results = []

for query in queries:

    result = run_retrieval(
        query,
        index,
        documents
    )

    all_results.append(result)


# SAVE RESULTS WITH TIMESTAMP
timestamp = datetime.now().strftime(
    "%Y%m%d_%H%M%S"
)

output_file = f"retrieval_results_{timestamp}.json"

with open(
    output_file,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        all_results,
        file,
        indent=4
    )

print(f"\nResults stored in: {output_file}")