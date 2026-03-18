from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Sample data (better dataset)
documents = [
    "Rahul is a Java Backend Developer with Spring Boot and Microservices",
    "Sneha is a Frontend Developer skilled in React and JavaScript",
    "Arjun is a Data Analyst with Python, SQL, and Power BI",
    "Priya is a Full Stack Developer with Java, React, and MySQL"
]

# Convert documents to embeddings
doc_embeddings = model.encode(documents)

# Create FAISS index
dimension = len(doc_embeddings[0])
index = faiss.IndexFlatL2(dimension)
index.add(np.array(doc_embeddings))

# Continuous loop for multiple queries
while True:
    query = input("\nAsk something (type 'exit' to stop): ")

    if query.lower() == "exit":
        print("Goodbye!")
        break

    # Convert query to embedding
    query_embedding = model.encode([query])

    # Search top 2 results
    k = 2
    distances, indices = index.search(np.array(query_embedding), k)

    print("\nTop Matches:")

    for i in range(len(indices[0])):
        print(f"{i+1}. {documents[indices[0][i]]}")