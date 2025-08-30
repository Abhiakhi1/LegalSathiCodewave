
import os
import pickle
import json
import numpy as np
import time
import requests
from src.search import retriever   # Your semantic retriever
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- Load the semantic graph ---
with open(r"C:\Users\Ketan\Desktop\legalsaathi\LegalSathi\graph builder\graph.pkl", "rb") as f:
    G = pickle.load(f)

with open(r"C:\Users\Ketan\Desktop\legalsaathi\LegalSathi\graph builder\keyword_relations.json") as f:
    relations = json.load(f)

# --- Gemini API Configuration ---
load_dotenv()
API_KEY = os.getenv("google_api_key")


class IntegratedRAGPipeline:
    """
    Enhanced RAG Pipeline integrating semantic retriever with graph expansion
    """

    def __init__(self):
        self.max_context_length = 3000  # Adjust based on model limits

    def retrieve_with_semantic_search(self, query: str, use_graph_expansion: bool = True):
        """
        MAIN INTEGRATION FUNCTION: Combines semantic retriever with graph expansion
        """
        print(f"\n--- Processing Query: '{query}' ---")

        # Step 1: Optional graph expansion 
        expanded_queries = [query]  # Start with original query

        if use_graph_expansion:
            print("Step 1: Expanding query with knowledge graph...")
            graph_keywords = self.expand_query_with_graph(query, G, top_k=2)
            expanded_queries.extend(graph_keywords[1:])  # Add expanded terms
            print(f"Expanded to: {expanded_queries}")

        # Step 2: Use semantic retriever for each expanded query
        print("Step 2: Retrieving documents with semantic search...")

        all_results = []
        seen_doc_ids = set()  # Prevent duplicates

        for expanded_query in expanded_queries:
            try:
                # CALL YOUR SEMANTIC RETRIEVER HERE
                search_results = retriever.search_top_categories(
                    query=expanded_query,
                    top_n_cats=3,      # Search top 3 categories
                    global_k=6         # Get up to 6 documents per query
                )

                # Extract unique documents
                for hit in search_results["hits"]:
                    doc_id = hit.get("id", "")
                    if doc_id not in seen_doc_ids:
                        seen_doc_ids.add(doc_id)
                        all_results.append({
                            "content": hit.get("doc", ""),
                            "similarity": hit.get("sim", 0.0),
                            "acts": hit.get("acts", ""),
                            "sections": hit.get("sections", ""),
                            "jurisdiction": hit.get("jurisdiction", ""),
                            "category": hit.get("category", ""),
                            "source_query": expanded_query,
                            "metadata": hit.get("metadata", {})
                        })

            except Exception as e:
                print(f"Error retrieving for query '{expanded_query}': {e}")
                continue

        # Step 3: Sort by similarity and limit results
        all_results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = all_results[:8]  # Keep top 8 most relevant

        print(f"Retrieved {len(top_results)} unique documents")

        return {
            "original_query": query,
            "expanded_queries": expanded_queries,
            "documents": top_results,
            "total_found": len(all_results)
        }

    def build_rag_prompt(self, query: str, retrieval_results: dict) -> str:
        """
        BUILD THE FINAL PROMPT: This is what gets sent to your LLM
        """
        documents = retrieval_results["documents"]

        if not documents:
            return f"Query: {query}\n\nNo relevant documents found in the legal database."

        # Build document context with legal metadata
        context_sections = []
        current_length = 0

        for i, doc in enumerate(documents):
            if not doc["content"].strip():
                continue

            # Format document with legal metadata
            doc_header = f"[DOCUMENT {i+1}]"

            # Add legal metadata if available
            metadata_parts = []
            if doc["acts"]:
                metadata_parts.append(f"Acts: {doc['acts']}")
            if doc["sections"]:
                metadata_parts.append(f"Sections: {doc['sections']}")
            if doc["jurisdiction"]:
                metadata_parts.append(f"Jurisdiction: {doc['jurisdiction']}")
            if doc["similarity"]:
                metadata_parts.append(f"Relevance: {doc['similarity']:.3f}")

            if metadata_parts:
                doc_header += " | " + " | ".join(metadata_parts)

            # Full document with header
            full_doc = f"{doc_header}\n{doc['content']}"

            # Check length limits
            doc_length = len(full_doc)
            if current_length + doc_length > self.max_context_length:
                # Truncate if needed
                remaining = self.max_context_length - current_length - 200
                if remaining > 300:  # Only include if meaningful space
                    truncated = full_doc[:remaining] + "\n[Document truncated...]"
                    context_sections.append(truncated)
                break

            context_sections.append(full_doc)
            current_length += doc_length

        # Build final RAG prompt
        context_text = "\n\n---\n\n".join(context_sections)

        # FINAL PROMPT STRUCTURE
        final_prompt = f"""You are a legal AI assistant. Use the provided legal documents to answer the user's query accurately. If the documents don't contain sufficient information, state that clearly.

            USER QUERY: {query}

            LEGAL CONTEXT FROM DATABASE:
            {context_text}

            INSTRUCTIONS:
            - Base your response primarily on the provided legal documents
            - Reference specific acts, sections, or jurisdictions when applicable  
            - If information is insufficient, say so explicitly
            - Provide clear, actionable legal guidance when possible

            RESPONSE:"""

        return final_prompt

    def expand_query_with_graph(self, query, graph, top_k=3):
        """
        Enhanced graph expansion (from your existing code)
        """
        query_lower = query.lower()
        expanded = [query_lower]

        # Try exact match first
        if query_lower in graph.nodes:
            neighbors = list(graph.successors(query_lower))
            expanded.extend(neighbors[:top_k])
        else:
            # Try partial matches in graph nodes
            matching_nodes = [node for node in graph.nodes if query_lower in node or node in query_lower]
            for node in matching_nodes[:top_k]:
                neighbors = list(graph.successors(node))
                expanded.extend(neighbors[:2])

        return list(set(expanded))[:top_k+1]  # Remove duplicates and limit

    def generate_answer(self, query: str, context_documents: list):
        """
        Generate answer using Gemini API (enhanced version of your existing function)
        """
        # Build the RAG prompt
        retrieval_results = {"documents": context_documents}
        rag_prompt = self.build_rag_prompt(query, retrieval_results)

        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-05-20:generateContent?key={API_KEY}"

        payload = {
            "contents": [{"parts": [{"text": rag_prompt}]}],
            "tools": [{"google_search": {}}],
            "systemInstruction": {
                "parts": [{"text": "You are an expert legal assistant. Provide accurate, well-sourced legal guidance based on the provided documents."}]
            }
        }

        # Exponential backoff (your existing logic)
        retries = 0
        max_retries = 5

        while retries < max_retries:
            try:
                response = requests.post(
                    api_url,
                    headers={'Content-Type': 'application/json'},
                    data=json.dumps(payload)
                )

                response.raise_for_status()
                result = response.json()

                candidate = result.get('candidates', [])[0]
                if candidate and candidate.get('content') and candidate['content'].get('parts'):
                    return candidate['content']['parts'][0]['text']
                else:
                    return "I could not generate a response based on the provided context."

            except requests.exceptions.RequestException as e:
                if hasattr(response, 'status_code') and response.status_code in [429, 500, 503]:
                    wait_time = (2 ** retries)
                    print(f"API call failed with status code {response.status_code}. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    retries += 1
                else:
                    print(f"An error occurred: {e}")
                    break

        return "Failed to get a response from the LLM after multiple retries."

    def process_query(self, user_query: str) -> dict:
        """
        MAIN PIPELINE FUNCTION: Complete end-to-end processing
        """
        print(f"\n{'='*60}")
        print(f"PROCESSING QUERY: {user_query}")
        print(f"{'='*60}")

        # Step 1: Retrieve with semantic search + graph expansion
        retrieval_results = self.retrieve_with_semantic_search(user_query)

        # Step 2: Generate final answer
        print("Step 3: Generating answer with LLM...")
        final_answer = self.generate_answer(user_query, retrieval_results["documents"])

        return {
            "query": user_query,
            "retrieval_results": retrieval_results,
            "final_answer": final_answer,
            "prompt_used": self.build_rag_prompt(user_query, retrieval_results)
        }

# USAGE EXAMPLE
def main():
    """
    Example usage of the integrated pipeline
    """
    pipeline = IntegratedRAGPipeline()

    # Test queries
    test_queries = [
        "Cruelty to animals generally (Section 11): This is the most critical provision of the Act. It defines by listing various acts that are considered offenses. These include:"
    ]

    for query in test_queries:
        result = pipeline.process_query(query)

        print(f"\nQUERY: {result['query']}")
        print(f"DOCUMENTS FOUND: {len(result['retrieval_results']['documents'])}")
        print(f"EXPANDED QUERIES: {result['retrieval_results']['expanded_queries']}")
        print(f"\nFINAL ANSWER:\n{result['final_answer']}")
        print(f"\n{'-'*50}")

if __name__ == "__main__":
    main()
