# LegalSaathi: AI-Powered Legal Research Assistant

**LegalSaathi** is an advanced AI system designed to assist legal professionals by providing semantic document retrieval and intelligent legal analysis. It leverages state-of-the-art NLP techniques, knowledge graphs, and large language models to deliver accurate, relevant, and well-sourced legal information.

## Key Features
- **Semantic Search:** Uses vector databases and sentence transformers to find conceptually relevant legal documents.
- **Knowledge Graph Expansion:** Enhances queries with related legal concepts and relationships.
- **Legal Metadata Extraction:** Automatically captures acts, sections, and jurisdictions for precise context.
- **Robust RAG Pipeline:** Integrates with LLMs like Gemini, Ollama, or OpenAI for answer generation.
- **Novel Algorithms:** Implements novelty detection and auto category selection for efficient retrieval.

## Architecture Overview
- **Semantic Retrieval:** Implements a vector-based search using ChromaDB and sentence-transformers.
- **Legal Graph Expansion:** Utilizes a knowledge graph to broaden search queries.
- **Metadata Management:** Extracts and organizes legal metadata for accurate referencing.
- **LLM Integration:** Builds prompts dynamically and queries large language models for comprehensive answers.
- **Advanced Retrieval:** Employs relevance ranking, deduplication, and adaptive thresholds.

## Usage & Workflow
1. **Query Expansion:** Enhance user queries via a legal knowledge graph.
2. **Document Retrieval:** Perform semantic searches across legal document collections.
3. **Context Building:** Format retrieved documents with metadata for context.
4. **Answer Generation:** Call the chosen LLM to generate precise legal responses.

## Technology Stack
- Python
- LangChain, LangSmith
- ChromaDB, Sentence-Transformers
- Knowledge Graph (NetworkX)
- Large Language Models (Gemini, Ollama, GPT-4, etc.)

## Applications
- Legal research automation
- Document analysis and summarization
- Legal decision support systems
- Academic research projects

## Future Improvements
- Enhance knowledge graph connectivity
- Integrate more LLM providers dynamically
- Optimize retrieval speed and relevance
- Expand legal metadata and jurisdiction coverage

## License
This project is for educational and research use. For commercial use, ensure compliance with relevant legal and licensing terms.

---

*This project showcases the integration of advanced NLP, semantic search, and legal domain expertise, demonstrating scalable and intelligent legal research solutions.* 
