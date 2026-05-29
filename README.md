# Regulatory Document Intelligence Assistant

A Retrieval-Augmented Generation (RAG) system that enables natural language querying
of Canadian banking regulatory documents with cited, auditable answers.

## Problem

Credit risk analysts and compliance teams spend significant time manually searching
regulatory documents (OSFI guidelines, IFRS standards) for specific requirements.
This system enables plain-English queries with responses grounded in source documents
and cited by page number — critical for audit trails in a regulated environment.

## Architecture

```
PDFs (OSFI E-23, OSFI B-20, IFRS 9)
        ↓
  Text Extraction (PyMuPDF)
        ↓
  Chunking (RecursiveCharacterTextSplitter — 800 tokens, 150 overlap)
        ↓
  Embeddings (OpenAI text-embedding-3-small)
        ↓
  Vector Store (ChromaDB, persisted locally)
        ↓
  Retriever (top-5 similarity search, deduplicated)
        ↓
  LLM (GPT-4o-mini, temperature=0)
        ↓
  Answer + Page-Level Source Citations
```

## Key Design Decisions

- **Chunk size 800 / overlap 150** — tuned for dense regulatory text to preserve
  context across section boundaries
- **temperature=0** — eliminates response variability for compliance use cases
  where consistency and accuracy are critical
- **Deduplication** — retrieved chunks are deduplicated by source and page before
  passing to LLM, reducing noise in context
- **Strict grounding prompt** — LLM is instructed to answer only from retrieved
  context, preventing hallucination in a compliance setting

## Evaluation Results

Evaluated on 15 hand-crafted question-answer pairs across all three documents:

| Metric | Score |
|--------|-------|
| Faithfulness | 0.93 |
| Answer Relevancy | 0.93 |

Faithfulness measures whether answers are grounded in retrieved context (hallucination
check). Relevancy measures whether answers directly address the question asked.

## Example Queries

**Q: What is a loan-to-value ratio under B-20?**
> The LTV ratio is an evaluation of the amount of collateral value that can be used
> to support a loan. It is highly correlated with credit risk. Residential mortgage
> loans with higher LTV ratios generally perform worse than those with lower LTV ratios.
> Source: Residential mortgage underwriting practices and procedures (page 12)

**Q: What does IFRS 9 require for 12-month expected credit losses?**
> If credit risk has not increased significantly since initial recognition, the loss
> allowance should be measured at an amount equal to 12-month expected credit losses.
> Source: ifrs-9-financial-instruments.pdf (page 25)

**Q: What does OSFI require for model validation under E-23?**
> OSFI requires independent validation of models at inception and on a regular basis,
> commensurate with the model risk rating.
> Source: Guideline E-23 Model Risk Management (page 5)

## Documents Indexed

| Document | Pages | Chunks |
|----------|-------|--------|
| OSFI E-23 — Model Risk Management (2027) | ~6 | ~28 |
| OSFI B-20 — Residential Mortgage Underwriting | ~30 | ~120 |
| IFRS 9 — Financial Instruments | ~185 | ~754 |
| **Total** | **221** | **902** |

## Stack

- **LangChain** — orchestration
- **ChromaDB** — vector store
- **OpenAI** — embeddings (text-embedding-3-small) and LLM (gpt-4o-mini)
- **PyMuPDF** — PDF text extraction
- **Python-dotenv** — environment management

## Setup

```bash
git clone https://github.com/David-Tam/RAG-Regulatory-Assistant.git
cd RAG-Regulatory-Assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Add your OpenAI API key to `.env`:

```
OPENAI_API_KEY=your_key_here
```

Run ingestion (first time only):

```bash
python src/ingest.py
```

Run the assistant:

```bash
python src/chain.py
```

Run evaluation:

```bash
python src/evaluate.py
```

## Future Improvements

- Swap ChromaDB for Azure AI Search for enterprise-scale deployment
- Add Azure Blob Storage for document management
- Expand document set to include OSFI B-10, Basel III, and IFRS 17
- Add MLflow experiment tracking for evaluation runs
