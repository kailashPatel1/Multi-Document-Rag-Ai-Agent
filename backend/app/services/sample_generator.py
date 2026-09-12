import os
import logging
import pandas as pd
from pathlib import Path
from ..config.settings import settings

logger = logging.getLogger(__name__)

def generate_pdf_document(filepath: str, title: str, pages_content: list):
    """Generates a genuine multi-page PDF document using ReportLab."""
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
    from reportlab.lib import colors

    doc = SimpleDocTemplate(filepath, pagesize=letter, leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54)
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=14
    )
    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#3b82f6'),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )
    bullet_style = ParagraphStyle(
        'DocBullet',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13,
        textColor=colors.HexColor('#1e293b'),
        leftIndent=15,
        spaceAfter=4
    )

    story = []

    for page_idx, page_data in enumerate(pages_content):
        if page_idx > 0:
            story.append(PageBreak())

        # Page Header
        story.append(Paragraph(f"{title} — Page {page_idx + 1}", title_style))
        story.append(Spacer(1, 10))

        for section in page_data.get("sections", []):
            if section.get("heading"):
                story.append(Paragraph(section["heading"], h2_style))
            if section.get("text"):
                story.append(Paragraph(section["text"], body_style))
            for item in section.get("bullets", []):
                story.append(Paragraph(f"• {item}", bullet_style))
            story.append(Spacer(1, 8))

    doc.build(story)
    logger.info(f"Generated PDF: {filepath}")

def generate_sample_documents():
    """Generates the 5 demo files in data/sample_documents."""
    out_dir = settings.SAMPLE_DOCS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resume.pdf
    resume_path = out_dir / "Resume.pdf"
    if not resume_path.exists():
        generate_pdf_document(
            str(resume_path),
            "Alex Rivera — Senior AI Engineer Resume",
            [
                {
                    "sections": [
                        {
                            "heading": "Candidate Summary",
                            "text": "Senior Artificial Intelligence & Machine Learning Engineer with 6+ years of production experience architecting Retrieval-Augmented Generation (RAG) systems, multi-agent frameworks (LangGraph), vector databases (ChromaDB, Pinecone), and scalable LLM inference pipelines."
                        },
                        {
                            "heading": "Core Technical Skills",
                            "bullets": [
                                "Languages: Python (Advanced), SQL, TypeScript, Bash",
                                "AI/GenAI: LangChain, LangGraph, ChromaDB, Sentence-Transformers, HuggingFace, LlamaIndex, OpenAI, Anthropic, Groq, Gemini",
                                "Machine Learning: PyTorch, Scikit-learn, HuggingFace Transformers, LoRA, QLoRA, Fine-Tuning",
                                "Backend & Cloud: FastAPI, Docker, PostgreSQL, Redis, AWS (SageMaker, S3, ECS), Git CI/CD",
                                "RAG Capabilities: Hybrid Retrieval (Vector + BM25), Reciprocal Rank Fusion, Chunking Optimization, Context Reranking"
                            ]
                        },
                        {
                            "heading": "Professional Experience",
                            "bullets": [
                                "Staff AI Engineer at Cortex Labs (2022 - Present): Designed multi-document RAG assistant serving 200k+ daily queries with sub-250ms latency. Implemented LangGraph multi-agent workflows for autonomous data analytics.",
                                "AI Research Engineer at DeepScale AI (2019 - 2022): Fine-tuned open-source LLMs using LoRA/QLoRA on specialized enterprise corpora. Developed hybrid BM25 + dense vector search pipeline."
                            ]
                        }
                    ]
                }
            ]
        )

    # 2. Job_Description.pdf
    jd_path = out_dir / "Job_Description.pdf"
    if not jd_path.exists():
        generate_pdf_document(
            str(jd_path),
            "Apex AI Labs — Senior GenAI & Knowledge Systems Engineer",
            [
                {
                    "sections": [
                        {
                            "heading": "Position Overview",
                            "text": "Apex AI Labs is seeking a Staff / Senior GenAI Engineer to lead the architecture and deployment of enterprise-grade Multi-Document Knowledge Agents, Hybrid RAG pipelines, and agentic workflows."
                        },
                        {
                            "heading": "Mandatory Qualifications & Requirements",
                            "bullets": [
                                "5+ years software engineering with deep focus on Python and PyTorch.",
                                "Hands-on experience implementing LangGraph, LangChain, and autonomous Tool-Calling agent state machines.",
                                "Extensive production experience with Vector Databases (ChromaDB, Qdrant, Milvus) and Hybrid Search (BM25 + Semantic Vector).",
                                "Proven expertise in RAG Evaluation frameworks (measuring retrieval precision, recall, context relevance, faithfulness).",
                                "Knowledge of production FastAPI backend design, secure authentication, and real-time streaming."
                            ]
                        },
                        {
                            "heading": "Nice-to-Have / Missing Skills to Watch For",
                            "bullets": [
                                "Knowledge Graphs (Neo4j) integrated with Vector RAG.",
                                "Distributed Kubernetes cluster orchestration and GPU kernel optimization (Triton / TensorRT-LLM).",
                                "Advanced GraphRAG multi-hop relational reasoning."
                            ]
                        }
                    ]
                }
            ]
        )

    # 3. RAG_Notes.pdf
    rag_path = out_dir / "RAG_Notes.pdf"
    if not rag_path.exists():
        generate_pdf_document(
            str(rag_path),
            "Comprehensive Guide to Retrieval-Augmented Generation (RAG)",
            [
                # Page 1
                {
                    "sections": [
                        {
                            "heading": "1. What is RAG and Core Architecture",
                            "text": "Retrieval-Augmented Generation (RAG) is an architectural pattern that enhances Large Language Models by dynamically retrieving relevant facts from private, external knowledge repositories before generating an answer. This prevents hallucinations, grounds answers in verified facts, and allows real-time private document queries without costly retraining."
                        },
                        {
                            "heading": "2. Document Ingestion & Chunking Strategies",
                            "bullets": [
                                "Recursive Character Splitting: Breaks texts based on natural semantic boundaries (\n\n, \n, sentence terminators) with configurable overlap.",
                                "Chunk Size: Typically 400 - 800 tokens. Smaller chunks preserve precision; larger chunks preserve context.",
                                "Chunk Overlap: 10% - 20% overlap ensures semantic continuity across chunk boundaries.",
                                "Metadata Enrichment: Every chunk must record document_name, document_id, page_number, chunk_id, and section headers."
                            ]
                        }
                    ]
                },
                # Page 2
                {
                    "sections": [
                        {
                            "heading": "3. Text Embeddings and Vector Databases",
                            "text": "Embeddings are dense numerical vectors representing the semantic meaning of text in a high-dimensional space. Distance metrics such as Cosine Similarity measure semantic proximity."
                        },
                        {
                            "heading": "4. Vector Database Architecture",
                            "bullets": [
                                "ChromaDB: Lightweight, open-source embedded vector database with persistent local indexing.",
                                "Indexing Algorithms: Hierarchical Navigable Small World (HNSW) and Inverted File (IVF) index graphs.",
                                "Metadata Filtering: Query filtering allows narrowing vector search to specific document subsets or timestamps."
                            ]
                        }
                    ]
                },
                # Page 3
                {
                    "sections": [
                        {
                            "heading": "5. Hybrid Search & Reciprocal Rank Fusion",
                            "text": "Hybrid Search fuses dense vector semantic similarity with sparse lexical keyword matching (BM25). This solves semantic blindness to exact product codes, acronyms, and rare entity names."
                        },
                        {
                            "heading": "6. Reciprocal Rank Fusion (RRF)",
                            "bullets": [
                                "RRF Formula: RRF_Score(d) = alpha * (1 / (60 + rank_vector)) + (1 - alpha) * (1 / (60 + rank_bm25)).",
                                "Alpha Weighting: Alpha=0.5 provides balanced semantic and keyword precision.",
                                "Reranking: Cross-encoders rescore top candidate passages for maximum relevance."
                            ]
                        }
                    ]
                }
            ]
        )

    # 4. ML_Notes.pdf
    ml_path = out_dir / "ML_Notes.pdf"
    if not ml_path.exists():
        generate_pdf_document(
            str(ml_path),
            "Machine Learning & Deep Neural Architecture Notes",
            [
                # Page 1
                {
                    "sections": [
                        {
                            "heading": "1. Transformer Architecture & Self-Attention",
                            "text": "Transformers rely on Multi-Head Self-Attention mechanisms to process input sequences in parallel, computing dynamic attention weights between all token pairs using Query (Q), Key (K), and Value (V) projections: Attention(Q,K,V) = softmax(QK^T / sqrt(d_k)) * V."
                        },
                        {
                            "heading": "2. Positional Encodings & Feed-Forward Networks",
                            "bullets": [
                                "Rotary Position Embeddings (RoPE): Encodes relative token positions by rotating query/key vectors in complex space.",
                                "Gated Linear Units (SwiGLU): Improves non-linear feature representation in transformer feedforward layers."
                            ]
                        }
                    ]
                },
                # Page 2
                {
                    "sections": [
                        {
                            "heading": "3. Parameter-Efficient Fine-Tuning (PEFT)",
                            "text": "LoRA (Low-Rank Adaptation) freezes pre-trained model weights and injects trainable rank-decomposition matrices into transformer layers, drastically reducing trainable parameters."
                        },
                        {
                            "heading": "4. Quantization Techniques",
                            "bullets": [
                                "4-bit NormalFloat (NF4): Optimized 4-bit data type for normally distributed neural weights.",
                                "Double Quantization: Quantizes quantization constants to save additional memory."
                            ]
                        }
                    ]
                }
            ]
        )

    # 5. Sample_Sales.csv
    csv_path = out_dir / "Sample_Sales.csv"
    if not csv_path.exists():
        sales_data = {
            "Transaction_ID": [f"TXN-{1000 + i}" for i in range(12)],
            "Date": ["2026-01-15", "2026-01-22", "2026-02-05", "2026-02-18", "2026-03-02", "2026-03-20", "2026-04-10", "2026-04-25", "2026-05-12", "2026-05-28", "2026-06-15", "2026-06-30"],
            "Product": ["Enterprise AI Platform", "Cloud Vector Search", "Enterprise AI Platform", "Agent Orchestrator", "Cloud Vector Search", "Enterprise AI Platform", "Data Pipeline Pro", "Agent Orchestrator", "Enterprise AI Platform", "Cloud Vector Search", "Agent Orchestrator", "Data Pipeline Pro"],
            "Region": ["North America", "Europe", "Asia-Pacific", "North America", "Europe", "North America", "Asia-Pacific", "Europe", "North America", "Asia-Pacific", "North America", "Europe"],
            "Units_Sold": [12, 25, 8, 15, 30, 18, 40, 20, 22, 35, 28, 45],
            "Unit_Price": [5000, 1200, 5000, 3500, 1200, 5000, 800, 3500, 5000, 1200, 3500, 800],
            "Revenue": [60000, 30000, 40000, 52500, 36000, 90000, 32000, 70000, 110000, 42000, 98000, 36000],
            "Customer_Segment": ["Fortune 500", "Mid-Market", "Enterprise", "Startup", "Enterprise", "Fortune 500", "Mid-Market", "Enterprise", "Fortune 500", "Enterprise", "Fortune 500", "Mid-Market"]
        }
        df = pd.DataFrame(sales_data)
        df.to_csv(str(csv_path), index=False)
        logger.info(f"Generated CSV: {csv_path}")

    return {
        "resume": str(resume_path),
        "job_description": str(jd_path),
        "rag_notes": str(rag_path),
        "ml_notes": str(ml_path),
        "sample_sales": str(csv_path)
    }
