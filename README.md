# FrameRecall - QR Code Video-Based AI Memory

[![PyPI version](https://badge.fury.io/py/framerecall.svg)](https://pypi.org/project/framerecall/)
[![Downloads](https://pepy.tech/badge/framerecall)](https://pepy.tech/project/framerecall)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Ultra-fast toolkit crafted within a leading scripting language, capable of generating, archiving, then recalling artificial intelligence recollections through two-dimensional matrix clip sequences. The platform supplies meaning-based lookup spanning countless document fragments, answering in under one second.

## 🚀 Why FrameRecall?

### Transformative Approach
- **Clips as Storage**: Archive vast amounts of textual information in a compact .mp4
- **Blazing Access**: Retrieve relevant insights within milliseconds using meaning-based queries
- **Superior Compression**: Frame encoding significantly lowers data requirements
- **Serverless Design**: Operates entirely via standalone files – no backend needed
- **Fully Local**: Entire system runs independently once memory footage is created

### Streamlined System
- **Tiny Footprint**: Core logic spans fewer than 1,000 lines of code
- **Resource-Conscious**: Optimised to perform well on standard processors
- **Self-Contained**: Entire intelligence archive stored in one clip
- **Remote-Friendly**: Media can be delivered directly from online storage

## 📦 Installation

### Quick Install
```bash
pip install framerecall
```

### For PDF Support
```bash
pip install framerecall PyPDF2
```

### Recommended Setup (Virtual Environment)
```bash
# Create a new project directory
mkdir my-framerecall-project
cd my-framerecall-project

# Create virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install framerecall
pip install framerecall

# For PDF support:
pip install PyPDF2
```

## 🎯 Getting Started Instantly

```python
from framerecall import FrameRecallEncoder, FrameRecallChat

# Construct memory sequence using textual inputs
segments = ["Crucial insight 1", "Crucial insight 2", "Contextual knowledge snippet", ...]
builder = FrameRecallEncoder()
builder.add_chunks(segments)
builder.build_video("archive.mp4", "archive_index.json")

# Interact with stored intelligence
assistant = FrameRecallChat("archive.mp4", "archive_index.json")
assistant.start_session()
output = assistant.chat("Tell me what’s known about past happenings?")
print(output)
```

### Constructing Memory from Files
```python
from framerecall import FrameRecallEncoder
import os

# Prepare input texts
assembler = FrameRecallEncoder(chunk_size=512, overlap=50)

# Inject content from directory
for filename in os.listdir("documents"):
    with open(f"documents/{filename}", "r") as document:
        assembler.add_text(document.read(), metadata={"source": filename})

# Generate compressed video sequence
assembler.build_video(
    "knowledge_base.mp4",
    "knowledge_index.json",
    fps=30,        # More chunks processed per second
    frame_size=512 # Expanded resolution accommodates extra information
)
```

### Intelligent Lookup & Extraction
```python
from framerecall import FrameRecallRetriever

# Set up fetcher
fetcher = FrameRecallRetriever("knowledge_base.mp4", "knowledge_index.json")

# Contextual discovery
matches = fetcher.search("machine learning algorithms", top_k=5)
for fragment, relevance in matches:
    print(f"Score: {relevance:.3f} | {fragment[:100]}...")

# Retrieve neighbouring fragments
window = fetcher.get_context("explain neural networks", max_tokens=2000)
print(window)
```

### Conversational Interface
```python
from framerecall import FrameRecallInteractive

# Open real-time discussion UI
interactive = FrameRecallInteractive("knowledge_base.mp4", "knowledge_index.json")
interactive.run()  # Web panel opens at http://localhost:7860
```

### Full Demo: Converse with a PDF Book
```bash
# 1. Prepare project directory and virtual environment
mkdir book-chat-demo
cd book-chat-demo
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install necessary packages
pip install framerecall PyPDF2

# 3. Build book_chat.py
cat > book_chat.py << 'EOF'
from framerecall import FrameRecallEncoder, chat_with_memory
import os

# Path to your document
book_pdf = "book.pdf"  # Replace with your PDF filename

# Encode video from book
encoder = FrameRecallEncoder()
encoder.add_pdf(book_pdf)
encoder.build_video("book_memory.mp4", "book_index.json")

# Initiate interactive Q&A
api_key = os.getenv("OPENAI_API_KEY")  # Optional for model output
chat_with_memory("book_memory.mp4", "book_index.json", api_key=api_key)
EOF

# 4. Launch the assistant
export OPENAI_API_KEY="your-api-key"  # Optional
python book_chat.py
```

## 🔧 API Summary

### FrameRecallEncoder
```python
encoder = FrameRecallEncoder(
    chunk_size=512,       # Letters per segment
    overlap=50,           # Shared characters between segments
    model_name='all-MiniLM-L6-v2'  # Embedding architecture
)

# Available Functions
encoder.add_chunks(chunks: List[str], metadata: List[dict] = None)
encoder.add_text(text: str, metadata: dict = None)
encoder.build_video(video_path: str, index_path: str, fps: int = 30, qr_size: int = 512)
```

### FrameRecallRetriever
```python
retriever = FrameRecallRetriever(
    video_path: str,
    index_path: str,
    cache_size: int = 100  # Frames retained in memory
)

# Search Features
results = retriever.search(query: str, top_k: int = 5)
context = retriever.get_context(query: str, max_tokens: int = 2000)
chunks = retriever.get_chunks_by_ids(chunk_ids: List[int])
```

### FrameRecallChat
```python
chat = FrameRecallChat(
    video_path: str,
    index_path: str,
    llm_backend: str = 'openai',  # Options: 'openai', 'anthropic', 'local'
    model: str = 'gpt-4'
)

# Dialogue Capabilities
chat.start_session(system_prompt: str = None)
response = chat.chat(message: str, stream: bool = False)
chat.clear_history()
chat.export_conversation(path: str)
```

## 🛠️ Extended Setup

### Tailored Embeddings
```python
from sentence_transformers import SentenceTransformer

# Load alternative semantic model
custom_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
encoder = FrameRecallEncoder(embedding_model=custom_model)
```

### Parallelized Workloads
```python
# Accelerate processing with concurrency
encoder = FrameRecallEncoder(n_workers=8)
encoder.add_chunks_parallel(massive_chunk_list)
```

## 🐛 Debugging Guide

### Frequent Pitfalls

**ModuleNotFoundError: No module named 'framerecall'**
```bash
# Confirm the correct Python interpreter is being used
which python  # Expected to point to your environment
# If incorrect, reactivate the virtual setup:
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**ImportError: PyPDF2 missing for document parsing**
```bash
pip install PyPDF2
```

**Missing or Invalid OpenAI Token**
```bash
# Provide your OpenAI credentials (register at https://platform.openai.com)
export OPENAI_API_KEY="sk-..."  # macOS/Linux
# On Windows:
set OPENAI_API_KEY=sk-...
```

**Handling Extensive PDFs**
```python
# Reduce segment length for better handling
encoder = FrameRecallEncoder()
encoder.add_pdf("large_book.pdf", chunk_size=400, overlap=50)
```

## 🤝 Get Involved

We’re excited to collaborate! Refer to our [Contribution Manual](CONTRIBUTING.md) for full instructions.

```bash
# Execute test suite
pytest tests/

# Execute with coverage reporting
pytest --cov=framerecall tests/

# Apply code styling
black framerecall/
```

## 🆚 How FrameRecall Compares to Other Technologies

| Capability          | FrameRecall | Embedding Stores | Relational Systems |
|---------------------|-------------|------------------|--------------------|
| Data Compression    | 🌟🌟🌟🌟🌟      | 🌟🌟              | 🌟🌟🌟              |
| Configuration Time  | Minimal     | Advanced         | Moderate           |
| Conceptual Matching | ✔️          | ✔️               | ✖️                 |
| Disconnected Access | ✔️          | ✖️               | ✔️                 |
| Mobility            | Standalone File | Hosted        | Hosted             |
| Throughput Limits   | Multi-million | Multi-million   | Multi-billion      |
| Financial Impact    | No Charge   | High Fees        | Moderate Expense   |

## 🗺️ What’s Coming Next

- [ ] **v0.2.0** – International text handling
- [ ] **v0.3.0** – On-the-fly memory insertion
- [ ] **v0.4.0** – Parallel video segmentation
- [ ] **v0.5.0** – Visual and auditory embedding
- [ ] **v1.0.0** – Enterprise-grade, stable release

## 📚 Illustrative Use Cases

Explore the [examples/](examples/) folder to discover:
- Transforming Wikipedia datasets into searchable memories
- Developing custom insight archives
- Multilingual capabilities
- Live content updates
- Linking with top-tier LLM platforms

## 🔗 Resources

- [Package on PyPI](https://pypi.org/project/framerecall)
- [Codebase](https://github.com/sabih-urrehman/framerecall)
- [Community Forum](https://x.com/framerecall)
- [Changelog](https://github.com/sabih-urrehman/framerecall/releases)

## 📄 Usage Rights

Licensed under the MIT agreement — refer to the [LICENSE](LICENSE) document for specifics.

**Time to redefine how your LLMs recall information — deploy FrameRecall and ignite knowledge!** 🚀