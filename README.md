# 🚀 FrameRecall

[![Test Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen)](https://github.com/sabih-urrehman/framerecall)
[![PyPI Version](https://img.shields.io/pypi/v/framerecall.svg)](https://pypi.org/project/framerecall)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## 🌟 Description

FrameRecall is your ultimate solution for **searchable video memory**, seamlessly integrating with Language Models (LLMs). 📹💡 Quickly slice, embed, and search millions of video frames in mere seconds, transforming vast visual content into powerful insights.

Ideal for AI researchers, video analysts, developers, and businesses seeking ultra-fast visual data queries for applications such as security, surveillance, behavioral analytics, and intelligent marketing. 🌐

---

## 📚 Table of Contents

* [✨ Key Features](#-key-features)
* [🛠 Installation](#-installation)
* [🚀 Usage](#-usage)
* [⚙️ Configuration](#️-configuration)
* [🔧 Architecture](#-architecture)
* [📖 API / CLI Reference](#-api--cli-reference)
* [🧪 Testing](#-testing)
* [📄 License](#-license)

---

## ✨ Key Features

* ⚡ **Instant searches** across millions of video frames.
* 🤖 Seamless **OpenAI and other LLM integrations**.
* 🎨 User-friendly **browser-based UI** powered by Streamlit.
* 🐳 **Dockerized deployment** for effortless setup.
* ☁️ Designed specifically for **AWS and other cloud environments**.
* 🔍 Comprehensive **video indexing and embedding system**.
* 🚀 High-performance, scalable infrastructure.

---

## 🛠 Installation

**Prerequisites:**

* Python 3.9 or higher
* Docker (recommended)

### Step-by-step guide:

```bash
# Clone the repository
git clone https://github.com/sabih-urrehman/framerecall.git
cd framerecall

# Install dependencies
pip install -r requirements.txt

# Run with Docker (recommended)
docker build -t framerecall .
docker run -p 8501:8501 framerecall
```

---

## 🚀 Usage

### Launching the application

```bash
# Start local server
python app.py
```

### CLI Examples

```bash
# Query using CLI
python query.py --video path/to/video.mp4 --query "person running"
```

**CLI Options:**

* `--video`: Path to your video file
* `--query`: Descriptive text for frame search

**Note:** Explore detailed use-cases and examples in the `examples/` directory provided in the repository.

---

## ⚙️ Configuration

Customize your setup via the `.env` file:

```bash
OPENAI_API_KEY=your-openai-api-key
AWS_ACCESS_KEY=your-aws-access-key
AWS_SECRET_KEY=your-aws-secret-key
```

---

## 🔧 Architecture

![Architecture Diagram](https://i.imgur.com/Qsi4QV7.png "FrameRecall Architecture Overview")

### Core Components:

* 🌐 **Web Interface:** Built with Streamlit for easy interaction.
* 📈 **Indexing & Embedding Service:** Efficiently manages and retrieves visual embeddings.
* 🤖 **OpenAI API Integration:** Powers semantic video frame queries.
* ☁️ **AWS Cloud Storage:** Reliable and scalable video storage.

---

## 📖 API / CLI Reference

### Command-Line Interface (CLI)

```bash
python query.py [OPTIONS]
```

| Option    | Description                           |
| --------- | ------------------------------------- |
| `--video` | Path to the input video file          |
| `--query` | Text description for searching frames |

---

## 🧪 Testing

Tests ensure reliability and functionality across the application.

### Running Tests:

```bash
# Execute all tests
pytest tests/
```

**Test Details:**

* Includes unit tests, integration tests, and benchmarks.
* Test coverage maintained consistently above 90%.
* Detailed test scenarios available in the `tests/` directory.
* Tests utilize fixtures and mocks for comprehensive coverage.

---

## 📄 License

MIT License © 2025 Sabih Ur Rehman
