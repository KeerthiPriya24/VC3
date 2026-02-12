# 🗂️ SEFS — Semantic Entropy File System

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

> **Self-Organizing File Management Powered by AI Semantic Understanding**

---

## 📋 Table of Contents

1. [Project Title](#-sefs--semantic-entropy-file-system)
2. [Description](#-description)
3. [Tech Stack Used](#-tech-stack-used)
4. [How to Run the Project](#-how-to-run-the-project)
5. [Dependencies](#-dependencies)
6. [Important Instructions](#-important-instructions)
7. [Demo Videos of MVP](#-demo-videos-of-mvp)
8. [Demo Images of MVP](#-demo-images-of-mvp)

---

## 📖 Description

**SEFS (Semantic Entropy File System)** is an intelligent, AI-powered file management system that automatically organizes your files based on their **semantic content** rather than traditional folder hierarchies.

### 🎯 Problem Statement
Traditional file systems require manual organization, leading to:
- Scattered documents across multiple folders
- Difficulty finding related files
- Time wasted on manual categorization
- Inconsistent folder naming conventions

### 💡 Solution
SEFS uses **Natural Language Processing (NLP)** and **Machine Learning** to:

1. **Analyze File Content**: Extracts text from documents (TXT, MD, RST, LOG, PDF)
2. **Generate Semantic Embeddings**: Creates vector representations of file content using transformer models
3. **Cluster Similar Files**: Groups files with related content using DBSCAN clustering
4. **Auto-Generate Folder Names**: Creates meaningful, context-aware folder names based on keywords
5. **Real-Time Monitoring**: Watches for file changes and automatically reorganizes
6. **Visual Dashboard**: Interactive 2D visualization of file clusters

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **Semantic Analysis** | Uses sentence-transformers to understand file content meaning |
| 📊 **Recursive Clustering** | Hierarchical grouping with configurable depth (up to 3 levels) |
| 🔄 **Real-Time Sync** | Watchdog-based file monitoring with automatic reorganization |
| 🎨 **Visual Interface** | Streamlit dashboard with t-SNE 2D projections |
| 📁 **Smart Naming** | TF-IDF based keyword extraction for folder names |
| 💾 **SQLite Storage** | Persistent metadata and embedding cache |
| ⚡ **Incremental Updates** | Only re-analyzes changed files (checksum-based) |

---

## 🛠️ Tech Stack Used

### Core Technologies

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.10+ | Primary programming language |
| **ML/NLP** | sentence-transformers | Semantic embedding generation |
| **Clustering** | scikit-learn (DBSCAN) | File grouping algorithm |
| **File Monitoring** | watchdog | Real-time filesystem events |
| **Database** | SQLite | Metadata and embedding storage |
| **Visualization** | Streamlit + Plotly | Interactive web dashboard |
| **Dimensionality Reduction** | t-SNE | 2D visualization of embeddings |

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SEFS Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐ │
│   │ File Monitor │───▶│ Content Analyzer │───▶│   Database   │ │
│   │  (watchdog)  │    │  (transformers)  │    │   (SQLite)   │ │
│   └──────────────┘    └──────────────────┘    └──────────────┘ │
│          │                     │                      │        │
│          ▼                     ▼                      ▼        │
│   ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐ │
│   │     Main     │◀──▶│ Semantic Engine  │◀──▶│OS Synchronizer│ │
│   │ Orchestrator │    │    (DBSCAN)      │    │ (File Moves) │ │
│   └──────────────┘    └──────────────────┘    └──────────────┘ │
│                               │                                 │
│                               ▼                                 │
│                      ┌──────────────────┐                       │
│                      │ Visual Interface │                       │
│                      │   (Streamlit)    │                       │
│                      └──────────────────┘                       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Libraries & Frameworks

- **sentence-transformers** (`all-mpnet-base-v2` model) - State-of-the-art embeddings
- **scikit-learn** - DBSCAN clustering, TF-IDF vectorization, cosine similarity
- **PyPDF2** - PDF text extraction
- **chardet** - Character encoding detection
- **numpy/pandas** - Numerical operations and data handling
- **plotly** - Interactive visualizations
- **streamlit-autorefresh** - Real-time dashboard updates

---

## 🚀 How to Run the Project

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- ~2GB disk space (for ML models)

### Installation

#### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd VC3
```

#### Step 2: Create Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# OR
.venv\Scripts\activate     # On Windows
```

#### Step 3: Install Dependencies

```bash
pip install -e .
# OR for development
pip install -e ".[dev]"
```

### Running SEFS

#### Option 1: Command Line (Backend Only)

```bash
# Basic usage - organize files in a directory
python -m sefs.main ~/your_folder_to_organize

# With custom similarity threshold (0.0 - 1.0)
python -m sefs.main ~/your_folder --threshold 0.55

# Run in background
python -m sefs.main ~/your_folder &
```

#### Option 2: Visual Dashboard (Streamlit UI)

```bash
# Start the visual interface
streamlit run sefs/visual_interface.py
```

Then open your browser to: **http://localhost:8501**

#### Option 3: Full System (Recommended)

```bash
# Terminal 1: Start the backend
python -m sefs.main ~/sefs_root --threshold 0.55

# Terminal 2: Start the dashboard
streamlit run sefs/visual_interface.py
```

### Command Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `root_directory` | Path to monitor and organize | Required |
| `--threshold` | Similarity threshold (0.0-1.0) | 0.55 |
| `--model` | Embedding model name | `all-mpnet-base-v2` |
| `--max-depth` | Maximum folder nesting depth | 3 |

---

## 📦 Dependencies

### Required Dependencies

```toml
[project]
dependencies = [
    "watchdog>=3.0",              # File system monitoring
    "sentence-transformers>=2.2",  # Semantic embeddings (ML)
    "PyPDF2>=3.0",                # PDF text extraction
    "chardet>=5.0",               # Encoding detection
    "scikit-learn>=1.3",          # Clustering & ML utilities
    "numpy>=1.24",                # Numerical operations
    "streamlit>=1.28",            # Web dashboard
    "plotly>=5.18",               # Interactive charts
    "streamlit-autorefresh>=1.0", # Auto-refresh dashboard
    "pandas>=2.0",                # Data manipulation
]
```

### Development Dependencies

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4",        # Testing framework
    "hypothesis>=6.90",   # Property-based testing
    "pytest-cov>=4.1",    # Coverage reporting
]
```

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Python** | 3.10 | 3.11+ |
| **RAM** | 4GB | 8GB+ |
| **Disk** | 2GB | 5GB+ |
| **OS** | macOS, Linux, Windows | Any |

---

## ⚠️ Important Instructions

### Before Running

1. **First-Time Setup**: The first run downloads the `all-mpnet-base-v2` model (~400MB). Ensure stable internet connection.

2. **Backup Your Files**: While SEFS is designed to be safe, always backup important files before auto-organization.

3. **Supported File Types**:
   - Text files: `.txt`, `.md`, `.rst`, `.log`
   - Documents: `.pdf`

### Configuration

SEFS stores configuration in `.sefs/config.json` within your monitored folder:

```json
{
  "root_directory": "/path/to/your/folder",
  "similarity_threshold": 0.55,
  "clustering_algorithm": "dbscan",
  "embedding_model": "all-mpnet-base-v2",
  "max_depth": 3,
  "enable_preview_mode": false
}
```

### Key Parameters

| Parameter | Description | Recommended |
|-----------|-------------|-------------|
| `similarity_threshold` | Higher = stricter grouping | 0.50-0.65 |
| `max_depth` | Folder nesting levels | 2-3 |
| `min_cluster_size_for_split` | Min files before sub-clustering | 3 |

### Tips for Best Results

1. **File Content Matters**: Files need substantial text content for accurate clustering
2. **Give It Time**: Initial analysis can take a few minutes for large folders
3. **Threshold Tuning**: Start with 0.55, adjust based on results:
   - Lower (0.4-0.5): More files grouped together
   - Higher (0.6-0.7): Stricter, smaller groups
4. **Dashboard Refresh**: The UI auto-refreshes every 5 seconds

### Stopping SEFS

```bash
# Kill backend processes
pkill -f "sefs.main"
pkill -f "streamlit"
```

---

## 🎬 Demo Videos of MVP

> *Add your demo video links here*

### Video 1: Project Overview & Setup
```
[📹 Watch Video: Project Setup Guide]
URL: <Add your video link>
Duration: ~5 minutes
```

### Video 2: Real-Time File Organization Demo
```
[📹 Watch Video: Live Clustering Demo]
URL: <Add your video link>
Duration: ~3 minutes
```

### Video 3: Visual Dashboard Walkthrough
```
[📹 Watch Video: Dashboard Features]
URL: <Add your video link>
Duration: ~4 minutes
```

### Suggested Demo Recording Checklist

- [ ] Show initial unorganized folder
- [ ] Run SEFS main command
- [ ] Demonstrate automatic folder creation
- [ ] Add new files and show real-time reorganization
- [ ] Navigate the Streamlit dashboard
- [ ] Show 2D cluster visualization
- [ ] Demonstrate file search functionality

---

## 🖼️ Demo Images of MVP

> *Replace placeholder paths with actual screenshots*

### 1. Initial Unorganized Folder
![Before SEFS](docs/images/before_sefs.png)
*Files scattered without semantic organization*

### 2. After SEFS Organization
![After SEFS](docs/images/after_sefs.png)
*Automatically created semantic folders*

### 3. Streamlit Dashboard Overview
![Dashboard](docs/images/dashboard_overview.png)
*Main dashboard showing file clusters and statistics*

### 4. 2D Semantic Visualization
![Visualization](docs/images/2d_visualization.png)
*t-SNE projection showing semantic file clusters*

### 5. Cluster Details View
![Cluster Details](docs/images/cluster_details.png)
*Detailed view of files within a semantic cluster*

### 6. File Similarity Matrix
![Similarity Matrix](docs/images/similarity_matrix.png)
*Heatmap showing file-to-file semantic similarity*

---

## 📁 Project Structure

```
VC3/
├── sefs/                       # Main package
│   ├── __init__.py
│   ├── main.py                 # Entry point & orchestrator
│   ├── config.py               # Configuration management
│   ├── content_analyzer.py     # Text extraction & embeddings
│   ├── semantic_engine.py      # Clustering & folder naming
│   ├── database.py             # SQLite persistence layer
│   ├── file_monitor.py         # Watchdog file watcher
│   ├── os_synchronizer.py      # File system operations
│   ├── visual_interface.py     # Streamlit dashboard
│   └── models.py               # Data models
├── tests/                      # Test suite
│   ├── test_config.py
│   ├── test_content_analyzer.py
│   ├── test_semantic_engine.py
│   └── ...
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

---

## 🧪 Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=sefs

# Run specific test file
pytest tests/test_semantic_engine.py -v
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**SEFS Development Team**

---

<p align="center">
  Made with ❤️ and 🤖 AI
</p>
