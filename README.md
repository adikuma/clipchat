# Multimodal RAG for Video

A system that processes videos to enable natural language questions about video content using both visual and textual information.

## What is Multimodal RAG?

Traditional RAG (Retrieval-Augmented Generation) systems only work with text. Multimodal RAG extends this to handle images, videos, and audio by representing all content types in the same vector space.

For videos, this means you can ask questions like "What is BitNet?" and get answers based on both what was said (transcript) and what was shown (video frames).

## How It Works

### 1. Video Processing Pipeline

* **Download** : Extract video from YouTube URL
* **Transcription** : Use Whisper to generate accurate transcripts with timestamps
* **Frame Extraction** : Extract key frames that correspond to transcript segments
* **Chunking** : Combine multiple transcript segments for better context

### 2. Multimodal Embedding Generation

* **BridgeTower Model** : Processes both image and text together
* **Shared Vector Space** : Creates embeddings where related visual and textual content are positioned close together
* **Cross-Modal Understanding** : Links what is spoken with what is shown

### 3. Vector Storage and Retrieval

* **ChromaDB** : Stores embeddings with metadata (timestamps, frame paths, transcripts)
* **Semantic Search** : Find relevant video segments based on question similarity
* **Context Retrieval** : Return both transcript text and corresponding video frames

### 4. Response Generation

* **Query Processing** : Convert user question to embedding
* **Relevant Segment Retrieval** : Find most similar video segments
* **Context Augmentation** : Combine retrieved transcripts with user question
* **LLM Response** : Generate answer using GPT with video context

## Technical Architecture

```
Video URL → Download → Transcription → Frame Extraction → Chunking
    ↓
Multimodal Embeddings (BridgeTower) → Vector Storage (ChromaDB)
    ↓
User Question → Query Embedding → Similarity Search → Context Retrieval
    ↓
Prompt Augmentation → LLM (GPT) → Final Answer
```

## Installation and Setup

### Prerequisites

* Python 3.8+
* OpenAI API key
* GPU recommended for faster processing

### Install Dependencies

```bash
pip install torch transformers opencv-python webvtt-py whisper moviepy yt-dlp chromadb langchain langchain-openai langchain-community openai python-dotenv pillow click rich numpy tqdm
```

### Environment Setup

Create `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Basic Workflow

1. **Process a video:**

```bash
python -m src.cli process --url "https://youtube.com/watch?v=VIDEO_ID"
```

2. **Ask questions:**

```bash
python -m src.cli ask --query "What is the main topic discussed?"
```

3. **Get video summary:**

```bash
python -m src.cli summarize
```

### Interactive Mode

```bash
python -m src.cli interactive --url "https://youtube.com/watch?v=VIDEO_ID"
```

Then ask questions directly:

```
clipchat> What is BitNet?
clipchat> Explain the key concepts
clipchat> What happens at 5:30?
clipchat> summary
```

### Advanced Options

**Custom chunking for better context:**

```bash
python -m src.cli process --url "..." --chunk-size 10 --stride 2
```

**Search specific video:**

```bash
python -m src.cli ask --query "..." --video-id "specific_video"
```

**More detailed results:**

```bash
python -m src.cli ask --query "..." --results 5
```

### Database Management

```bash
# Show statistics
python -m src.cli stats

# Clear all data
python -m src.cli clear
```

## Configuration Options

### Chunking Parameters

* `chunk-size`: Number of transcript segments to combine (default: 7)
* `stride`: Step size between chunks, smaller = more overlap (default: 3)

For technical content with complex topics, use larger chunks:

```bash
--chunk-size 10 --stride 2
```

For conversational content:

```bash
--chunk-size 5 --stride 4
```

### Models

* **Embedding** : BridgeTower/bridgetower-base (multimodal)
* **Transcription** : Whisper small model
* **LLM** : GPT-4.1

## File Structure

```
src/
├── video_processor.py    # Download, transcribe, extract frames
├── embeddings.py         # BridgeTower multimodal embeddings
├── database.py          # ChromaDB vector operations
├── rag.py               # Complete RAG pipeline
└── cli.py               # Command line interface

data/
├── videos/              # Downloaded videos
├── frames/              # Extracted video frames
├── metadata/            # Video segment metadata
└── chroma_db/           # Vector database
```
