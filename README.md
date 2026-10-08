# Math Physics AI Tutor - Production Ready with RAG & Fine-tuning

A complete, production-ready tutoring system for mathematics and physics with:

- **RAG (Retrieval-Augmented Generation)**: Document retrieval with FAISS embeddings
- **LLM Integration**: Hugging Face models (Phi-3, Llama, etc.)
- **Web UI**: Interactive Streamsync-based interface
- **Training Pipeline**: Fine-tuning models on custom math/physics datasets
- **API**: FastAPI backend with health checks and chat endpoints

## Features

### Core
- ✅ Question answering for math and physics
- ✅ Step-by-step explanations
- ✅ Difficulty-aware responses (beginner, intermediate, advanced)
- ✅ Subject-aware logic

### RAG System
- ✅ Document embedding with Sentence Transformers
- ✅ FAISS vector search
- ✅ PDF and text file support
- ✅ Context injection into answers

### LLM Integration
- ✅ Local Hugging Face model support
- ✅ LoRA fine-tuning for efficient adaptation
- ✅ Mock backend for local development
- ✅ Auto device mapping

### Web UI
- ✅ Interactive chat interface
- ✅ Subject and difficulty selection
- ✅ RAG status indicator
- ✅ Conversation history
- ✅ Real-time API communication

### Training Pipeline
- ✅ Data loading (JSONL, CSV)
- ✅ Dataset splitting (train/val/test)
- ✅ Model training with Hugging Face Trainer
- ✅ LoRA support for efficient fine-tuning
- ✅ Checkpoint management

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/varspaga-png/Math-Physics-AI-Tutor.git
cd Math-Physics-AI-Tutor

python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the API

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 3. Start the Web UI

```bash
# In another terminal
cd ui
streamsync run app.py --host 0.0.0.0 --port 3000
```

The UI will be available at `http://localhost:3000`

### 4. Test the API

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the derivative of x^2",
    "subject": "math",
    "difficulty": "beginner"
  }'
```

## Loading RAG Documents

### Option 1: Via API

```bash
curl -X POST http://localhost:8000/api/v1/load-rag-directory \
  -H "Content-Type: application/json" \
  -d '{
    "directory": "./data",
    "pattern": "*.txt"
  }'
```

### Option 2: Via Script

```bash
python scripts/load_rag_documents.py ./data --output ./rag_index
```

Place your documents in the `data/` directory as `.txt`, `.pdf`, or `.md` files.

## Fine-tuning Models

### 1. Prepare Training Data

Create a JSONL or CSV file with your training examples:

```json
{"question": "What is x^2?", "answer": "x^2 is x multiplied by itself", "subject": "math", "difficulty": "beginner"}
{"question": "What is F=ma?", "answer": "Newton's Second Law relates force, mass, and acceleration", "subject": "physics", "difficulty": "beginner"}
```

### 2. Run Training

```bash
python scripts/train_model.py data/training_data.jsonl \
  --model microsoft/Phi-3-mini-4k-instruct \
  --output ./fine_tuned_model \
  --epochs 3 \
  --batch-size 4 \
  --learning-rate 2e-4
```

### 3. Use Fine-tuned Model

Update `.env` or code to use your fine-tuned model:

```bash
MODEL_PROVIDER=huggingface
HF_MODEL_NAME=./fine_tuned_model
```

Then restart the API.

## Project Structure

```
.
├── app/
│   ├── api/
│   │   └── routes.py              # FastAPI endpoints
│   ├── rag/
│   │   ├── embeddings.py          # FAISS embeddings
│   │   ├── document_loader.py     # PDF/text loading
│   │   └── retriever.py           # RAG retriever
│   ├── services/
│   │   ├── model_manager.py       # Model abstraction
│   │   ├── tutor_service.py       # Base tutor logic
│   │   └── rag_tutor_service.py   # RAG-enhanced tutor
│   ├── training/
│   │   ├── data_processor.py      # Data loading & prep
│   │   └── trainer.py             # Model training
│   ├── config.py                  # Configuration
│   ├── schemas.py                 # Request/response models
│   └── main.py                    # App setup
├── ui/
│   ├── app.py                     # Streamsync UI
│   └── requirements.txt
├── scripts/
│   ├── train_model.py             # Training script
│   └── load_rag_documents.py      # RAG loading script
├── data/
│   └── sample_training_data.jsonl # Sample data
├── tests/                         # Test suite
├── Dockerfile                     # API container
├── Dockerfile.ui                  # UI container
├── docker-compose.yml             # Multi-container setup
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Docker Deployment

### Run with Docker Compose

```bash
docker-compose up
```

This will start:
- API on `http://localhost:8000`
- UI on `http://localhost:3000`

### Build Individually

```bash
# API
docker build -t math-physics-tutor-api .
docker run -p 8000:8000 math-physics-tutor-api

# UI
docker build -f Dockerfile.ui -t math-physics-tutor-ui .
docker run -p 3000:3000 math-physics-tutor-ui
```

## Environment Variables

Create a `.env` file:

```bash
# Model configuration
MODEL_PROVIDER=mock              # or 'huggingface'
HF_MODEL_NAME=microsoft/Phi-3-mini-4k-instruct

# App configuration
APP_HOST=0.0.0.0
APP_PORT=8000
DEBUG=false
```

## API Endpoints

### Health Check
```
GET /api/v1/health
```

Response:
```json
{
  "status": "ok",
  "service": "math-physics-ai-tutor",
  "version": "0.1.0",
  "model_provider": "mock",
  "rag_enabled": false
}
```

### Ask Question
```
POST /api/v1/ask
```

Request:
```json
{
  "question": "What is the derivative of x^2?",
  "subject": "math",
  "difficulty": "beginner",
  "context": null
}
```

Response:
```json
{
  "answer": "...",
  "subject": "math",
  "difficulty": "beginner",
  "model_provider": "mock",
  "steps": ["...", "..."]
}
```

### Load RAG Documents
```
POST /api/v1/load-rag-directory
```

Request:
```json
{
  "directory": "./data",
  "pattern": "*.txt"
}
```

## Testing

```bash
pytest tests/
```

## Supported Models

- Microsoft Phi-3 (recommended for local deployment)
- Meta Llama-2
- OpenAI GPT-2 / GPT-3
- Hugging Face compatible models

## Performance Notes

- **Mock Backend**: Fast, no GPU required, for development
- **Phi-3-mini**: ~4GB VRAM, ~1-2 sec per query
- **RAG Retrieval**: <100ms for 1000 documents on CPU
- **Fine-tuning**: 3-6 hours on single GPU for 1000 examples

## Troubleshooting

### Out of Memory
Reduce batch size or use gradient accumulation:
```bash
python scripts/train_model.py data.jsonl --batch-size 1
```

### Slow Inference
Use quantized models or enable model optimization:
```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="float16",  # Reduced precision
)
```

### RAG Not Working
Ensure documents are in the correct directory:
```bash
ls -la data/
ls -la rag_index/
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

MIT

## Citation

If you use this project, please cite:

```bibtex
@software{math_physics_tutor,
  title={Math Physics AI Tutor},
  author={Your Name},
  year={2025},
  url={https://github.com/varspaga-png/Math-Physics-AI-Tutor}
}
```

## Support

For issues and questions:
- Open an issue on GitHub
- Check existing discussions
- Review documentation

---

**Last Updated**: 2025
**Status**: Production Ready ✅
