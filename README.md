# Emotion Highlighting System for Patient Notes

A comparative analysis system for detecting and highlighting emotional content in patient notes using three different approaches:
1. LLM-based API approach
2. Traditional NLP pipeline
3. Custom classical classifier

## Project Structure
```
.
├── README.md
├── requirements.txt
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── models/           # Model implementations
│   │   ├── llm/         # LLM-based implementation
│   │   ├── nlp/         # Traditional NLP implementation
│   │   └── classifier/  # Custom classifier implementation
│   ├── utils/           # Shared utilities
│   └── evaluation/      # Evaluation scripts and metrics
├── tests/               # Test cases
├── data/                # Sample and evaluation datasets
└── notebooks/          # Jupyter notebooks for analysis
```

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```

## Features

- Multi-approach emotion detection in text
- Support for Hebrew text (with English/Russian code-switching)
- Standardized output format for all approaches
- Evaluation framework for comparing approaches
- Visual highlighting system for clinical review

## Emotion Categories

- 🔴 Danger (Red)
- 🟠 Emotional Distress (Orange)
- 🟢 Emotional Progress (Green)
- 🔵 Emotionally Intense (Blue/Bold)

## API Response Format

```json
{
  "text": "Original patient text...",
  "tags": [
    {
      "label": "danger",
      "start": 14,
      "end": 30,
      "text": "I don't want to live"
    }
  ]
}
```

## Evaluation

Run evaluation:
```bash
python -m src.evaluation.run_eval
```

## License

MIT License 

## Running the API

To run the API, use the following command:
```bash
uvicorn src.api.main:app --reload
```

# Emotion Classification Model Training

This project trains an emotion classification model using PyTorch and Transformers.

## Running on RunPod

### Prerequisites
- RunPod account
- Docker installed locally (for testing)

### Setup Steps

1. **Create RunPod Template**
   - Go to RunPod.io
   - Choose "Templates" from the sidebar
   - Click "New Template"
   - Use the following settings:
     - Container Image: `your-dockerhub-username/emotion-analysis:latest`
     - Container Disk: `20GB`
     - Volume Disk: `20GB`
     - Start Command: `python runpod_train.py`

2. **Build and Push Docker Image**
   ```bash
   # Build the image
   docker build -t your-dockerhub-username/emotion-analysis:latest .
   
   # Push to DockerHub
   docker push your-dockerhub-username/emotion-analysis:latest
   ```

3. **Deploy on RunPod**
   - Go to RunPod.io
   - Select "Deploy"
   - Choose your template
   - Select GPU type (recommended: A100 or H100)
   - Click "Deploy"

### Project Structure
```
.
├── data/
│   ├── evaluation_data_filtered.json  # Training data
│   └── results/                       # Results directory
├── src/                              # Source code
├── models/                           # Trained models
├── Dockerfile                        # Docker configuration
├── requirements.txt                  # Python dependencies
├── runpod_train.py                  # RunPod training script
└── README.md                        # This file
```

### Environment Variables
- `WANDB_API_KEY`: (Optional) Weights & Biases API key for logging
- `PYTHONPATH`: Set automatically in Dockerfile

### GPU Requirements
- Minimum VRAM: 24GB
- Recommended GPU: NVIDIA A100 (80GB) or H100
- Supported architectures: CUDA 11.7+

### Training Parameters
- Batch size: 16
- Learning rate: 2e-5
- Epochs: 10
- Model: RoBERTa-base

### Results
Training results will be saved in:
- `models/emotion_classifier/`: Best model checkpoint
- `data/results/evaluation_results.json`: Evaluation metrics

## Local Development

To run locally:
```bash
# Install dependencies
pip install -r requirements.txt

# Run training
python src/train.py
```

## Project Structure

- `src/`: Source code
  - `train.py`: Main training script
  - `models/`: Model implementations
- `data/`: Training and evaluation data
- `requirements.txt`: Python dependencies
- `Dockerfile`: Container configuration 