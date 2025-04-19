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

1. Create a RunPod account at https://www.runpod.io/

2. Create a new pod:
   - Select "Secure Cloud" or "Community Cloud"
   - Choose a GPU instance (recommended: A100 or H100 for faster training)
   - Select Ubuntu 22.04 as the base image
   - Set the container image to your Docker Hub repository (after building and pushing the image)

3. Build and push the Docker image:
```bash
# Build the image
docker build -t your-username/emotion-classifier:latest .

# Push to Docker Hub
docker push your-username/emotion-classifier:latest
```

4. Configure the pod:
   - Set the container image to your pushed image
   - Set the command to: `python3 src/train.py`
   - Configure storage volumes if needed
   - Set environment variables if required

5. Start the pod and monitor training progress through the RunPod console

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