import os
import json
from typing import List, Dict
from dotenv import load_dotenv
from ..models.base import AnalysisResult, EmotionTag, EmotionLabel
from ..models.llm.analyzer import LLMEmotionAnalyzer
from ..models.nlp.analyzer import NLPEmotionAnalyzer
from ..models.classifier.analyzer import CustomEmotionAnalyzer
from .metrics import evaluate_results

load_dotenv()

def load_evaluation_data(data_path: str) -> List[Dict]:
    """Load evaluation dataset from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        # Only load first 100 entries for testing
        return data[:100]

def convert_to_analysis_result(data: Dict) -> AnalysisResult:
    """Convert JSON data to AnalysisResult object."""
    tags = [
        EmotionTag(
            label=EmotionLabel(tag["label"]),
            start=tag["start"],
            end=tag["end"],
            text=tag["text"]
        )
        for tag in data.get("tags", [])  # Use get() with default empty list
    ]
    return AnalysisResult(text=data["text"], tags=tags)

def main():
    # Initialize analyzers
    nlp_analyzer = NLPEmotionAnalyzer()
    
    # Load evaluation data
    eval_data = load_evaluation_data("data/evaluation_data.json")
    ground_truth = [convert_to_analysis_result(data) for data in eval_data]
    texts = [data["text"] for data in eval_data]
    
    print("Running NLP analyzer...")
    nlp_results = nlp_analyzer.batch_analyze(texts)
    
    print("\nEvaluating NLP approach:")
    nlp_metrics = evaluate_results(nlp_results, ground_truth)
    nlp_metrics.print_report()
    
    # Save detailed results
    results = {
        "nlp": {
            "metrics": {
                "overall": {
                    "precision": nlp_metrics.overall.precision,
                    "recall": nlp_metrics.overall.recall,
                    "f1": nlp_metrics.overall.f1
                },
                "per_label": {
                    label.value: {
                        "precision": eval.precision,
                        "recall": eval.recall,
                        "f1": eval.f1
                    }
                    for label, eval in nlp_metrics.per_label.items()
                }
            },
            "predictions": [
                {
                    "text": result.text,
                    "tags": [tag.__dict__ for tag in result.tags]
                }
                for result in nlp_results
            ]
        }
    }
    
    # Save results
    os.makedirs("data/results", exist_ok=True)
    with open("data/results/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\nDetailed results saved to data/results/evaluation_results.json")

if __name__ == "__main__":
    main() 