from typing import List, Dict, Set, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np
from ..models.base import AnalysisResult, EmotionTag, EmotionLabel

@dataclass
class SpanEvaluation:
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    
    @property
    def precision(self) -> float:
        if self.true_positives + self.false_positives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_positives)
    
    @property
    def recall(self) -> float:
        if self.true_positives + self.false_negatives == 0:
            return 0.0
        return self.true_positives / (self.true_positives + self.false_negatives)
    
    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0.0
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

@dataclass
class EvaluationMetrics:
    per_label: Dict[EmotionLabel, SpanEvaluation]
    overall: SpanEvaluation
    
    def print_report(self):
        """Print a formatted evaluation report."""
        print("\nEvaluation Report")
        print("=" * 60)
        print(f"{'Label':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 60)
        
        for label, eval in self.per_label.items():
            print(f"{label.value:<20} {eval.precision:>10.3f} {eval.recall:>10.3f} {eval.f1:>10.3f}")
        
        print("-" * 60)
        print(f"{'Overall':<20} {self.overall.precision:>10.3f} {self.overall.recall:>10.3f} {self.overall.f1:>10.3f}")

def _has_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """Check if two spans overlap."""
    return not (span1[1] <= span2[0] or span2[1] <= span1[0])

def _compute_span_matches(
    predicted: List[EmotionTag],
    ground_truth: List[EmotionTag],
    iou_threshold: float = 0.5
) -> Tuple[Set[int], Set[int]]:
    """Compute matching spans between predicted and ground truth tags."""
    matched_pred = set()
    matched_truth = set()
    
    # Sort by start position for efficient matching
    pred_spans = [(i, (t.start, t.end)) for i, t in enumerate(predicted)]
    truth_spans = [(i, (t.start, t.end)) for i, t in enumerate(ground_truth)]
    
    pred_spans.sort(key=lambda x: x[1][0])
    truth_spans.sort(key=lambda x: x[1][0])
    
    i, j = 0, 0
    while i < len(pred_spans) and j < len(truth_spans):
        pred_idx, pred_span = pred_spans[i]
        truth_idx, truth_span = truth_spans[j]
        
        if _has_overlap(pred_span, truth_span):
            # Compute IoU
            intersection = (
                max(pred_span[0], truth_span[0]),
                min(pred_span[1], truth_span[1])
            )
            union = (
                min(pred_span[0], truth_span[0]),
                max(pred_span[1], truth_span[1])
            )
            
            intersection_length = intersection[1] - intersection[0]
            union_length = union[1] - union[0]
            iou = intersection_length / union_length
            
            if (iou >= iou_threshold and
                predicted[pred_idx].label == ground_truth[truth_idx].label):
                matched_pred.add(pred_idx)
                matched_truth.add(truth_idx)
        
        # Advance the pointer with the earlier end position
        if pred_span[1] < truth_span[1]:
            i += 1
        else:
            j += 1
    
    return matched_pred, matched_truth

def evaluate_results(
    predicted_results: List[AnalysisResult],
    ground_truth_results: List[AnalysisResult],
    iou_threshold: float = 0.5
) -> EvaluationMetrics:
    """Evaluate predicted results against ground truth.
    
    Args:
        predicted_results: List of predicted analysis results
        ground_truth_results: List of ground truth analysis results
        iou_threshold: Minimum IoU for considering spans as matching
        
    Returns:
        EvaluationMetrics containing precision, recall, and F1 scores
    """
    per_label_eval = {label: SpanEvaluation() for label in EmotionLabel}
    overall_eval = SpanEvaluation()
    
    for pred_result, truth_result in zip(predicted_results, ground_truth_results):
        # Group tags by label
        pred_by_label = defaultdict(list)
        truth_by_label = defaultdict(list)
        
        for tag in pred_result.tags:
            pred_by_label[tag.label].append(tag)
        for tag in truth_result.tags:
            truth_by_label[tag.label].append(tag)
        
        # Evaluate each label separately
        for label in EmotionLabel:
            pred_tags = pred_by_label[label]
            truth_tags = truth_by_label[label]
            
            matched_pred, matched_truth = _compute_span_matches(
                pred_tags, truth_tags, iou_threshold
            )
            
            # Update per-label metrics
            per_label_eval[label].true_positives += len(matched_pred)
            per_label_eval[label].false_positives += len(pred_tags) - len(matched_pred)
            per_label_eval[label].false_negatives += len(truth_tags) - len(matched_truth)
            
            # Update overall metrics
            overall_eval.true_positives += len(matched_pred)
            overall_eval.false_positives += len(pred_tags) - len(matched_pred)
            overall_eval.false_negatives += len(truth_tags) - len(matched_truth)
    
    return EvaluationMetrics(per_label_eval, overall_eval) 