import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
from typing import List, Dict, Tuple
import random
from dotenv import load_dotenv
from models.base import EmotionTag, EmotionLabel
from models.classifier.analyzer import CustomEmotionAnalyzer
import nlpaug.augmenter.word as naw
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def load_data(data_path: str) -> List[Dict]:
    """Load all data from JSON file."""
    with open(data_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def convert_to_tags(data: Dict) -> List[EmotionTag]:
    """Convert JSON data to EmotionTag objects."""
    return [
        EmotionTag(
            label=EmotionLabel(tag["label"]),
            start=tag["start"],
            end=tag["end"],
            text=tag["text"]
        )
        for tag in data.get("tags", [])
    ]

def get_text_label_distribution(text_tags: List[EmotionTag]) -> str:
    """Get a string representation of the label distribution for stratification."""
    if not text_tags:
        return "O"
    counter = Counter(tag.label.value for tag in text_tags)
    return "_".join(f"{label}_{count}" for label, count in sorted(counter.items()))

def augment_data(
    texts: List[str],
    tags: List[List[EmotionTag]],
    num_augmentations: int = 3
):
    """Augment training data while preserving emotion tags."""
    logger.info("Initializing RoBERTa-based augmentation...")
    
    # Initialize only RoBERTa-based augmenter
    augmenter = naw.ContextualWordEmbsAug(
        model_path='roberta-base',
        action="substitute",
        aug_p=0.3,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        batch_size=32,
        top_k=20  # Increase diversity of substitutions
    )
    
    augmented_texts = []
    augmented_tags = []
    
    for idx, (text, text_tags) in enumerate(tqdm(zip(texts, tags), total=len(texts))):
        if len(text.split()) < 5 or not text_tags:
            continue
            
        try:
            # Apply augmentation multiple times
            for _ in range(num_augmentations):
                aug_text = augmenter.augment(text)[0]
                aug_tags = []
                
                for tag in text_tags:
                    tag_text = tag.text.lower()
                    aug_text_lower = aug_text.lower()
                    
                    if tag_text in aug_text_lower:
                        start_idx = aug_text_lower.find(tag_text)
                        end_idx = start_idx + len(tag_text)
                        
                        aug_tags.append(EmotionTag(
                            label=tag.label,
                            start=start_idx,
                            end=end_idx,
                            text=aug_text[start_idx:end_idx]
                        ))
                
                if aug_tags:
                    augmented_texts.append(aug_text)
                    augmented_tags.append(aug_tags)
                    
        except Exception as e:
            logger.warning(f"Augmentation failed for text {idx}: {str(e)}")
            continue
    
    logger.info(f"Generated {len(augmented_texts)} augmented examples")
    return augmented_texts, augmented_tags

def split_data(
    data: List[Dict],
    n_splits: int = 5
) -> List[Tuple[List[Dict], List[Dict]]]:
    """Split data using stratified k-fold cross validation."""
    label_distributions = [
        get_text_label_distribution([EmotionTag(
            label=EmotionLabel(tag["label"]),
            start=tag["start"],
            end=tag["end"],
            text=tag["text"]
        ) for tag in item.get("tags", [])])
        for item in data
    ]
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    splits = []
    for train_idx, val_idx in skf.split(data, label_distributions):
        train_data = [data[i] for i in train_idx]
        val_data = [data[i] for i in val_idx]
        splits.append((train_data, val_data))
    
    return splits

def main():
    # Set random seed for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    # Load and prepare data
    logger.info("Loading data...")
    data = load_data("data/evaluation_data_filtered.json")
    
    # Perform k-fold cross validation
    splits = split_data(data, n_splits=5)
    
    best_f1 = 0
    best_model_fold = 0
    
    for fold, (train_data, val_data) in enumerate(splits, 1):
        logger.info(f"\nTraining fold {fold}/5")
        
        # Extract texts and tags
        train_texts = [item["text"] for item in train_data]
        train_tags = [convert_to_tags(item) for item in train_data]
        
        val_texts = [item["text"] for item in val_data]
        val_tags = [convert_to_tags(item) for item in val_data]

        # Augment training data
        aug_texts, aug_tags = augment_data(
            train_texts,
            train_tags,
            num_augmentations=3
        )
        train_texts.extend(aug_texts)
        train_tags.extend(aug_tags)
        
        logger.info(f"Training set size (with augmentation): {len(train_texts)}")
        logger.info(f"Validation set size: {len(val_texts)}")
        
        # Initialize and train model
        logger.info("\nInitializing model...")
        model = CustomEmotionAnalyzer()
        
        # Train with optimized parameters for H200
        logger.info("\nStarting training...")
        model.train(
            train_texts=train_texts,
            train_tags=train_tags,
            val_texts=val_texts,
            val_tags=val_tags,
            output_dir=f"models/emotion_classifier_fold{fold}",
            num_epochs=10,
            batch_size=32,  # Increased for H200
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01
        )
        
        # Evaluate on validation set
        logger.info("\nEvaluating fold...")
        results = model.batch_analyze(val_texts)
        
        # Calculate metrics
        correct = 0
        total = 0
        for pred, true in zip(results, val_tags):
            pred_set = {(tag.label, tag.start, tag.end) for tag in pred.tags}
            true_set = {(tag.label, tag.start, tag.end) for tag in true}
            
            correct += len(pred_set.intersection(true_set))
            total += len(true_set)
        
        precision = correct / (sum(len(r.tags) for r in results) + 1e-10)
        recall = correct / (total + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        logger.info(f"\nFold {fold} Metrics:")
        logger.info(f"Precision: {precision:.4f}")
        logger.info(f"Recall: {recall:.4f}")
        logger.info(f"F1 Score: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model_fold = fold
            logger.info(f"\nNew best model found in fold {fold}!")
            
            # Copy best model to final directory
            os.system(f"cp -r models/emotion_classifier_fold{fold}/* models/emotion_classifier/")
    
    logger.info(f"\nTraining completed! Best model from fold {best_model_fold} with F1 score: {best_f1:.4f}")
    logger.info("Final model saved to models/emotion_classifier")

if __name__ == "__main__":
    main() 