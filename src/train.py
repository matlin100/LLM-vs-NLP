import os
import json
from typing import List, Dict, Tuple
import random
from dotenv import load_dotenv
from models.base import EmotionTag, EmotionLabel
from models.classifier.analyzer import CustomEmotionAnalyzer
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter

load_dotenv()
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

print("Loading translation models manually...")

en_de_model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-en-de").to("cuda")
en_de_tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-en-de")

de_en_model = FSMTForConditionalGeneration.from_pretrained("facebook/wmt19-de-en").to("cuda")
de_en_tokenizer = FSMTTokenizer.from_pretrained("facebook/wmt19-de-en")


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

def augment_data(texts: List[str], tags: List[List[EmotionTag]], num_augmentations: int = 3):
    """Augment training data while preserving emotion tags."""
    print("Augmenting training data...")
    
    # Initialize augmenters with more diverse options
    augmenters = [
        # Synonym replacement
        naw.SynonymAug(
            aug_src='wordnet',
            aug_p=0.3,
            aug_min=1,
            aug_max=10
        ),
        # Contextual word embeddings
        naw.ContextualWordEmbsAug(
            model_path='roberta-base',
            action="substitute",
            aug_p=0.3
        ),
        # Back translation through multiple languages
        naw.BackTranslationAug(
            from_model_name='facebook/wmt19-en-de',
            to_model_name='facebook/wmt19-de-en',
            device='cuda',
            src_model=en_de_model,
            src_tokenizer=en_de_tokenizer,
            tgt_model=de_en_model,
            tgt_tokenizer=de_en_tokenizer
        ),
        # Random insertion
        naw.RandomWordAug(
            action="insert",
            aug_p=0.3
        ),
        # Sentence-level augmentation
        nas.ContextualWordEmbsForSentenceAug(
            model_path='roberta-base',
            device='cuda'
        )
    ]
    
    augmented_texts = []
    augmented_tags = []
    
    for idx, (text, text_tags) in enumerate(tqdm(zip(texts, tags), total=len(texts))):
        # Skip augmentation for very short texts or those without tags
        if len(text.split()) < 5 or not text_tags:
            continue
            
        try:
            # Apply each augmenter
            for augmenter in augmenters:
                for _ in range(num_augmentations):
                    aug_text = augmenter.augment(text)[0]
                    
                    # Adjust tag positions for augmented text
                    aug_tags = []
                    for tag in text_tags:
                        # Find the augmented position of the tagged text
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
                    
                    if aug_tags:  # Only add if we could preserve some tags
                        augmented_texts.append(aug_text)
                        augmented_tags.append(aug_tags)
                        
        except Exception as e:
            print(f"Augmentation failed for text {idx}: {str(e)}")
            continue
    
    print(f"Generated {len(augmented_texts)} augmented examples")
    return augmented_texts, augmented_tags

def split_data(
    data: List[Dict],
    n_splits: int = 5
) -> List[Tuple[List[Dict], List[Dict]]]:
    """Split data using stratified k-fold cross validation."""
    # Create label distribution for stratification
    label_distributions = [
        get_text_label_distribution([EmotionTag(
            label=EmotionLabel(tag["label"]),
            start=tag["start"],
            end=tag["end"],
            text=tag["text"]
        ) for tag in item.get("tags", [])])
        for item in data
    ]
    
    # Initialize k-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Generate splits
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
    
    # Load and prepare data
    print("Loading data...")
    data = load_data("data/evaluation_data_filtered.json")
    
    # Perform k-fold cross validation
    splits = split_data(data, n_splits=5)
    
    best_f1 = 0
    best_model_fold = 0
    
    for fold, (train_data, val_data) in enumerate(splits, 1):
        print(f"\nTraining fold {fold}/5")
        
        # Extract texts and tags
        train_texts = [item["text"] for item in train_data]
        train_tags = [convert_to_tags(item) for item in train_data]
        
        val_texts = [item["text"] for item in val_data]
        val_tags = [convert_to_tags(item) for item in val_data]
        
        # Augment training data
        aug_texts, aug_tags = augment_data(train_texts, train_tags)
        train_texts.extend(aug_texts)
        train_tags.extend(aug_tags)
        
        print(f"Training set size (with augmentation): {len(train_texts)}")
        print(f"Validation set size: {len(val_texts)}")
        
        # Initialize and train model
        print("\nInitializing model...")
        model = CustomEmotionAnalyzer()
        
        # Train with longer epochs and advanced techniques
        print("\nStarting training...")
        model.train(
            train_texts=train_texts,
            train_tags=train_tags,
            val_texts=val_texts,
            val_tags=val_tags,
            output_dir=f"models/emotion_classifier_fold{fold}",
            num_epochs=10,
            batch_size=16,
            learning_rate=2e-5,
            warmup_ratio=0.1,
            weight_decay=0.01
        )
        
        # Evaluate on validation set
        print("\nEvaluating fold...")
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
        
        print(f"\nFold {fold} Metrics:")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Save best model
        if f1 > best_f1:
            best_f1 = f1
            best_model_fold = fold
            print(f"\nNew best model found in fold {fold}!")
            
            # Copy best model to final directory
            os.system(f"cp -r models/emotion_classifier_fold{fold}/* models/emotion_classifier/")
    
    print(f"\nTraining completed! Best model from fold {best_model_fold} with F1 score: {best_f1:.4f}")
    print("Final model saved to models/emotion_classifier")

if __name__ == "__main__":
    main() 