from typing import List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    Trainer, 
    TrainingArguments,
    AutoModel,
    RobertaTokenizer,
    RobertaForTokenClassification,
    RobertaConfig,
    RobertaModel,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
import numpy as np
from ..base import EmotionAnalyzer, AnalysisResult, EmotionTag, EmotionLabel
from sklearn.metrics import precision_score, recall_score, f1_score
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
import wandb
from tqdm import tqdm
import os
import random  # Add this for random sampling in dataset

class EmotionClassifier(nn.Module):
    def __init__(self, num_labels: int, model_name: str = "roberta-base"):
        super(EmotionClassifier, self).__init__()
        self.num_labels = num_labels
        self.config = RobertaConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            output_attentions=True,
            output_hidden_states=True
        )
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)
        
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        pooled_output = outputs[1]  # Take the [CLS] token representation
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        return logits

class EmotionDataset(Dataset):
    def __init__(self, texts: List[str], tags: Optional[List[List[EmotionTag]]] = None, tokenizer=None):
        self.texts = texts
        self.tags = tags
        self.tokenizer = tokenizer
        
        # Map emotion labels to IDs
        self.label2id = {label.value: idx for idx, label in enumerate(EmotionLabel)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=512,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
            return_special_tokens_mask=True
        )
        
        item = {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
        }
        
        if self.tags is not None:
            # Create label array (one label per token)
            labels = torch.zeros(len(item["input_ids"]), dtype=torch.long)
            special_tokens_mask = encoding["special_tokens_mask"].squeeze().bool()
            
            # Get token offsets
            offsets = encoding["offset_mapping"].squeeze()
            
            # Assign labels based on tags
            for tag in self.tags[idx]:
                # Find tokens that overlap with the tag span
                token_start = None
                token_end = None
                for i, (start, end) in enumerate(offsets):
                    if not special_tokens_mask[i]:  # Skip special tokens
                        if start <= tag.start < end:
                            token_start = i
                        if start < tag.end <= end:
                            token_end = i
                            break
                
                if token_start is not None and token_end is not None:
                    # Apply label smoothing for better generalization
                    context_window = 3  # Increased context window
                    start_idx = max(0, token_start - context_window)
                    end_idx = min(len(labels), token_end + context_window + 1)
                    
                    # Main span gets full label
                    labels[token_start:token_end + 1] = self.label2id[tag.label.value]
                    
                    # Context tokens get weighted label with decay
                    for i in range(start_idx, token_start):
                        if not special_tokens_mask[i]:
                            weight = 1 - (token_start - i) / (context_window + 1)
                            if random.random() < weight:  # Probabilistic labeling
                                labels[i] = self.label2id[tag.label.value]
                    
                    for i in range(token_end + 1, end_idx):
                        if not special_tokens_mask[i]:
                            weight = 1 - (i - token_end) / (context_window + 1)
                            if random.random() < weight:  # Probabilistic labeling
                                labels[i] = self.label2id[tag.label.value]
            
            item["labels"] = labels
            
        return item

class CustomEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self._load_model()

    def _load_model(self):
        if self.model is None:
            try:
                print("[CustomModel] Loading tokenizer...")
                # Ensure we're using RoBERTa tokenizer with explicit vocab files
                try:
                    self.tokenizer = RobertaTokenizer.from_pretrained(
                        self.model_name,
                        model_max_length=512,
                        add_prefix_space=True,  # Important for RoBERTa
                        local_files_only=False  # Allow downloading if needed
                    )
                except Exception as e:
                    print(f"[CustomModel] Error loading RoBERTa tokenizer: {str(e)}")
                    print("[CustomModel] Attempting to load AutoTokenizer as fallback...")
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        model_max_length=512,
                        use_fast=True,
                        local_files_only=False
                    )
                
                # Initialize a new model
                print(f"[CustomModel] Initializing model with {len(EmotionLabel)} labels")
                self.model = EmotionClassifier(
                    num_labels=len(EmotionLabel),
                    model_name=self.model_name
                )
                
                # Initialize model weights if no trained model exists
                model_path = os.getenv("CUSTOM_MODEL_PATH", "./emotion_model/best_model.pt")
                if os.path.exists(model_path):
                    print(f"[CustomModel] Loading trained model from {model_path}")
                    try:
                        state_dict = torch.load(model_path, map_location=self.device)
                        self.model.load_state_dict(state_dict, strict=False)
                        print("[CustomModel] Trained model loaded successfully")
                    except Exception as e:
                        print(f"[CustomModel] Error loading trained model: {str(e)}")
                        print("[CustomModel] Falling back to base model initialization")
                else:
                    print("[CustomModel] No trained model found. Using base model for inference.")
                    # Ensure the model directory exists
                    os.makedirs("./emotion_model", exist_ok=True)
                
                self.model.to(self.device)
                self.model.eval()
                print("[CustomModel] Model loaded and ready for inference")
            except Exception as e:
                print(f"[CustomModel] Error loading model: {str(e)}")
                raise
    
    def analyze(self, text: str) -> AnalysisResult:
        try:
            if self.model is None:
                self._load_model()
            
            self.model.eval()
            print(f"\n[CustomModel] Analyzing text: {text[:100]}...")
            
            with torch.no_grad():
                # Tokenize
                print("[CustomModel] Tokenizing input...")
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=512,
                    padding="max_length",
                    return_tensors="pt",
                    return_offsets_mapping=True
                )
                
                # Move inputs to device
                input_ids = inputs["input_ids"].to(self.device)
                attention_mask = inputs["attention_mask"].to(self.device)
                
                # Get predictions
                print("[CustomModel] Running inference...")
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                # Convert logits to predictions
                if isinstance(logits, tuple):
                    logits = logits[0]
                
                # Get tags
                print("[CustomModel] Converting predictions to tags...")
                tags = self._convert_predictions_to_tags(
                    text,
                    logits[0],
                    attention_mask[0],
                    inputs["offset_mapping"][0]
                )
                
                print(f"[CustomModel] Analysis complete. Found {len(tags)} emotion tags.")
                for tag in tags:
                    print(f"[CustomModel] Found {tag.label}: '{tag.text}' (confidence: {tag.confidence:.2f})")
                
                return AnalysisResult(text=text, tags=tags)
        except Exception as e:
            print(f"[CustomModel] Error during analysis: {str(e)}")
            raise

    def _compute_metrics(self, pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        
        # Flatten the arrays and remove padding tokens
        valid_mask = labels != -100
        labels_flat = labels[valid_mask]
        preds_flat = preds[valid_mask]
        
        # Calculate metrics per class
        metrics = {}
        for label in EmotionLabel:
            label_id = next(i for i, l in enumerate(EmotionLabel) if l == label)
            label_mask = labels_flat == label_id
            if label_mask.sum() > 0:
                metrics[f'{label.value}_precision'] = precision_score(
                    labels_flat == label_id,
                    preds_flat == label_id,
                    average='binary'
                )
                metrics[f'{label.value}_recall'] = recall_score(
                    labels_flat == label_id,
                    preds_flat == label_id,
                    average='binary'
                )
                metrics[f'{label.value}_f1'] = f1_score(
                    labels_flat == label_id,
                    preds_flat == label_id,
                    average='binary'
                )
        
        # Overall metrics
        metrics.update({
            'precision': precision_score(labels_flat, preds_flat, average='weighted'),
            'recall': recall_score(labels_flat, preds_flat, average='weighted'),
            'f1': f1_score(labels_flat, preds_flat, average='weighted')
        })
        
        return metrics

    def train(
        self,
        train_texts: List[str],
        train_tags: List[List[EmotionTag]],
        val_texts: Optional[List[str]] = None,
        val_tags: Optional[List[List[EmotionTag]]] = None,
        output_dir: str = "./emotion_model",
        num_epochs: int = 10,
        batch_size: int = 64,  # Increased for H200
        learning_rate: float = 4e-5,  # Optimized for larger batch
        warmup_ratio: float = 0.1,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        fp16: bool = True,  # Enable mixed precision
        bf16: bool = True,  # Enable bfloat16
        flash_attention: bool = True,  # Enable flash attention
        gradient_checkpointing: bool = True  # Enable gradient checkpointing
    ):
        """Train the model with optimized settings for H200."""
        
        # Initialize datasets
        train_dataset = EmotionDataset(train_texts, train_tags, self.tokenizer)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,  # Increased for faster data loading
            pin_memory=True  # Enable pinned memory for faster GPU transfer
        )
        
        if val_texts and val_tags:
            val_dataset = EmotionDataset(val_texts, val_tags, self.tokenizer)
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size * 2,  # Larger batch size for validation
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
        
        # Configure optimizer with larger batch optimizations
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Enable automatic mixed precision
        scaler = torch.cuda.amp.GradScaler(enabled=fp16)
        
        # Enable flash attention if available
        if flash_attention and hasattr(self.model, 'enable_flash_attention'):
            self.model.enable_flash_attention()
        
        # Enable gradient checkpointing for memory efficiency
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        
        # Training loop with optimizations
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for batch in progress_bar:
                with torch.cuda.amp.autocast(enabled=fp16 or bf16, dtype=torch.bfloat16 if bf16 else torch.float16):
                    loss = self.model(**batch)[0]
                    loss = loss / gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % gradient_accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                
                total_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{total_loss/(batch_idx+1):.4f}"})
            
            # Validation
            if val_texts and val_tags:
                val_loss = self._validate(val_loader, fp16 or bf16, bf16)
                print(f"Validation Loss: {val_loss:.4f}")
        
        # Save the model
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def _validate(self, val_loader, amp_enabled, bf16):
        """Validation with mixed precision."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                with torch.cuda.amp.autocast(enabled=amp_enabled, dtype=torch.bfloat16 if bf16 else torch.float16):
                    loss = self.model(**batch)[0]
                total_loss += loss.item()
        
        return total_loss / len(val_loader)

    def _convert_predictions_to_tags(
        self,
        text: str,
        predictions: torch.Tensor,
        attention_mask: torch.Tensor,
        offset_mapping: torch.Tensor,
        confidence_threshold: float = 0.3  # Lower threshold for base model
    ) -> List[EmotionTag]:
        """Convert token predictions to character-level tags with confidence scores."""
        tags = []
        current_label = None
        start_idx = None
        current_probs = []
        
        # Apply softmax to get probabilities
        probs = F.softmax(predictions, dim=-1)
        
        for idx, (pred, mask, (char_start, char_end)) in enumerate(zip(
            probs, attention_mask, offset_mapping
        )):
            if mask == 0 or char_start == char_end:  # Skip padding and special tokens
                continue
                
            max_prob, label_id = torch.max(pred, dim=0)
            max_prob = max_prob.item()
            label_id = label_id.item()
            
            if max_prob < confidence_threshold:  # Skip low confidence predictions
                if current_label is not None:
                    avg_confidence = sum(current_probs) / len(current_probs)
                    if avg_confidence >= confidence_threshold:
                        tags.append(EmotionTag(
                            label=current_label,
                            start=start_idx,
                            end=int(offset_mapping[idx-1][1]),
                            text=text[start_idx:int(offset_mapping[idx-1][1])],
                            confidence=avg_confidence
                        ))
                    current_label = None
                    current_probs = []
                continue
            
            try:
                predicted_label = EmotionLabel(self.model.config.id2label[label_id])
                if current_label != predicted_label:
                    if current_label is not None:
                        avg_confidence = sum(current_probs) / len(current_probs)
                        if avg_confidence >= confidence_threshold:
                            tags.append(EmotionTag(
                                label=current_label,
                                start=start_idx,
                                end=int(offset_mapping[idx-1][1]),
                                text=text[start_idx:int(offset_mapping[idx-1][1])],
                                confidence=avg_confidence
                            ))
                    current_label = predicted_label
                    start_idx = int(char_start)
                    current_probs = [max_prob]
                else:
                    current_probs.append(max_prob)
            except ValueError as e:
                print(f"Error converting label ID {label_id}: {str(e)}")
                continue
        
        # Handle the last tag if it exists
        if current_label is not None:
            avg_confidence = sum(current_probs) / len(current_probs)
            if avg_confidence >= confidence_threshold:
                tags.append(EmotionTag(
                    label=current_label,
                    start=start_idx,
                    end=int(offset_mapping[-1][1]),
                    text=text[start_idx:int(offset_mapping[-1][1])],
                    confidence=avg_confidence
                ))
        
        return tags
    
    def batch_analyze(self, texts: List[str]) -> List[AnalysisResult]:
        self._load_model()
        self.model.eval()
        
        dataset = EmotionDataset(texts, tokenizer=self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=8, num_workers=4)
        
        results = []
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"]
                )
                
                for idx, (text, pred_logits, mask) in enumerate(zip(
                    texts[len(results):len(results) + len(logits)],
                    logits if isinstance(logits, torch.Tensor) else logits[0],
                    batch["attention_mask"]
                )):
                    tags = self._convert_predictions_to_tags(text, pred_logits, mask)
                    results.append(AnalysisResult(text=text, tags=tags))
        
        return results 