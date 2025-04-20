from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
from enum import Enum

class EmotionLabel(str, Enum):
    DANGER = "danger"
    DISTRESS = "emotional_distress"
    PROGRESS = "emotional_progress"
    INTENSE = "emotionally_intense"

@dataclass
class EmotionTag:
    label: EmotionLabel
    start: int
    end: int
    text: str
    confidence: Optional[float] = None

@dataclass
class AnalysisResult:
    text: str
    tags: List[EmotionTag]

class EmotionAnalyzer(ABC):
    """Base class for all emotion analysis approaches."""
    
    @abstractmethod
    def analyze(self, text: str) -> AnalysisResult:
        """Analyze text and return emotional tags.
        
        Args:
            text: The input text to analyze
            
        Returns:
            AnalysisResult containing the original text and emotion tags
        """
        pass
    
    @abstractmethod
    def batch_analyze(self, texts: List[str]) -> List[AnalysisResult]:
        """Analyze multiple texts in batch.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            List of AnalysisResult objects
        """
        pass 