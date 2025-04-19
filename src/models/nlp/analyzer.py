from typing import List, Dict, Set
import spacy
from spacy.matcher import Matcher, PhraseMatcher
from spacy.tokens import Doc, Span
from ..base import EmotionAnalyzer, AnalysisResult, EmotionTag, EmotionLabel

# Emotion lexicons - these would be expanded with domain expertise
EMOTION_PATTERNS = {
    EmotionLabel.DANGER: {
        "phrases": [
            "kill myself", "end it all", "suicide", "hurt myself", "self harm",
            "don't want to live", "want to die", "end my life", "take my life",
            "harm myself", "cut myself", "thinking of ending", "considering suicide",
            "disappearing would be easier", "wondering if disappearing"
        ],
        "keywords": [
            "death", "die", "dead", "knife", "pills", "overdose", "suicidal",
            "dangerous", "harmful", "fatal", "lethal", "danger", "risk",
            "disappearing", "disappear"
        ]
    },
    EmotionLabel.DISTRESS: {
        "phrases": [
            "can't handle", "too much", "overwhelmed", "breaking down",
            "feeling down", "really down", "very anxious", "so depressed",
            "can't cope", "losing hope", "giving up", "at my limit",
            "feeling anxious", "stressed out", "under pressure", "feeling overwhelmed",
            "struggling with", "having trouble", "difficult time", "hard to deal with",
            "feeling utterly broken", "feeling broken", "invisible to everyone",
            "hands tremble", "heart races", "haunting dreams", "memories haunt"
        ],
        "keywords": [
            "anxiety", "depression", "panic", "scared", "afraid", "worried",
            "stress", "distress", "suffering", "struggle", "pain", "hurt",
            "lonely", "hopeless", "helpless", "worthless", "miserable",
            "nervous", "fear", "concern", "tension", "pressure", "difficulty",
            "problem", "issue", "challenge", "trouble", "burden", "broken",
            "invisible", "tremble", "haunt"
        ]
    },
    EmotionLabel.PROGRESS: {
        "phrases": [
            "feeling better", "making progress", "getting better", "more hopeful",
            "starting to improve", "seeing improvement", "doing better",
            "managed to", "accomplished", "proud of myself", "small victory",
            "feeling good", "making strides", "moving forward", "on the right track",
            "positive change", "better days", "improving", "recovering",
            "getting there", "making it work", "rebuild herself", "began to rebuild"
        ],
        "keywords": [
            "improvement", "hope", "better", "positive", "progress",
            "achievement", "success", "victory", "proud", "accomplished",
            "stronger", "healing", "recovering", "improving", "growth",
            "development", "advancement", "breakthrough", "milestone",
            "success", "achievement", "accomplishment", "rebuild", "began"
        ]
    },
    EmotionLabel.INTENSE: {
        "phrases": [
            "absolutely", "completely", "extremely", "totally",
            "so much", "very much", "really need", "desperately",
            "can't stand", "can't take", "completely overwhelmed",
            "totally stressed", "extremely anxious", "absolutely necessary",
            "really important", "very serious", "quite difficult",
            "utterly broken", "pretending everything"
        ],
        "keywords": [
            "never", "always", "must", "need", "terrible", "horrible",
            "unbearable", "overwhelming", "extreme", "intense", "severe",
            "desperate", "critical", "urgent", "crucial", "essential",
            "vital", "important", "significant", "major", "serious",
            "utterly", "everything"
        ]
    }
}

class NLPEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.nlp = spacy.load(model_name)
        self._setup_matchers()
        print("NLP Analyzer initialized successfully")
    
    def _setup_matchers(self):
        """Set up phrase and pattern matchers for each emotion category."""
        self.phrase_matchers = {}
        self.pattern_matchers = {}
        
        # Set up phrase matchers
        for label in EmotionLabel:
            phrase_matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            patterns = [self.nlp.make_doc(text.lower()) 
                       for text in EMOTION_PATTERNS[label]["phrases"]]
            phrase_matcher.add(label.value, patterns)
            self.phrase_matchers[label] = phrase_matcher
            print(f"Added {len(patterns)} phrase patterns for {label}")
        
        # Set up keyword pattern matchers with context
        for label in EmotionLabel:
            matcher = Matcher(self.nlp.vocab)
            # Single word patterns
            patterns = [[{"LOWER": {"IN": EMOTION_PATTERNS[label]["keywords"]}}]]
            
            # Add contextual patterns
            if label == EmotionLabel.DISTRESS:
                patterns.extend([
                    [{"LOWER": "feeling"}, {"LOWER": {"IN": ["down", "bad", "awful", "terrible", "horrible"]}}],
                    [{"LOWER": {"IN": ["so", "very", "really"]}}, {"LOWER": {"IN": ["anxious", "depressed", "sad", "upset"]}}],
                    [{"LOWER": "not"}, {"LOWER": "coping"}, {"LOWER": "well"}]
                ])
            elif label == EmotionLabel.PROGRESS:
                patterns.extend([
                    [{"LOWER": {"IN": ["feel", "feeling", "getting"]}}, {"LOWER": "better"}],
                    [{"LOWER": "made"}, {"LOWER": "progress"}],
                    [{"LOWER": "managed"}, {"LOWER": "to"}],
                    [{"LOWER": {"IN": ["small", "little"]}}, {"LOWER": {"IN": ["step", "victory", "win", "achievement"]}}]
                ])
            
            matcher.add(label.value, patterns)
            self.pattern_matchers[label] = matcher
            print(f"Added {len(patterns)} keyword patterns for {label}")
    
    def _get_matches(self, doc: Doc) -> List[EmotionTag]:
        """Get all emotion matches in the document."""
        matches = []
        seen_spans = set()  # Track overlapping spans
        
        # Helper function to check if a span overlaps with any seen spans
        def is_overlapping(start: int, end: int) -> bool:
            for seen_start, seen_end in seen_spans:
                if (start >= seen_start and start < seen_end) or \
                   (end > seen_start and end <= seen_end) or \
                   (start <= seen_start and end >= seen_end):
                    return True
            return False
        
        print(f"\nAnalyzing text: {doc.text[:100]}...")
        
        # Check phrase matches first (they take priority)
        for label, matcher in self.phrase_matchers.items():
            for match_id, start, end in matcher(doc):
                span_text = doc[start:end].text
                char_start = doc[start].idx
                char_end = doc[end-1].idx + len(doc[end-1].text)
                
                print(f"Found phrase match: '{span_text}' ({label})")
                
                # Skip if overlapping with existing match
                if is_overlapping(char_start, char_end):
                    print(f"Skipping overlapping match: {span_text}")
                    continue
                
                # Check for negation
                if start > 0 and doc[start-1].lower_ in ["no", "not", "never"]:
                    print(f"Skipping negated match: {span_text}")
                    continue
                
                matches.append(EmotionTag(
                    label=label,
                    start=char_start,
                    end=char_end,
                    text=span_text
                ))
                seen_spans.add((char_start, char_end))
        
        # Check pattern matches
        for label, matcher in self.pattern_matchers.items():
            for match_id, start, end in matcher(doc):
                span_text = doc[start:end].text
                char_start = doc[start].idx
                char_end = doc[end-1].idx + len(doc[end-1].text)
                
                print(f"Found pattern match: '{span_text}' ({label})")
                
                # Skip if overlapping with existing match
                if is_overlapping(char_start, char_end):
                    print(f"Skipping overlapping match: {span_text}")
                    continue
                
                # Check for negation
                if start > 0 and doc[start-1].lower_ in ["no", "not", "never"]:
                    print(f"Skipping negated match: {span_text}")
                    continue
                
                # For single-word matches, try to expand context
                if end - start == 1:
                    # Look for intensifiers before
                    if start > 0 and doc[start-1].lower_ in ["very", "so", "really", "extremely"]:
                        char_start = doc[start-1].idx
                        span_text = doc[start-1:end].text
                        print(f"Expanded match with intensifier: '{span_text}'")
                    
                    # Look for relevant context after
                    if end < len(doc) and doc[end].lower_ in ["about", "that", "with", "by"]:
                        char_end = doc[end].idx + len(doc[end].text)
                        span_text = doc[start:end+1].text
                        print(f"Expanded match with context: '{span_text}'")
                
                matches.append(EmotionTag(
                    label=label,
                    start=char_start,
                    end=char_end,
                    text=span_text
                ))
                seen_spans.add((char_start, char_end))
        
        print(f"\nFound {len(matches)} total matches")
        return matches
    
    def analyze(self, text: str) -> AnalysisResult:
        doc = self.nlp(text)
        matches = self._get_matches(doc)
        return AnalysisResult(text=text, tags=matches)
    
    def batch_analyze(self, texts: List[str]) -> List[AnalysisResult]:
        return [self.analyze(text) for text in texts] 