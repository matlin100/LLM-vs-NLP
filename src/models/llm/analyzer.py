from typing import List, Dict, Any
import json
from openai import OpenAI
from ..base import EmotionAnalyzer, AnalysisResult, EmotionTag, EmotionLabel

SYSTEM_PROMPT = """You are an expert in analyzing emotional content in patient notes.
Your task is to identify and tag spans of text that express:
1. Danger (self-harm, suicide ideation, harm to others)
2. Emotional distress (anxiety, depression, suffering)
3. Emotional progress (improvement, hope, positive changes)
4. Emotionally intense expressions (strong feelings, emphasis)

For each identified span, provide:
- The emotion label
- The start and end character positions
- The exact text that was tagged

Be precise with character positions and maintain the exact original text."""

USER_PROMPT_TEMPLATE = """Analyze the following patient note for emotional content.
Return the analysis in JSON format with the following structure:
{{
    "text": "original text",
    "tags": [
        {{
            "label": "danger|emotional_distress|emotional_progress|emotionally_intense",
            "start": <int>,
            "end": <int>,
            "text": "exact span text"
        }}
    ]
}}

Patient note:
{text}"""

class LLMEmotionAnalyzer(EmotionAnalyzer):
    def __init__(self, api_key: str, model: str = "gpt-4-turbo-preview"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            return {"text": "", "tags": []}
    
    def _convert_to_analysis_result(self, response: Dict[str, Any]) -> AnalysisResult:
        tags = []
        for tag in response.get("tags", []):
            try:
                tags.append(EmotionTag(
                    label=EmotionLabel(tag["label"]),
                    start=tag["start"],
                    end=tag["end"],
                    text=tag["text"]
                ))
            except (KeyError, ValueError):
                continue
        
        return AnalysisResult(
            text=response.get("text", ""),
            tags=tags
        )
    
    def analyze(self, text: str) -> AnalysisResult:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(text=text)}
        ]
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        
        result = self._parse_response(response.choices[0].message.content)
        return self._convert_to_analysis_result(result)
    
    def batch_analyze(self, texts: List[str]) -> List[AnalysisResult]:
        # Process each text individually for now
        # Could be optimized with async or batch API calls if available
        return [self.analyze(text) for text in texts] 