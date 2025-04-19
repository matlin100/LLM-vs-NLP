from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import os
import logging
from dotenv import load_dotenv
from openai import RateLimitError

from ..models.base import EmotionLabel, EmotionTag, AnalysisResult
from ..models.llm.analyzer import LLMEmotionAnalyzer
from ..models.nlp.analyzer import NLPEmotionAnalyzer
from ..models.classifier.analyzer import CustomEmotionAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(
    title="Emotion Analysis API",
    description="API for analyzing emotions in patient notes using multiple approaches",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React development server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize analyzers
try:
    llm_analyzer = LLMEmotionAnalyzer(api_key=os.getenv("OPENAI_API_KEY"))
    nlp_analyzer = NLPEmotionAnalyzer()
    custom_analyzer = CustomEmotionAnalyzer(
        model_name=os.getenv("CUSTOM_MODEL_PATH", "allenai/longformer-base-4096")
    )
    logger.info("Successfully initialized all analyzers")
except Exception as e:
    logger.error(f"Error initializing analyzers: {str(e)}")
    raise

class EmotionTagModel(BaseModel):
    label: EmotionLabel
    start: int
    end: int
    text: str
    confidence: Optional[float] = None

class AnalysisResultModel(BaseModel):
    text: str
    tags: List[EmotionTagModel]

class AnalysisRequest(BaseModel):
    text: str = Field(..., description="The text to analyze")
    approach: str = Field(
        "llm",
        description="The analysis approach to use (llm, nlp, or custom)"
    )

@app.post(
    "/analyze",
    response_model=AnalysisResultModel,
    description="Analyze emotions in the provided text"
)
async def analyze_text(request: AnalysisRequest) -> AnalysisResultModel:
    """Analyze emotions in the provided text using the specified approach."""
    try:
        logger.info(f"Received analysis request for approach: {request.approach}")
        logger.debug(f"Text to analyze: {request.text[:100]}...")  # Log first 100 chars
        
        if request.approach == "llm":
            try:
                result = llm_analyzer.analyze(request.text)
            except RateLimitError as e:
                error_msg = "OpenAI API quota exceeded. Please check your API key and billing status."
                logger.error(error_msg)
                raise HTTPException(status_code=429, detail=error_msg)
        elif request.approach == "nlp":
            logger.info("Using NLP analyzer...")
            try:
                result = nlp_analyzer.analyze(request.text)
                logger.info(f"NLP analysis completed. Found {len(result.tags)} emotion tags.")
            except Exception as e:
                logger.error(f"NLP analysis failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"NLP analysis failed: {str(e)}")
        elif request.approach == "custom":
            logger.info("Using Custom Model analyzer...")
            try:
                result = custom_analyzer.analyze(request.text)
                logger.info(f"Custom Model analysis completed. Found {len(result.tags)} emotion tags.")
            except Exception as e:
                logger.error(f"Custom Model analysis failed: {str(e)}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Custom Model analysis failed: {str(e)}")
        else:
            error_msg = f"Invalid approach: {request.approach}"
            logger.error(error_msg)
            raise HTTPException(status_code=400, detail=error_msg)
        
        logger.info(f"Successfully analyzed text with {request.approach} approach")
        return AnalysisResultModel(
            text=result.text,
            tags=[EmotionTagModel(**tag.__dict__) for tag in result.tags]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error analyzing text: {str(e)}"
        logger.error(error_msg, exc_info=True)
        raise HTTPException(status_code=500, detail=error_msg)

class BatchAnalysisRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to analyze")
    approach: str = Field(
        "llm",
        description="The analysis approach to use (llm, nlp, or custom)"
    )

@app.post(
    "/analyze/batch",
    response_model=List[AnalysisResultModel],
    description="Analyze emotions in multiple texts"
)
async def analyze_texts(request: BatchAnalysisRequest) -> List[AnalysisResultModel]:
    """Analyze emotions in multiple texts using the specified approach."""
    try:
        if request.approach == "llm":
            results = llm_analyzer.batch_analyze(request.texts)
        elif request.approach == "nlp":
            results = nlp_analyzer.batch_analyze(request.texts)
        elif request.approach == "custom":
            results = custom_analyzer.batch_analyze(request.texts)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid approach: {request.approach}"
            )
        
        return [
            AnalysisResultModel(
                text=result.text,
                tags=[EmotionTagModel(**tag.__dict__) for tag in result.tags]
            )
            for result in results
        ]
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing texts: {str(e)}"
        )

@app.get(
    "/test/custom-model",
    response_model=List[AnalysisResultModel],
    description="Test the custom model with predefined texts"
)
async def test_custom_model() -> List[AnalysisResultModel]:
    """Test the custom model with a variety of emotional texts."""
    test_texts = [
        # Test for DANGER
        "I can't take this anymore, sometimes I think about disappearing completely.",
        
        # Test for DISTRESS
        "Everything feels overwhelming and I'm struggling to cope with daily tasks.",
        
        # Test for PROGRESS
        "I've been working hard on my recovery and starting to see small improvements.",
        
        # Test for INTENSE
        "I absolutely cannot handle how extremely difficult this situation has become.",
        
        # Test for mixed emotions
        "While I'm making progress with therapy, there are still moments when everything feels utterly hopeless.",
        
        # Test for subtle emotions
        "The days pass by, and I find myself staring out the window more often."
    ]
    
    results = []
    for text in test_texts:
        try:
            logger.info(f"Testing custom model with text: {text[:50]}...")
            result = custom_analyzer.analyze(text)
            results.append(AnalysisResultModel(
                text=text,
                tags=[EmotionTagModel(**tag.__dict__) for tag in result.tags]
            ))
        except Exception as e:
            logger.error(f"Error analyzing test text: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Custom model test failed: {str(e)}"
            )
    
    return results

if __name__ == "__main__":
    import uvicorn
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000, help="Port to run the server on")
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port) 