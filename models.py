from typing import List, Optional
from enum import Enum
from pydantic import BaseModel


class StageType(str, Enum):
    START = "START"
    NORMAL = "NORMAL"
    END = "END"
    GLOBAL = "GLOBAL"


class NextStage(BaseModel):
    nextStageId: str
    condition: Optional[str] = ""


class Stage(BaseModel):
    id: str
    name: str
    type: StageType
    prompt: Optional[str] = ""
    nextStages: Optional[List[NextStage]] = None
    final_prompt: Optional[str] = None
    generic_prompt: Optional[str] = None
    inCondition: Optional[str] = None


class LLMResponse(BaseModel):
    """Structured output model for LLM responses"""
    response: str
    next_stage: str
    confidence: Optional[float] = 1.0
