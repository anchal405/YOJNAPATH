import json
from typing import Any, Dict, List, Optional, Union

from models import NextStage, Stage, StageType
from prompt_manager import BedrockNovaPromptManager

# Remove the problematic imports that don't exist
# from langchain_core.prompts import ChatPromptTemplate
# from loguru import logger
# from apps.backend_app.utils.json_utils import dict_to_markdown

import logging
logger = logging.getLogger(__name__)

def dict_to_markdown(data: dict, root_element_name: str = "data") -> str:
    """Simple dictionary to markdown converter"""
    if not data:
        return ""
    
    markdown = f"## {root_element_name}\n"
    for key, value in data.items():
        markdown += f"- **{key}**: {value}\n"
    return markdown

class Boto3StageManager():
    """Boto3 implementation of the StageManager."""
    
    def __init__(
        self,
        generic_prompt: str,
        stages_info: str,
        input_variables: Optional[dict] = None,
    ):
        
        self.flow_variables = dict_to_markdown(input_variables or {}, root_element_name="input_variables")
        self.input_variables = input_variables or {}  # Store raw input variables for substitution
        self.stage_id_2_stage: Dict[str, Stage] = {}
        self.conversation_id_2_active_stage: Dict[str, str] = {}
        self.generic_prompt = generic_prompt
        self.prompt_manager = BedrockNovaPromptManager()
        
        # Set stage_id_2_stage reference in prompt manager
        self.prompt_manager.stage_id_2_stage = self.stage_id_2_stage
        # Set stage manager reference for variable substitution
        self.prompt_manager._stage_manager = self
        
        self.load_stages(stages_info)
        logger.info(f"Loaded {len(self.stage_id_2_stage)} stages.")
    
    def check_if_stage_exists(self, stage_id: str) -> bool:
        return stage_id in self.stage_id_2_stage
    
    def load_stages(self, stages_info: str) -> None:
        stages_data = json.loads(stages_info)
        
        global_stages = [Stage(**stage_data) for stage_data in stages_data if stage_data.get("type") == StageType.GLOBAL]
        logger.debug(f"Global stages: {global_stages}")
        
        # Load each stage
        for stage_data in stages_data:
            stage = Stage(**stage_data)
            # Ensure nextStages is always a list
            if stage.nextStages is None:
                stage.nextStages = []
            
            # Add generic prompt to the stage if not already present
            stage.generic_prompt = self.generic_prompt
            self.stage_id_2_stage[stage.id] = stage
        
        # Update prompts based on next stage details and conditions
        for stage in self.stage_id_2_stage.values():
            # Add global stages to each stage's nextStages
            for global_stage in global_stages:
                if global_stage.id != stage.id:
                    if stage.nextStages is None:
                        stage.nextStages = []
                    stage.nextStages.append(NextStage(nextStageId=global_stage.id, condition=global_stage.inCondition or ""))
            
            if stage.nextStages and len(stage.nextStages) > 0:
                stage.final_prompt = self.formulate_prompt_for_stage(stage)
            else:
                stage.final_prompt = stage.prompt
    
    def substitute_variables_in_text(self, text: str) -> str:
        """Substitute input variables in text with actual values."""
        if not text or not self.input_variables:
            return text
        
        try:
            # Use string formatting to replace placeholders
            return text.format(**self.input_variables)
        except KeyError as e:
            logger.warning(f"Missing variable in stage prompt: {e}")
            return text
        except Exception as e:
            logger.error(f"Error substituting variables: {e}")
            return text
    
    def formulate_prompt_for_stage(self, stage: Stage) -> str:
        """Formulate the prompt for a stage using the prompt manager."""
        # First substitute variables in the stage prompt
        if stage.prompt:
            stage.prompt = self.substitute_variables_in_text(stage.prompt)
        
        # Format next stage prompts (they also need variable substitution)
        next_stage_prompts = self.prompt_manager.format_next_stage_prompt(stage)
        
        # Create the final formatted prompt
        stage.final_prompt = self.prompt_manager.format_stage_prompt(stage, next_stage_prompts)
        return stage.final_prompt
    
    def get_active_stage(self, conversation_id: str) -> Optional[Stage]:
        stage_id = self.conversation_id_2_active_stage.get(conversation_id)
        active_stage = None
        if not stage_id:
            active_stage = self.get_start_stage()
            self.set_active_stage(conversation_id, active_stage.id)
            stage_id = active_stage.id
            logger.info(f"Set initial active stage for conversation ID: {conversation_id} to stage ID: {active_stage.id} and stage name is {active_stage.name}")
        
        return self.stage_id_2_stage.get(stage_id) if stage_id else None
    
    def find_stage_by_name(self, stage_name: str) -> Optional[Stage]:
        """Find a stage by its name or first 6 characters of its ID."""
        # First try to find by exact name match
        for stage in self.stage_id_2_stage.values():
            if stage.name == stage_name:
                return stage
        
        # If not found by name, try to find by first 6 characters of ID
        for stage in self.stage_id_2_stage.values():
            if stage.id.startswith(stage_name):
                return stage
        
        return None

    def set_active_stage(self, conversation_id: str, stage_id: str) -> None:
        """Sets the active stage for a conversation.
        
        Args:
            conversation_id: The ID of the conversation
            stage_id: The stage ID or stage name to set as active
        """
        # First check if it's a direct stage ID match
        if self.check_if_stage_exists(stage_id):
            stage = self.stage_id_2_stage.get(stage_id)
            if stage:
                logger.debug(f"Setting active stage for conversation ID: {conversation_id} to stage ID: {stage_id} with stage name: {stage.name}")
                self.conversation_id_2_active_stage[conversation_id] = stage_id
                return
            
        # If not found by direct ID, try to find by name
        stage = self.find_stage_by_name(stage_id)
        if stage:
            logger.debug(f"Setting active stage for conversation ID: {conversation_id} to stage ID: {stage.id} with stage name: {stage.name}")
            self.conversation_id_2_active_stage[conversation_id] = stage.id
        else:
            logger.warning(f"Stage not found for identifier: {stage_id}")
    
    def get_start_stage(self) -> Stage:
        start_stage = next((stage for stage in self.stage_id_2_stage.values() if stage.type == StageType.START), None)
        if not start_stage:
            raise Exception("Start stage not found")
        return start_stage
    
    def get_end_stage(self) -> Optional[Stage]:
        end_stage = next((stage for stage in self.stage_id_2_stage.values() if stage.type == StageType.END), None)
        if not end_stage:
            raise Exception("End stage not found")
        return end_stage
    
    def get_chain_for_current_active_stage(self, conversation_id: str, use_function_chain: bool = True) -> Any:
        """Get the appropriate prompt for the current active stage."""
        active_stage = self.get_active_stage(conversation_id)
        if not active_stage:
            active_stage = self.get_start_stage()
            self.set_active_stage(conversation_id, active_stage.id)
            logger.info(f"Set initial active stage for conversation ID: {conversation_id} to stage ID: {active_stage.id} and stage name is {active_stage.name}")
        else:
            logger.info(f"Retrieved active stage for conversation ID: {conversation_id} is stage ID: {active_stage.id} and stage name is {active_stage.name}")
        
        logger.info(f"Using standard prompt for stage ID: {active_stage.id} and stage name: {active_stage.name}")
        return active_stage.final_prompt
    
    def get_active_stage_message(self, conversation_id: str) -> Optional[str]:
        active_stage = self.get_active_stage(conversation_id)
        return active_stage.prompt if active_stage else None
    
    def get_stage_prompt_by_name(self, stage_name: str) -> Optional[str]:
        """Get the formatted prompt for a stage by name."""
        stage = self.find_stage_by_name(stage_name)
        if stage:
            return stage.final_prompt
        return None
    
    def get_stage_prompt_by_id(self, stage_id: str) -> Optional[str]:
        """Get the formatted prompt for a stage by ID."""
        stage = self.stage_id_2_stage.get(stage_id)
        if stage:
            return stage.final_prompt
        return None