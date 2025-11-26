# coding=utf-8
"""crop_health_chatbot_agentV6.py
CropHealthModelChatbotAgent using CropHealthModelAgentV6 (GRU-based) and LLM

Author: Samantha Lee
STATUS: FIXED - Proper date handling and clearer responses
"""

import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, List
from datetime import datetime, timedelta

from dotenv import load_dotenv
from groq import Groq

try:
    from .crop_health_model_agentV6 import CropHealthModelAgentV6
except ImportError as e:
    print(f"ERROR: Could not import CropHealthModelAgentV6: {e}")
    sys.exit(1)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)
load_dotenv() 


class CropHealthChatbotAgentV6:
    name = "CropHealthChatbotAgentV6"
    MODEL_NAME = "llama-3.3-70b-versatile" 

    def __init__(self, date: str = "2025-10-22", field_id: str = None):
        self.date = date
        self.field_id = field_id
        
        # Initialize predictor
        self.predictor = CropHealthModelAgentV6(date=self.date, field_id=self.field_id)
        
        # Get dynamic location data
        self.all_locations_list: List[str] = self.predictor.all_locations
        self.all_locations_str: str = ", ".join(self.all_locations_list)
        
        self.location_name = "Unknown"
        self.crop_type = "Unknown"
        self._update_location_details()

        self.groq_api_key = self._get_groq_api_key()
        self.client = Groq(api_key=self.groq_api_key) if self.groq_api_key else None

        self.nlu_prompt = self._get_nlu_prompt()
        self.conversation_history = []
        self.forecast_cache = {} 
        self.is_awaiting_location = False
        self.is_awaiting_crop_for_location: Optional[str] = None

    def _update_location_details(self):
        """Helper to parse field_id from the predictor's current state."""
        if self.predictor.field_id and '|' in self.predictor.field_id:
            try:
                parts = self.predictor.field_id.split('|')
                self.location_name = parts[1].replace('-', ' ').title() 
                self.crop_type = parts[2].capitalize() 
            except: 
                pass

    def _build_system_prompt(self):
        """Constructs the system prompt based on current predictor state."""
        selected_data = self.predictor.selected_row_data
        source_note = selected_data.get("source", "Historical Data")
        
        # Calculate expected forecast period
        try:
            anchor = datetime.strptime(self.date, "%Y-%m-%d")
            day1 = (anchor + timedelta(days=1)).strftime("%Y-%m-%d")
            day7 = (anchor + timedelta(days=7)).strftime("%Y-%m-%d")
            forecast_period = f"{day1} through {day7}"
        except:
            forecast_period = "Next 7 days"

        return f"""You are Agribot, a precision agriculture AI assistant.

CURRENT CONTEXT:
- Anchor Date: {self.date}
- Forecast Period: {forecast_period}
- Location: {self.location_name}
- Crop: {self.crop_type}
- Data Source: {source_note}

RESPONSE RULES:
1. Be concise and technical
2. No emojis or casual language
3. Use the forecast data provided in Context
4. Format tables in clean markdown

FORECAST FORMAT:
**Forecast Summary**
[One sentence describing the overall trend]

**Field Information**
- Field ID: {self.predictor.field_id or 'N/A'}
- Location: {self.location_name}
- Crop: {self.crop_type}
- Analysis Period: {forecast_period}

**7-Day NDVI Forecast**
| Date | NDVI | Status |
|:-----|:-----|:-------|
[Use the forecast data from Context - dates should be FUTURE dates starting from day+1]

**Key Insights**
1. [Technical observation about NDVI trend]
2. [Health classification for majority of period]
3. [Potential agronomic factor]

**Recommendations**
- [Specific action item 1]
- [Specific action item 2]
"""

    def _get_nlu_prompt(self):
        """Builds the NLU prompt with dynamic location examples."""
        return f"""
Intent classification for agricultural chatbot.
INTENTS: 'forecast', 'status', 'greeting', 'goodbye', 'general_knowledge'
LOCATIONS: [{self.all_locations_str}]
CROPS: ['sunflower', 'rapeseed', 'maize', 'winter_wheat']

Return JSON with: {{"intent": "...", "location": "...", "crop": "..."}}
"""

    @staticmethod
    def _get_groq_api_key():
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            print("WARNING: GROQ_API_KEY not found")
        return key

    def _detect_intent_and_location(self, user_input):
        """Use LLM for intent detection with fallback."""
        if not self.client:
            return self._simple_fallback_nlu(user_input)
            
        try:
            completion = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.nlu_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0,
                response_format={"type": "json_object"},
            )
            response_text = completion.choices[0].message.content
            data = json.loads(response_text)
            return data.get("intent", "unknown"), data.get("location"), data.get("crop")
        except Exception as e:
            print(f"NLU error: {e}")
            return self._simple_fallback_nlu(user_input)

    def _simple_fallback_nlu(self, user_input: str):
        """Keyword-based fallback NLU."""
        user_lower = user_input.lower()
        location = None
        crop = None
        intent = 'unknown'

        if 'forecast' in user_lower or 'predict' in user_lower or '7-day' in user_lower or '7 day' in user_lower:
            intent = 'forecast'
            for loc in self.all_locations_list:
                if loc.lower() in user_lower:
                    location = loc
                    break
            
            if 'sunflower' in user_lower: crop = 'sunflower'
            elif 'rapeseed' in user_lower or 'rapseed' in user_lower: crop = 'rapeseed'
            elif 'maize' in user_lower or 'corn' in user_lower: crop = 'maize'
            elif 'wheat' in user_lower: crop = 'winter_wheat'
            
        elif 'status' in user_lower or 'current' in user_lower:
            intent = 'status'
        elif 'season' in user_lower or 'timeline' in user_lower or 'explain' in user_lower:
            intent = 'general_knowledge'
        elif any(word in user_lower for word in ['hello', 'hi', 'hey']):
            intent = 'greeting'
        elif any(word in user_lower for word in ['bye', 'goodbye', 'exit']):
            intent = 'goodbye'
        
        return intent, location, crop

    def _get_llm_response(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """Get LLM response with context."""
        if not self.client:
            return "Error: API connection unavailable. Check GROQ_API_KEY."

        try:
            system_prompt = self._build_system_prompt()
            
            messages = [
                {"role": "system", "content": system_prompt},
                *self.conversation_history[-6:],  # Keep last 3 exchanges
                {"role": "user", "content": user_input + (f"\n\nContext:\n```json\n{json.dumps(context, indent=2)}\n```" if context else "")}
            ]

            response = self.client.chat.completions.create(
                model=self.MODEL_NAME,
                messages=messages,
                temperature=0.3,
                max_tokens=700,
            )
            assistant_response = response.choices[0].message.content

            # Update history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            
            return assistant_response

        except Exception as e:
            return f"LLM Error: {str(e)}"

    def _run_forecast(self, location_query: str, crop_query: Optional[str] = None) -> str:
        """Execute forecast and return formatted response."""
        
        # Set field context
        self.predictor.set_field_and_date(location_query, self.date, crop_query)
        self._update_location_details()

        # Check for errors
        if "error" in self.predictor.selected_row_data:
            error_msg = self.predictor.selected_row_data['error']
            return f"**Field Lookup Failed**\n\n{error_msg}\n\nAvailable locations: {self.all_locations_str}"
        
        # Generate forecast
        forecast_data = self.predictor.forecast_next_week()
        
        if 'error' in forecast_data:
            return f"**Forecast Generation Failed**\n\n{forecast_data['error']}"
        
        # Cache for later use
        self.forecast_cache[self.location_name] = forecast_data
        
        # Generate interpretation
        interpretation = self._get_llm_response(
            f"Analyze the 7-day NDVI forecast for {self.location_name} {self.crop_type}.", 
            forecast_data
        )
        
        return interpretation

    def handle_input(self, user_input: str) -> str:
        """
        Main entry point for processing user input.
        Returns the response string.
        """
        
        # Handle follow-up prompts
        if self.is_awaiting_location:
            self.is_awaiting_location = False
            return self._run_forecast(user_input, crop_query=None)
        
        if self.is_awaiting_crop_for_location:
            location_query = self.is_awaiting_crop_for_location
            self.is_awaiting_crop_for_location = None
            return self._run_forecast(location_query, user_input)

        # Detect intent
        intent, location_query, crop_query = self._detect_intent_and_location(user_input)

        # FORECAST REQUEST
        if intent == "forecast":
            # Check if location was detected
            if not location_query:
                self.is_awaiting_location = True
                return f"**Location Required**\n\nPlease specify a region from: {self.all_locations_str}"
            
            # Check for ambiguous crop selection
            if not crop_query and location_query:
                loc_str = self.predictor.location_map.get(location_query)
                if loc_str:
                    available_crops = []
                    
                    # Check both data sources
                    if self.predictor.df_historical is not None:
                        pattern = rf"\|{loc_str}\|"
                        fids_main = self.predictor.df_historical[
                            self.predictor.df_historical['field_id'].str.contains(pattern, regex=True, case=False, na=False)
                        ]['field_id'].unique()
                        available_crops.extend([f.split('|')[2].lower() for f in fids_main if len(f.split('|')) > 2])

                    if self.predictor.df_evaluation is not None:
                        pattern = rf"\|{loc_str}\|"
                        fids_eval = self.predictor.df_evaluation[
                            self.predictor.df_evaluation['field_id'].str.contains(pattern, regex=True, case=False, na=False)
                        ]['field_id'].unique()
                        available_crops.extend([f.split('|')[2].lower() for f in fids_eval if len(f.split('|')) > 2])

                    available_crops = sorted(list(set(available_crops)))
                    
                    if len(available_crops) > 1:
                        self.is_awaiting_crop_for_location = location_query
                        return f"**Crop Specification Required**\n\nMultiple crops found for {location_query}. Please specify: {', '.join(available_crops)}"
                    elif len(available_crops) == 1:
                        crop_query = available_crops[0]
            
            return self._run_forecast(location_query, crop_query)
        
        # STATUS REQUEST
        elif intent == "status":
            if location_query:
                self.predictor.set_field_and_date(location_query, None, crop_query)
                self._update_location_details()
                 
            selected_data = self.predictor.selected_row_data
            if 'error' in selected_data:
                return f"**Status Lookup Failed**\n\n{selected_data['error']}"
            
            return f"""**Current Field Status**

**Field Configuration**
- Location: {self.location_name}
- Crop: {self.crop_type}
- Field ID: {selected_data.get('field_id', 'Unknown')}
- Date: {selected_data.get('date', 'Unknown')}

**Health Metrics**
- NDVI: {selected_data.get('ndvi_mean', 'Unknown')}
- Status: {selected_data.get('health_status', 'Unknown')}

**Environmental Conditions**
- Temperature: {selected_data.get('temperature_mean', 'Unknown')}°C
- Precipitation: {selected_data.get('precipitation_sum', 'Unknown')} mm
- Soil Moisture: {selected_data.get('soil_moisture', 'Unknown')}
"""
        
        # GENERAL KNOWLEDGE
        elif intent == "general_knowledge":
            return self._get_llm_response(user_input, context=self.forecast_cache)

        # GREETING
        elif intent == "greeting":
            return f"**Agribot Online**\n\nReady to analyze crop health data. Current anchor date: {self.date}\n\nAvailable regions: {self.all_locations_str}"

        # GOODBYE
        elif intent == "goodbye":
            return "**Session Terminated**\n\nThank you for using Agribot."

        # UNKNOWN
        else: 
            return f"**Intent Not Recognized**\n\nPlease try:\n- 'Forecast for [location] [crop]'\n- 'Status of [location]'\n- General agricultural questions\n\nAvailable locations: {self.all_locations_str}"