"""
LLM Text Parser for Architectural Requirements
Uses open-source LLMs via HuggingFace Inference API (FREE)
"""

import json
import re
from typing import Dict, List, Optional
import requests
from rule_engine import RoomType


class LLMTextParser:
    """
    Parses natural language architectural requirements into structured layout specifications
    Uses HuggingFace Inference API (free tier)
    """
    
    # Free models available via HuggingFace Inference API
    RECOMMENDED_MODELS = {
        'qwen': 'Qwen/Qwen2.5-7B-Instruct',  # Best for structured output
        'mistral': 'mistralai/Mistral-7B-Instruct-v0.2',
        'phi': 'microsoft/Phi-3-mini-4k-instruct',
        'llama': 'meta-llama/Llama-3.2-3B-Instruct'  # May require access approval
    }
    
    def __init__(self, hf_token: Optional[str] = None, model: str = 'qwen'):
        """
        Initialize parser
        
        Args:
            hf_token: HuggingFace API token (optional for public models, get free at https://huggingface.co/settings/tokens)
            model: Model to use ('qwen', 'mistral', 'phi', 'llama')
        """
        self.hf_token = hf_token
        self.model_name = self.RECOMMENDED_MODELS.get(model, self.RECOMMENDED_MODELS['qwen'])
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.headers = {}
        if hf_token:
            self.headers = {"Authorization": f"Bearer {hf_token}"}
    
    def _call_llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call HuggingFace Inference API"""
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": max_tokens,
                "temperature": 0.3,  # Lower temperature for more consistent structured output
                "top_p": 0.9,
                "return_full_text": False
            }
        }
        
        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            
            if isinstance(result, list) and len(result) > 0:
                return result[0].get('generated_text', '')
            elif isinstance(result, dict):
                return result.get('generated_text', '')
            return str(result)
        except Exception as e:
            print(f"LLM API call failed: {e}")
            return ""
    
    def parse_architectural_requirements(self, text_description: str) -> Dict:
        """
        Parse natural language architectural requirements into structured layout specification
        
        Args:
            text_description: Natural language description (e.g., "2BHK apartment with attached bathrooms")
        
        Returns:
            Structured layout specification (JSON)
        """
        
        prompt = self._build_parsing_prompt(text_description)
        llm_response = self._call_llm(prompt)
        
        # Extract JSON from response
        layout_spec = self._extract_json(llm_response)
        
        # Post-process and validate
        layout_spec = self._post_process_layout(layout_spec, text_description)
        
        return layout_spec
    
    def _build_parsing_prompt(self, text_description: str) -> str:
        """Build structured prompt for LLM"""
        
        prompt = f"""You are an architectural assistant. Convert the following natural language description into a structured JSON layout specification.

User Request: "{text_description}"

Generate a JSON object with this EXACT structure:

{{
  "layout_type": "apartment|house|office|commercial",
  "num_bedrooms": <number>,
  "total_estimated_area": <square feet>,
  "rooms": [
    {{
      "type": "living_room|bedroom|master_bedroom|kitchen|bathroom|balcony|corridor|dining|study",
      "estimated_area": <square feet>,
      "requirements": ["window", "ventilation", "attached_bathroom", etc],
      "preferences": ["spacious", "well-lit", "open", etc]
    }}
  ],
  "adjacency_requirements": [
    {{"room1": "kitchen", "room2": "living_room", "connection": "open|door|corridor"}}
  ],
  "special_requirements": ["modern", "vastu-compliant", "wheelchair-accessible", etc],
  "design_style": "modern|traditional|minimalist|luxury"
}}

Key room types to use:
- living_room (main living area)
- master_bedroom (primary bedroom, usually with attached bathroom)
- bedroom (secondary bedrooms)
- kitchen
- bathroom (full bathroom)
- toilet (half bath/powder room)
- balcony
- dining (dining area)
- corridor (hallway/passage)
- entrance (entry foyer)
- utility (laundry/storage)
- study (home office)

Standard room sizes (reference):
- Living room: 200-300 sq ft
- Master bedroom: 150-200 sq ft
- Bedroom: 120-150 sq ft
- Kitchen: 80-120 sq ft
- Bathroom: 40-60 sq ft
- Balcony: 40-80 sq ft

Parse the requirements carefully. Extract:
1. Number and type of rooms
2. Special features (attached bathrooms, open kitchen, etc.)
3. Adjacency preferences
4. Size expectations

Output ONLY valid JSON, no other text.

JSON Output:"""
        
        return prompt
    
    def _extract_json(self, llm_response: str) -> Dict:
        """Extract JSON from LLM response"""
        
        # Try to find JSON block
        json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.load(json_str)
            except:
                pass
        
        # Fallback: try to parse entire response
        try:
            # Clean up response
            cleaned = llm_response.strip()
            # Remove markdown code blocks if present
            cleaned = re.sub(r'```json\s*', '', cleaned)
            cleaned = re.sub(r'```\s*', '', cleaned)
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"Failed to parse JSON: {e}")
            print(f"LLM Response: {llm_response}")
            return self._get_fallback_structure()
    
    def _get_fallback_structure(self) -> Dict:
        """Return minimal valid structure if parsing fails"""
        return {
            "layout_type": "apartment",
            "num_bedrooms": 2,
            "total_estimated_area": 1000,
            "rooms": [
                {"type": "living_room", "estimated_area": 200, "requirements": ["window"]},
                {"type": "bedroom", "estimated_area": 150, "requirements": ["window"]},
                {"type": "kitchen", "estimated_area": 80, "requirements": ["window", "ventilation"]},
                {"type": "bathroom", "estimated_area": 40, "requirements": ["ventilation"]}
            ],
            "adjacency_requirements": [],
            "special_requirements": [],
            "design_style": "modern"
        }
    
    def _post_process_layout(self, layout_spec: Dict, original_text: str) -> Dict:
        """Post-process and enhance layout specification"""
        
        # Add room dimensions based on estimated areas
        rooms = layout_spec.get('rooms', [])
        for room in rooms:
            area = room.get('estimated_area', 100)
            # Calculate reasonable dimensions (assuming rectangular rooms)
            # Use golden ratio (1.6:1) as default
            width = (area / 1.6) ** 0.5
            height = width * 1.6
            
            room['dimensions'] = {
                'width': round(width, 1),
                'height': round(height, 1)
            }
            
            # Add default structural elements
            room_type = room.get('type')
            if 'requirements' not in room:
                room['requirements'] = []
            
            # Add windows for rooms that need natural light
            if room_type in ['living_room', 'bedroom', 'master_bedroom', 'kitchen']:
                if 'window' not in room['requirements']:
                    room['requirements'].append('window')
            
            # Add ventilation for bathrooms and kitchens
            if room_type in ['bathroom', 'kitchen', 'toilet']:
                if 'ventilation' not in room['requirements']:
                    room['requirements'].append('ventilation')
        
        # Infer adjacencies from common patterns
        adjacencies = layout_spec.get('adjacency_requirements', [])
        
        # Kitchen usually connects to living/dining
        kitchen_exists = any(r.get('type') == 'kitchen' for r in rooms)
        living_exists = any(r.get('type') == 'living_room' for r in rooms)
        if kitchen_exists and living_exists:
            # Check if "open kitchen" mentioned
            if 'open' in original_text.lower() and 'kitchen' in original_text.lower():
                adjacencies.append({
                    'room1': 'kitchen',
                    'room2': 'living_room',
                    'connection': 'open'
                })
            else:
                adjacencies.append({
                    'room1': 'kitchen',
                    'room2': 'living_room',
                    'connection': 'door'
                })
        
        # Master bedroom with attached bathroom
        if 'attached' in original_text.lower() or 'ensuite' in original_text.lower():
            master_bedrooms = [r for r in rooms if r.get('type') == 'master_bedroom']
            bathrooms = [r for r in rooms if r.get('type') == 'bathroom']
            
            for i, mb in enumerate(master_bedrooms):
                if i < len(bathrooms):
                    adjacencies.append({
                        'room1': 'master_bedroom',
                        'room2': 'bathroom',
                        'connection': 'door',
                        'private': True
                    })
        
        layout_spec['adjacency_requirements'] = adjacencies
        
        # Add metadata
        layout_spec['parsed_from'] = original_text
        layout_spec['parser_version'] = '1.0'
        
        return layout_spec
    
    def convert_to_detailed_specification(self, layout_spec: Dict) -> Dict:
        """
        Convert high-level layout spec to detailed specification for image generation
        
        This creates the final JSON format that the image generation model will use
        """
        
        detailed_spec = {
            'layout_id': 'generated',
            'total_area': layout_spec.get('total_estimated_area', 1000),
            'layout_type': layout_spec.get('layout_type', 'apartment'),
            'rooms': [],
            'connections': [],
            'metadata': {
                'design_style': layout_spec.get('design_style', 'modern'),
                'special_requirements': layout_spec.get('special_requirements', [])
            }
        }
        
        # Convert rooms
        x_offset = 0
        for idx, room_spec in enumerate(layout_spec.get('rooms', [])):
            room = {
                'id': f"room_{idx}",
                'type': room_spec.get('type'),
                'area': room_spec.get('estimated_area', 100),
                'dimensions': room_spec.get('dimensions', {'width': 10, 'height': 10}),
                'position': {'x': x_offset, 'y': 0},  # Simple linear placement
                'doors': [],
                'windows': [],
                'requirements': room_spec.get('requirements', [])
            }
            
            # Add windows if required
            if 'window' in room['requirements']:
                room['windows'].append({
                    'position': 'exterior',
                    'width': 4,
                    'height': 5
                })
            
            # Add exhaust for ventilation
            if 'ventilation' in room['requirements']:
                room['has_exhaust'] = True
            
            detailed_spec['rooms'].append(room)
            x_offset += room_spec.get('dimensions', {}).get('width', 10)
        
        # Convert adjacencies to connections
        for adj in layout_spec.get('adjacency_requirements', []):
            connection = {
                'from': adj.get('room1'),
                'to': adj.get('room2'),
                'type': adj.get('connection', 'door')
            }
            detailed_spec['connections'].append(connection)
        
        return detailed_spec


# Example usage
if __name__ == "__main__":
    # Initialize parser (no token required for public models, but rate-limited)
    # Get free token at: https://huggingface.co/settings/tokens
    parser = LLMTextParser(hf_token=None, model='qwen')
    
    # Test examples
    test_descriptions = [
        "2BHK apartment with attached bathroom in master bedroom and open kitchen connected to living room",
        "3 bedroom house with separate dining area, modular kitchen, and balcony attached to living room",
        "Compact 1BHK with efficient layout, modern kitchen, and good ventilation",
        "Luxury 4BHK penthouse with study room, separate servant quarters, and multiple balconies"
    ]
    
    for desc in test_descriptions:
        print(f"\n{'='*80}")
        print(f"Input: {desc}")
        print(f"{'='*80}")
        
        # Parse requirements
        layout_spec = parser.parse_architectural_requirements(desc)
        print("\nParsed Layout Specification:")
        print(json.dumps(layout_spec, indent=2))
        
        # Convert to detailed specification
        detailed_spec = parser.convert_to_detailed_specification(layout_spec)
        print("\nDetailed Specification (for image generation):")
        print(json.dumps(detailed_spec, indent=2))
