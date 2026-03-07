"""
Layout Specification Parser
Converts text descriptions or trained model outputs to geometric layout specifications
"""

import json
import re
from typing import Dict, List, Tuple, Optional
from enum import Enum


class LayoutParser:
    """
    Parses natural language or extracts layout info to create geometric specifications
    """
    
    # Standard room dimensions (width, height) in feet
    ROOM_DIMENSIONS = {
        'living_room': (15, 12),
        'master_bedroom': (14, 12),
        'bedroom': (12, 10),
        'kitchen': (10, 8),
        'dining': (12, 10),
        'bathroom': (7, 5),
        'toilet': (5, 4),
        'balcony': (8, 4),
        'study': (10, 8),
        'utility': (6, 4),
        'corridor': (3, 8),
        'entrance': (6, 4),
    }
    
    # Room adjacency rules
    ADJACENCY_RULES = {
        'kitchen': ['dining', 'living_room'],
        'master_bedroom': ['bathroom'],
        'dining': ['living_room', 'kitchen'],
        'bathroom': ['master_bedroom', 'bedroom'],
    }
    
    def parse_text_description(self, description: str) -> Dict:
        """
        Parse natural language description into layout specification
        
        Example input: "2BHK apartment with attached bathroom and open kitchen"
        
        Returns: Geometric layout specification dict
        """
        
        description_lower = description.lower()
        
        # Extract number of bedrooms
        num_bedrooms = self._extract_bedroom_count(description_lower)
        
        # Detect room types mentioned
        rooms_mentioned = self._extract_room_types(description_lower)
        
        # Build room list
        room_specs = []
        
        # Add living room (standard in most layouts)
        if 'living' not in description_lower or num_bedrooms > 0:
            room_specs.append({
                'type': 'living_room',
                'label': 'Living Room'
            })
        
        # Add bedrooms
        if num_bedrooms >= 1:
            room_specs.append({
                'type': 'master_bedroom',
                'label': 'Master Bedroom'
            })
            
            for i in range(1, num_bedrooms):
                room_specs.append({
                    'type': 'bedroom',
                    'label': f'Bedroom {i+1}'
                })
        
        # Add kitchen
        if 'kitchen' in rooms_mentioned or num_bedrooms > 0:
            room_specs.append({
                'type': 'kitchen',
                'label': 'Kitchen'
            })
        
        # Add bathrooms
        num_bathrooms = self._extract_bathroom_count(description_lower)
        for i in range(num_bathrooms):
            if i == 0 and 'attached' in description_lower:
                room_specs.append({
                    'type': 'bathroom',
                    'label': 'Master Bathroom',
                    'attached_to': 'master_bedroom'
                })
            else:
                room_specs.append({
                    'type': 'bathroom',
                    'label': f'Bathroom {i+1}'
                })
        
        # Add dining if mentioned
        if 'dining' in rooms_mentioned:
            room_specs.append({
                'type': 'dining',
                'label': 'Dining Area'
            })
        
        # Add balcony if mentioned
        if 'balcony' in rooms_mentioned:
            room_specs.append({
                'type': 'balcony',
                'label': 'Balcony'
            })
        
        # Add study if mentioned
        if 'study' in rooms_mentioned:
            room_specs.append({
                'type': 'study',
                'label': 'Study Room'
            })
        
        # Create geometric layout from room specs
        geometric_spec = self._create_geometric_layout(room_specs, description_lower)
        
        return geometric_spec
    
    def _extract_bedroom_count(self, text: str) -> int:
        """Extract number of bedrooms from text"""
        
        # Look for patterns like "2BHK", "3 bedroom", "2 BHK"
        patterns = [
            r'(\d+)\s*bhk',
            r'(\d+)\s*bedroom',
            r'(\d+)\s*br',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return int(match.group(1))
        
        # Check for written numbers
        number_words = {
            'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
            'single': 1, 'double': 2, 'triple': 3
        }
        
        for word, num in number_words.items():
            if f'{word} bedroom' in text or f'{word}bhk' in text:
                return num
        
        return 2  # Default
    
    def _extract_bathroom_count(self, text: str) -> int:
        """Extract number of bathrooms"""
        
        # Look for explicit bathroom count
        match = re.search(r'(\d+)\s*bathroom', text)
        if match:
            return int(match.group(1))
        
        # Default based on bedroom count
        if '1bhk' in text or 'studio' in text:
            return 1
        elif '2bhk' in text:
            return 1 if 'attached' not in text else 2
        elif '3bhk' in text:
            return 2
        elif '4bhk' in text:
            return 3
        
        return 1
    
    def _extract_room_types(self, text: str) -> List[str]:
        """Extract mentioned room types"""
        
        rooms = []
        for room_type in self.ROOM_DIMENSIONS.keys():
            room_name = room_type.replace('_', ' ')
            if room_name in text or room_type in text:
                rooms.append(room_type)
        
        return rooms
    
    def _create_geometric_layout(self, room_specs: List[Dict], description: str) -> Dict:
        """
        Create geometric layout with precise positions and dimensions
        """
        
        # Determine layout strategy
        is_open_kitchen = 'open kitchen' in description or 'open plan' in description
        is_compact = 'compact' in description or 'studio' in description or '1bhk' in description
        
        # Calculate positions using optimal algorithm
        positioned_rooms = self._calculate_room_positions(room_specs, is_compact, is_open_kitchen)
        
        return {
            'layout_type': 'apartment',
            'rooms': positioned_rooms,
            'metadata': {
                'description': description,
                'is_open_kitchen': is_open_kitchen,
                'is_compact': is_compact
            }
        }
    
    def _calculate_room_positions(
        self, 
        room_specs: List[Dict], 
        is_compact: bool,
        is_open_kitchen: bool
    ) -> List[Dict]:
        """
        Calculate optimal room positions using grid-based layout algorithm
        """
        
        positioned_rooms = []
        current_x = 0
        current_y = 0
        max_height_in_row = 0
        max_row_width = 30 if not is_compact else 20  # feet
        
        # Group rooms for better layout
        living_dining_kitchen = []
        bedrooms = []
        bathrooms = []
        others = []
        
        for spec in room_specs:
            room_type = spec['type']
            if room_type in ['living_room', 'dining', 'kitchen']:
                living_dining_kitchen.append(spec)
            elif 'bedroom' in room_type:
                bedrooms.append(spec)
            elif 'bathroom' in room_type:
                bathrooms.append(spec)
            else:
                others.append(spec)
        
        # Layout order: living area, bedrooms, bathrooms, others
        ordered_rooms = living_dining_kitchen + bedrooms + bathrooms + others
        
        for idx, spec in enumerate(ordered_rooms):
            room_type = spec['type']
            label = spec['label']
            
            # Get standard dimensions
            width, height = self.ROOM_DIMENSIONS.get(room_type, (10, 10))
            
            # Adjust for compact layouts
            if is_compact:
                width *= 0.85
                height *= 0.85
            
            # Check if we need new row
            if current_x + width > max_row_width and idx > 0:
                current_x = 0
                current_y += max_height_in_row + 0.5  # Wall thickness
                max_height_in_row = 0
            
            # Determine door and window positions
            doors, windows = self._calculate_openings(
                room_type, current_x, current_y, width, height, spec
            )
            
            positioned_room = {
                'type': room_type,
                'label': label,
                'width': round(width, 1),
                'height': round(height, 1),
                'position': {'x': round(current_x, 1), 'y': round(current_y, 1)},
                'doors': doors,
                'windows': windows
            }
            
            positioned_rooms.append(positioned_room)
            
            # Update position
            current_x += width + 0.5
            max_height_in_row = max(max_height_in_row, height)
        
        return positioned_rooms
    
    def _calculate_openings(
        self,
        room_type: str,
        x: float,
        y: float,
        width: float,
        height: float,
        spec: Dict
    ) -> Tuple[List[Dict], List[Dict]]:
        """Calculate door and window positions"""
        
        doors = []
        windows = []
        
        # Default door position (left wall, middle)
        doors.append({
            'x': round(x, 1),
            'y': round(y + height / 2, 1),
            'width': 3.0,
            'swing_angle': 90,
            'wall_side': 'left'
        })
        
        # Default window positions based on room type
        if room_type in ['living_room', 'bedroom', 'master_bedroom']:
            # Window on top wall
            windows.append({
                'x': round(x + width / 2, 1),
                'y': round(y + height, 1),
                'width': 4.0,
                'height': 4.0,
                'wall_side': 'top'
            })
        
        if room_type == 'kitchen':
            # Kitchen window on right wall
            windows.append({
                'x': round(x + width, 1),
                'y': round(y + height / 2, 1),
                'width': 3.0,
                'height': 3.0,
                'wall_side': 'right'
            })
        
        if room_type in ['bathroom', 'toilet']:
            # Small window for ventilation
            windows.append({
                'x': round(x + width / 2, 1),
                'y': round(y + height, 1),
                'width': 2.0,
                'height': 2.0,
                'wall_side': 'top'
            })
        
        return doors, windows


def create_layout_from_description(description: str, output_path: str = None) -> Dict:
    """
    High-level function to create geometric layout from text description
    
    Args:
        description: Natural language description
        output_path: Optional path to save specification JSON
    
    Returns:
        Geometric layout specification
    """
    
    parser = LayoutParser()
    spec = parser.parse_text_description(description)
    
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(spec, f, indent=2)
        print(f"✓ Layout specification saved: {output_path}")
    
    return spec


# Example usage
if __name__ == "__main__":
    
    # Test examples
    test_descriptions = [
        "2BHK apartment with attached bathroom and open kitchen",
        "3 bedroom house with separate dining area",
        "Compact 1BHK with efficient layout",
        "Luxury 4BHK penthouse with study room",
        "Studio apartment with balcony"
    ]
    
    parser = LayoutParser()
    
    for desc in test_descriptions:
        print(f"\n{'='*80}")
        print(f"Description: {desc}")
        print(f"{'='*80}")
        
        spec = parser.parse_text_description(desc)
        
        print(f"\nGenerated Layout:")
        print(f"  Rooms: {len(spec['rooms'])}")
        for room in spec['rooms']:
            print(f"  - {room['label']}: {room['width']}' x {room['height']}' at ({room['position']['x']}, {room['position']['y']})")
        
        # Save specification
        filename = f"layout_{desc.replace(' ', '_')[:30]}.json"
        with open(filename, 'w') as f:
            json.dump(spec, f, indent=2)
        print(f"\n✓ Saved: {filename}")
