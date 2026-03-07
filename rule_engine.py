"""
Engineering Rule Engine for Architectural Layout Validation
Enforces building code constraints and architectural standards
"""

import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import math


class RoomType(Enum):
    LIVING_ROOM = "living_room"
    MASTER_BEDROOM = "master_bedroom"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    TOILET = "toilet"
    BALCONY = "balcony"
    DINING = "dining"
    STUDY = "study"
    CORRIDOR = "corridor"
    ENTRANCE = "entrance"
    UTILITY = "utility"
    STORE = "store"


@dataclass
class RoomSpec:
    """Room specification with size constraints"""
    type: RoomType
    min_area: float  # sq ft
    ideal_area_range: Tuple[float, float]  # (min, max) sq ft
    min_dimensions: Tuple[float, float]  # (width, height) in feet
    requires_window: bool
    requires_ventilation: bool
    can_be_interior: bool  # Can be without external wall


# Room Standards Database
ROOM_STANDARDS = {
    RoomType.LIVING_ROOM: RoomSpec(
        type=RoomType.LIVING_ROOM,
        min_area=150,
        ideal_area_range=(200, 400),
        min_dimensions=(10, 12),
        requires_window=True,
        requires_ventilation=True,
        can_be_interior=False
    ),
    RoomType.MASTER_BEDROOM: RoomSpec(
        type=RoomType.MASTER_BEDROOM,
        min_area=120,
        ideal_area_range=(150, 250),
        min_dimensions=(10, 12),
        requires_window=True,
        requires_ventilation=True,
        can_be_interior=False
    ),
    RoomType.BEDROOM: RoomSpec(
        type=RoomType.BEDROOM,
        min_area=100,
        ideal_area_range=(120, 180),
        min_dimensions=(9, 11),
        requires_window=True,
        requires_ventilation=True,
        can_be_interior=False
    ),
    RoomType.KITCHEN: RoomSpec(
        type=RoomType.KITCHEN,
        min_area=60,
        ideal_area_range=(80, 150),
        min_dimensions=(6, 8),
        requires_window=True,
        requires_ventilation=True,
        can_be_interior=False
    ),
    RoomType.BATHROOM: RoomSpec(
        type=RoomType.BATHROOM,
        min_area=30,
        ideal_area_range=(40, 80),
        min_dimensions=(5, 6),
        requires_window=False,
        requires_ventilation=True,
        can_be_interior=True
    ),
    RoomType.BALCONY: RoomSpec(
        type=RoomType.BALCONY,
        min_area=30,
        ideal_area_range=(40, 100),
        min_dimensions=(4, 6),
        requires_window=False,
        requires_ventilation=False,
        can_be_interior=False
    ),
    RoomType.CORRIDOR: RoomSpec(
        type=RoomType.CORRIDOR,
        min_area=20,
        ideal_area_range=(30, 100),
        min_dimensions=(3, 5),
        requires_window=False,
        requires_ventilation=True,
        can_be_interior=True
    ),
}


class StructuralConstants:
    """Building structural constants"""
    WALL_THICKNESS_STANDARD = 0.5  # feet (6 inches)
    WALL_THICKNESS_LOAD_BEARING = 1.0  # feet (12 inches)
    DOOR_WIDTH_STANDARD = 2.5  # feet
    DOOR_WIDTH_MAIN = 3.5  # feet
    DOOR_HEIGHT = 7.0  # feet
    WINDOW_MIN_WIDTH = 3.0  # feet
    WINDOW_MIN_HEIGHT = 4.0  # feet
    CORRIDOR_MIN_WIDTH = 3.0  # feet
    MIN_CEILING_HEIGHT = 9.0  # feet


class ValidationError:
    """Represents a validation error"""
    def __init__(self, severity: str, code: str, message: str, details: Dict = None):
        self.severity = severity  # 'critical', 'warning', 'suggestion'
        self.code = code
        self.message = message
        self.details = details or {}

    def __repr__(self):
        return f"[{self.severity.upper()}] {self.code}: {self.message}"


class LayoutValidator:
    """Validates architectural layouts against engineering rules"""
    
    def __init__(self):
        self.errors: List[ValidationError] = []
        self.warnings: List[ValidationError] = []
        
    def validate_layout(self, layout: Dict) -> Tuple[bool, List[ValidationError]]:
        """
        Validate complete layout
        Returns: (is_valid, list_of_errors)
        """
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
        self._validate_room_sizes(layout)
        self._validate_room_dimensions(layout)
        self._validate_ventilation(layout)
        self._validate_adjacencies(layout)
        self._validate_circulation(layout)
        self._validate_structural_elements(layout)
        self._validate_total_area(layout)
        
        # Critical errors make layout invalid
        is_valid = len([e for e in self.errors if e.severity == 'critical']) == 0
        
        return is_valid, self.errors + self.warnings
    
    def _validate_room_sizes(self, layout: Dict):
        """Validate room areas meet minimum requirements"""
        rooms = layout.get('rooms', [])
        
        for room in rooms:
            room_type_str = room.get('type')
            area = room.get('area', 0)
            
            try:
                room_type = RoomType(room_type_str)
                if room_type in ROOM_STANDARDS:
                    standard = ROOM_STANDARDS[room_type]
                    
                    if area < standard.min_area:
                        self.errors.append(ValidationError(
                            severity='critical',
                            code='ROOM_SIZE_TOO_SMALL',
                            message=f"{room_type.value} area ({area} sq ft) is below minimum ({standard.min_area} sq ft)",
                            details={'room': room_type.value, 'area': area, 'min_required': standard.min_area}
                        ))
                    
                    ideal_min, ideal_max = standard.ideal_area_range
                    if area < ideal_min or area > ideal_max:
                        self.warnings.append(ValidationError(
                            severity='warning',
                            code='ROOM_SIZE_NOT_IDEAL',
                            message=f"{room_type.value} area ({area} sq ft) outside ideal range ({ideal_min}-{ideal_max} sq ft)",
                            details={'room': room_type.value, 'area': area, 'ideal_range': (ideal_min, ideal_max)}
                        ))
            except ValueError:
                # Unknown room type, skip
                pass
    
    def _validate_room_dimensions(self, layout: Dict):
        """Validate room width and height meet minimum requirements"""
        rooms = layout.get('rooms', [])
        
        for room in rooms:
            room_type_str = room.get('type')
            dimensions = room.get('dimensions', {})
            width = dimensions.get('width', 0)
            height = dimensions.get('height', 0)
            
            try:
                room_type = RoomType(room_type_str)
                if room_type in ROOM_STANDARDS:
                    standard = ROOM_STANDARDS[room_type]
                    min_width, min_height = standard.min_dimensions
                    
                    if width < min_width or height < min_height:
                        self.errors.append(ValidationError(
                            severity='critical',
                            code='ROOM_DIMENSION_TOO_SMALL',
                            message=f"{room_type.value} dimensions ({width}x{height} ft) below minimum ({min_width}x{min_height} ft)",
                            details={'room': room_type.value, 'dimensions': (width, height), 'min_required': (min_width, min_height)}
                        ))
                    
                    # Check aspect ratio (rooms shouldn't be too narrow)
                    aspect_ratio = max(width, height) / min(width, height) if min(width, height) > 0 else 0
                    if aspect_ratio > 3:
                        self.warnings.append(ValidationError(
                            severity='warning',
                            code='ROOM_POOR_ASPECT_RATIO',
                            message=f"{room_type.value} has poor aspect ratio ({aspect_ratio:.1f}:1) - room too narrow",
                            details={'room': room_type.value, 'aspect_ratio': aspect_ratio}
                        ))
            except ValueError:
                pass
    
    def _validate_ventilation(self, layout: Dict):
        """Validate ventilation and natural light requirements"""
        rooms = layout.get('rooms', [])
        
        for room in rooms:
            room_type_str = room.get('type')
            windows = room.get('windows', [])
            has_exhaust = room.get('has_exhaust', False)
            
            try:
                room_type = RoomType(room_type_str)
                if room_type in ROOM_STANDARDS:
                    standard = ROOM_STANDARDS[room_type]
                    
                    # Check window requirement
                    if standard.requires_window and len(windows) == 0:
                        self.errors.append(ValidationError(
                            severity='critical',
                            code='MISSING_WINDOW',
                            message=f"{room_type.value} requires at least one window for natural light",
                            details={'room': room_type.value}
                        ))
                    
                    # Check ventilation requirement
                    if standard.requires_ventilation and len(windows) == 0 and not has_exhaust:
                        self.errors.append(ValidationError(
                            severity='critical',
                            code='MISSING_VENTILATION',
                            message=f"{room_type.value} requires ventilation (window or exhaust fan)",
                            details={'room': room_type.value}
                        ))
            except ValueError:
                pass
    
    def _validate_adjacencies(self, layout: Dict):
        """Validate room adjacency rules"""
        rooms = layout.get('rooms', [])
        connections = layout.get('connections', [])
        
        # Build adjacency map
        adjacency = {}
        for conn in connections:
            from_room = conn.get('from')
            to_room = conn.get('to')
            if from_room not in adjacency:
                adjacency[from_room] = []
            if to_room not in adjacency:
                adjacency[to_room] = []
            adjacency[from_room].append(to_room)
            adjacency[to_room].append(from_room)
        
        # Check forbidden adjacencies
        for room in rooms:
            room_id = room.get('id') or room.get('type')
            room_type_str = room.get('type')
            
            if room_type_str == 'kitchen':
                adjacent_rooms = adjacency.get(room_id, [])
                for adj in adjacent_rooms:
                    # Find adjacent room type
                    adj_room = next((r for r in rooms if (r.get('id') or r.get('type')) == adj), None)
                    if adj_room and adj_room.get('type') in ['bedroom', 'master_bedroom']:
                        self.warnings.append(ValidationError(
                            severity='warning',
                            code='KITCHEN_BEDROOM_ADJACENCY',
                            message="Kitchen directly connected to bedroom is not ideal",
                            details={'kitchen': room_id, 'bedroom': adj}
                        ))
        
        # Check bathroom accessibility
        bathrooms = [r for r in rooms if r.get('type') in ['bathroom', 'toilet']]
        for bathroom in bathrooms:
            bathroom_id = bathroom.get('id') or bathroom.get('type')
            if bathroom_id not in adjacency or len(adjacency[bathroom_id]) == 0:
                self.errors.append(ValidationError(
                    severity='critical',
                    code='BATHROOM_NOT_ACCESSIBLE',
                    message="Bathroom must be accessible from at least one room",
                    details={'bathroom': bathroom_id}
                ))
    
    def _validate_circulation(self, layout: Dict):
        """Validate circulation paths and accessibility"""
        rooms = layout.get('rooms', [])
        connections = layout.get('connections', [])
        
        # Check if there's an entrance
        entrance_exists = any(r.get('type') == 'entrance' for r in rooms)
        if not entrance_exists:
            self.warnings.append(ValidationError(
                severity='warning',
                code='NO_ENTRANCE_DEFINED',
                message="No entrance/entry point defined in layout",
                details={}
            ))
        
        # Check corridor widths
        corridors = [r for r in rooms if r.get('type') == 'corridor']
        for corridor in corridors:
            dimensions = corridor.get('dimensions', {})
            width = min(dimensions.get('width', 0), dimensions.get('height', 0))
            if width < StructuralConstants.CORRIDOR_MIN_WIDTH:
                self.errors.append(ValidationError(
                    severity='critical',
                    code='CORRIDOR_TOO_NARROW',
                    message=f"Corridor width ({width} ft) below minimum ({StructuralConstants.CORRIDOR_MIN_WIDTH} ft)",
                    details={'corridor': corridor.get('id'), 'width': width}
                ))
    
    def _validate_structural_elements(self, layout: Dict):
        """Validate doors, windows, and structural elements"""
        rooms = layout.get('rooms', [])
        
        for room in rooms:
            # Validate doors
            doors = room.get('doors', [])
            for door in doors:
                door_width = door.get('width', 0)
                if door_width < StructuralConstants.DOOR_WIDTH_STANDARD:
                    self.warnings.append(ValidationError(
                        severity='warning',
                        code='DOOR_TOO_NARROW',
                        message=f"Door width ({door_width} ft) below standard ({StructuralConstants.DOOR_WIDTH_STANDARD} ft)",
                        details={'room': room.get('type'), 'door_width': door_width}
                    ))
            
            # Validate windows
            windows = room.get('windows', [])
            for window in windows:
                window_width = window.get('width', 0)
                window_height = window.get('height', 0)
                if window_width < StructuralConstants.WINDOW_MIN_WIDTH or window_height < StructuralConstants.WINDOW_MIN_HEIGHT:
                    self.warnings.append(ValidationError(
                        severity='warning',
                        code='WINDOW_TOO_SMALL',
                        message=f"Window size ({window_width}x{window_height} ft) below minimum ({StructuralConstants.WINDOW_MIN_WIDTH}x{StructuralConstants.WINDOW_MIN_HEIGHT} ft)",
                        details={'room': room.get('type')}
                    ))
    
    def _validate_total_area(self, layout: Dict):
        """Validate total layout area and efficiency"""
        total_area = layout.get('total_area', 0)
        rooms = layout.get('rooms', [])
        
        room_area_sum = sum(r.get('area', 0) for r in rooms)
        
        if total_area > 0:
            efficiency = (room_area_sum / total_area) * 100
            
            # Typical efficiency is 75-85% (accounting for walls, corridors)
            if efficiency < 65:
                self.warnings.append(ValidationError(
                    severity='warning',
                    code='LOW_SPACE_EFFICIENCY',
                    message=f"Space efficiency ({efficiency:.1f}%) is low - too much area in walls/circulation",
                    details={'efficiency': efficiency}
                ))
            elif efficiency > 90:
                self.warnings.append(ValidationError(
                    severity='warning',
                    code='UNREALISTIC_EFFICIENCY',
                    message=f"Space efficiency ({efficiency:.1f}%) is unrealistically high",
                    details={'efficiency': efficiency}
                ))


def validate_layout_from_json(json_path: str) -> Tuple[bool, List[ValidationError]]:
    """Helper function to validate layout from JSON file"""
    with open(json_path, 'r') as f:
        layout = json.load(f)
    
    validator = LayoutValidator()
    return validator.validate_layout(layout)


# Example usage
if __name__ == "__main__":
    # Example layout
    sample_layout = {
        "layout_id": "test_001",
        "total_area": 1200,
        "rooms": [
            {
                "type": "living_room",
                "area": 250,
                "dimensions": {"width": 12, "height": 20},
                "windows": [{"position": "north", "width": 4, "height": 5}],
                "doors": [{"position": "east", "width": 3}]
            },
            {
                "type": "master_bedroom",
                "area": 180,
                "dimensions": {"width": 12, "height": 15},
                "windows": [{"position": "south", "width": 4, "height": 5}],
                "doors": [{"position": "west", "width": 2.5}],
                "attached_bathroom": True
            },
            {
                "type": "kitchen",
                "area": 100,
                "dimensions": {"width": 8, "height": 12},
                "windows": [{"position": "west", "width": 3, "height": 4}],
                "doors": [{"position": "north", "width": 2.5}],
                "has_exhaust": True
            }
        ],
        "connections": [
            {"from": "living_room", "to": "master_bedroom", "type": "door"},
            {"from": "living_room", "to": "kitchen", "type": "door"}
        ]
    }
    
    validator = LayoutValidator()
    is_valid, errors = validator.validate_layout(sample_layout)
    
    print(f"Layout Valid: {is_valid}")
    print(f"\nValidation Results:")
    for error in errors:
        print(f"  {error}")
