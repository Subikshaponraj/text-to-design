"""
Constraint Engine for Floor Plan Validation and Correction
Validates and corrects layouts before geometric rendering

Architecture:
Input → Constraint Engine → Output
  ↓          ↓                 ↓
Raw      Validate &         Clean
Layout   Correct           Layout

Validation Rules:
1. Non-overlapping rooms
2. Minimum area requirements
3. Adjacency constraints
4. Connectivity rules
5. Dimensional constraints
6. Building code compliance
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum
import copy


class ConstraintViolationType(Enum):
    """Types of constraint violations"""
    OVERLAP = "overlap"
    MINIMUM_AREA = "minimum_area"
    MAXIMUM_AREA = "maximum_area"
    ADJACENCY = "adjacency"
    CONNECTIVITY = "connectivity"
    DIMENSIONS = "dimensions"
    ACCESS = "access"
    VENTILATION = "ventilation"
    BUILDING_CODE = "building_code"


class SeverityLevel(Enum):
    """Severity levels for violations"""
    CRITICAL = "critical"  # Must fix - layout unusable
    ERROR = "error"        # Should fix - layout problematic
    WARNING = "warning"    # Can ignore - layout acceptable


@dataclass
class ConstraintViolation:
    """Represents a constraint violation"""
    type: ConstraintViolationType
    severity: SeverityLevel
    message: str
    affected_rooms: List[str]
    details: Dict
    auto_fixable: bool = False


@dataclass
class Rectangle:
    """Represents a room as a rectangle"""
    x: float  # Bottom-left X
    y: float  # Bottom-left Y
    width: float
    height: float
    room_id: str
    room_type: str
    
    @property
    def x2(self) -> float:
        return self.x + self.width
    
    @property
    def y2(self) -> float:
        return self.y + self.height
    
    @property
    def area(self) -> float:
        return self.width * self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        return (self.x + self.width/2, self.y + self.height/2)
    
    def overlaps(self, other: 'Rectangle') -> bool:
        """Check if this rectangle overlaps with another"""
        return not (self.x2 <= other.x or 
                   self.x >= other.x2 or 
                   self.y2 <= other.y or 
                   self.y >= other.y2)
    
    def overlap_area(self, other: 'Rectangle') -> float:
        """Calculate overlap area with another rectangle"""
        if not self.overlaps(other):
            return 0.0
        
        x_overlap = min(self.x2, other.x2) - max(self.x, other.x)
        y_overlap = min(self.y2, other.y2) - max(self.y, other.y)
        
        return x_overlap * y_overlap
    
    def is_adjacent(self, other: 'Rectangle', tolerance: float = 0.5) -> bool:
        """Check if this room is adjacent to another"""
        # Check if they share a wall (within tolerance)
        
        # Horizontal adjacency (side by side)
        if abs(self.x2 - other.x) < tolerance or abs(self.x - other.x2) < tolerance:
            # Check vertical overlap
            y_overlap = min(self.y2, other.y2) - max(self.y, other.y)
            if y_overlap > min(self.height, other.height) * 0.3:  # 30% overlap minimum
                return True
        
        # Vertical adjacency (top/bottom)
        if abs(self.y2 - other.y) < tolerance or abs(self.y - other.y2) < tolerance:
            # Check horizontal overlap
            x_overlap = min(self.x2, other.x2) - max(self.x, other.x)
            if x_overlap > min(self.width, other.width) * 0.3:
                return True
        
        return False


class ConstraintEngine:
    """
    Validates and corrects floor plan layouts
    Ensures architectural validity before geometric rendering
    """
    
    # Minimum room sizes (sq ft)
    MINIMUM_AREAS = {
        'living_room': 120,
        'master_bedroom': 100,
        'bedroom': 80,
        'kitchen': 50,
        'bathroom': 25,
        'toilet': 20,
        'balcony': 20,
        'dining': 80,
        'hall': 150,
        'corridor': 15,
        'puja': 25,
        'study': 60,
        'utility': 20,
        'portico': 80,
    }
    
    # Maximum room sizes (sq ft)
    MAXIMUM_AREAS = {
        'living_room': 400,
        'master_bedroom': 300,
        'bedroom': 250,
        'kitchen': 200,
        'bathroom': 80,
        'toilet': 40,
        'balcony': 100,
        'dining': 250,
        'hall': 500,
        'corridor': 100,
        'puja': 50,
        'study': 150,
        'utility': 60,
        'portico': 400,
    }
    
    # Minimum dimensions (feet)
    MINIMUM_DIMENSIONS = {
        'living_room': (10, 10),
        'bedroom': (9, 8),
        'master_bedroom': (10, 10),
        'kitchen': (6, 6),
        'bathroom': (5, 4),
        'toilet': (4, 3),
    }
    
    # Required adjacencies (room pairs that should be adjacent)
    REQUIRED_ADJACENCIES = {
        'master_bedroom': ['bathroom'],  # Master bedroom should have adjacent bathroom
    }
    
    # Prohibited adjacencies (room pairs that should NOT be adjacent)
    PROHIBITED_ADJACENCIES = {
        'kitchen': ['bathroom', 'toilet'],  # Kitchen shouldn't touch bathrooms
    }
    
    # Rooms that require ventilation (window or exhaust)
    VENTILATION_REQUIRED = ['kitchen', 'bathroom', 'toilet', 'bedroom', 'living_room', 'hall']
    
    def __init__(
        self,
        max_iterations: int = 10,
        auto_correct: bool = True,
        strict_mode: bool = False
    ):
        """
        Initialize constraint engine
        
        Args:
            max_iterations: Maximum attempts to fix violations
            auto_correct: Automatically fix violations when possible
            strict_mode: Enforce all warnings as errors
        """
        self.max_iterations = max_iterations
        self.auto_correct = auto_correct
        self.strict_mode = strict_mode
        
        self.violations: List[ConstraintViolation] = []
        self.corrections_applied: List[str] = []
    
    def validate_and_correct(self, layout_spec: Dict) -> Tuple[bool, Dict, List[ConstraintViolation]]:
        """
        Validate layout and apply corrections
        
        Args:
            layout_spec: Layout specification to validate
            
        Returns:
            (is_valid, corrected_layout, violations)
        """
        
        print(f"\n{'='*80}")
        print("CONSTRAINT ENGINE - Validation and Correction")
        print(f"{'='*80}\n")
        
        self.violations = []
        self.corrections_applied = []
        
        # Deep copy to avoid modifying original
        corrected_layout = copy.deepcopy(layout_spec)
        
        # Iterative correction
        for iteration in range(self.max_iterations):
            print(f"Iteration {iteration + 1}/{self.max_iterations}")
            print("-" * 40)
            
            # Run all validation checks
            current_violations = self._run_all_checks(corrected_layout)
            
            # Count by severity
            critical = sum(1 for v in current_violations if v.severity == SeverityLevel.CRITICAL)
            errors = sum(1 for v in current_violations if v.severity == SeverityLevel.ERROR)
            warnings = sum(1 for v in current_violations if v.severity == SeverityLevel.WARNING)
            
            print(f"  Violations: {critical} critical, {errors} errors, {warnings} warnings")
            
            # Check if valid
            if self.strict_mode:
                is_valid = len(current_violations) == 0
            else:
                is_valid = critical == 0 and errors == 0
            
            if is_valid:
                print(f"✓ Layout is valid!")
                break
            
            # Try to fix violations
            if self.auto_correct:
                print(f"  Applying corrections...")
                corrected_layout = self._apply_corrections(corrected_layout, current_violations)
            else:
                print(f"  Auto-correction disabled")
                break
        
        # Final validation
        self.violations = self._run_all_checks(corrected_layout)
        
        # Final status
        critical = sum(1 for v in self.violations if v.severity == SeverityLevel.CRITICAL)
        errors = sum(1 for v in self.violations if v.severity == SeverityLevel.ERROR)
        warnings = sum(1 for v in self.violations if v.severity == SeverityLevel.WARNING)
        
        if self.strict_mode:
            is_valid = len(self.violations) == 0
        else:
            is_valid = critical == 0 and errors == 0
        
        print(f"\n{'='*80}")
        print("CONSTRAINT ENGINE - Final Report")
        print(f"{'='*80}")
        print(f"Status: {'✓ VALID' if is_valid else '✗ INVALID'}")
        print(f"Violations: {critical} critical, {errors} errors, {warnings} warnings")
        print(f"Corrections Applied: {len(self.corrections_applied)}")
        
        if self.corrections_applied:
            print(f"\nCorrections:")
            for correction in self.corrections_applied:
                print(f"  • {correction}")
        
        if self.violations:
            print(f"\nRemaining Violations:")
            for violation in self.violations[:5]:  # Show first 5
                print(f"  [{violation.severity.value.upper()}] {violation.message}")
        
        print(f"{'='*80}\n")
        
        return is_valid, corrected_layout, self.violations
    
    def _run_all_checks(self, layout_spec: Dict) -> List[ConstraintViolation]:
        """Run all validation checks"""
        
        violations = []
        
        # Extract rooms as rectangles
        rectangles = self._extract_rectangles(layout_spec)
        
        # Check 1: Non-overlapping rooms
        violations.extend(self._check_overlaps(rectangles))
        
        # Check 2: Minimum area requirements
        violations.extend(self._check_minimum_areas(rectangles))
        
        # Check 3: Maximum area constraints
        violations.extend(self._check_maximum_areas(rectangles))
        
        # Check 4: Minimum dimensions
        violations.extend(self._check_minimum_dimensions(rectangles))
        
        # Check 5: Adjacency constraints
        violations.extend(self._check_adjacencies(rectangles, layout_spec))
        
        # Check 6: Connectivity (all rooms accessible)
        violations.extend(self._check_connectivity(rectangles, layout_spec))
        
        # Check 7: Ventilation requirements
        violations.extend(self._check_ventilation(layout_spec))
        
        # Check 8: Building code compliance
        violations.extend(self._check_building_codes(rectangles))
        
        return violations
    
    def _extract_rectangles(self, layout_spec: Dict) -> List[Rectangle]:
        """Convert layout spec to list of rectangles"""
        
        rectangles = []
        
        for idx, room in enumerate(layout_spec.get('rooms', [])):
            pos = room.get('position', {})
            
            rect = Rectangle(
                x=pos.get('x', 0),
                y=pos.get('y', 0),
                width=room.get('width', 10),
                height=room.get('height', 10),
                room_id=room.get('label', f"Room_{idx}"),
                room_type=room.get('type', 'unknown')
            )
            
            rectangles.append(rect)
        
        return rectangles
    
    def _check_overlaps(self, rectangles: List[Rectangle]) -> List[ConstraintViolation]:
        """Check for overlapping rooms"""
        
        violations = []
        
        for i, rect1 in enumerate(rectangles):
            for rect2 in rectangles[i+1:]:
                if rect1.overlaps(rect2):
                    overlap_area = rect1.overlap_area(rect2)
                    
                    violations.append(ConstraintViolation(
                        type=ConstraintViolationType.OVERLAP,
                        severity=SeverityLevel.CRITICAL,
                        message=f"Rooms overlap: {rect1.room_id} and {rect2.room_id}",
                        affected_rooms=[rect1.room_id, rect2.room_id],
                        details={
                            'overlap_area': overlap_area,
                            'rect1': (rect1.x, rect1.y, rect1.width, rect1.height),
                            'rect2': (rect2.x, rect2.y, rect2.width, rect2.height)
                        },
                        auto_fixable=True
                    ))
        
        return violations
    
    def _check_minimum_areas(self, rectangles: List[Rectangle]) -> List[ConstraintViolation]:
        """Check minimum area requirements"""
        
        violations = []
        
        for rect in rectangles:
            min_area = self.MINIMUM_AREAS.get(rect.room_type, 0)
            
            if rect.area < min_area:
                violations.append(ConstraintViolation(
                    type=ConstraintViolationType.MINIMUM_AREA,
                    severity=SeverityLevel.ERROR,
                    message=f"{rect.room_id}: Area {rect.area:.1f} sq ft < minimum {min_area} sq ft",
                    affected_rooms=[rect.room_id],
                    details={
                        'current_area': rect.area,
                        'minimum_area': min_area,
                        'deficit': min_area - rect.area
                    },
                    auto_fixable=True
                ))
        
        return violations
    
    def _check_maximum_areas(self, rectangles: List[Rectangle]) -> List[ConstraintViolation]:
        """Check maximum area constraints"""
        
        violations = []
        
        for rect in rectangles:
            max_area = self.MAXIMUM_AREAS.get(rect.room_type, float('inf'))
            
            if rect.area > max_area:
                violations.append(ConstraintViolation(
                    type=ConstraintViolationType.MAXIMUM_AREA,
                    severity=SeverityLevel.WARNING,
                    message=f"{rect.room_id}: Area {rect.area:.1f} sq ft > maximum {max_area} sq ft",
                    affected_rooms=[rect.room_id],
                    details={
                        'current_area': rect.area,
                        'maximum_area': max_area,
                        'excess': rect.area - max_area
                    },
                    auto_fixable=True
                ))
        
        return violations
    
    def _check_minimum_dimensions(self, rectangles: List[Rectangle]) -> List[ConstraintViolation]:
        """Check minimum dimension requirements"""
        
        violations = []
        
        for rect in rectangles:
            min_dims = self.MINIMUM_DIMENSIONS.get(rect.room_type)
            
            if min_dims:
                min_width, min_height = min_dims
                
                if rect.width < min_width or rect.height < min_height:
                    violations.append(ConstraintViolation(
                        type=ConstraintViolationType.DIMENSIONS,
                        severity=SeverityLevel.ERROR,
                        message=f"{rect.room_id}: Dimensions {rect.width:.1f}'×{rect.height:.1f}' below minimum {min_width}'×{min_height}'",
                        affected_rooms=[rect.room_id],
                        details={
                            'current': (rect.width, rect.height),
                            'minimum': min_dims
                        },
                        auto_fixable=True
                    ))
        
        return violations
    
    def _check_adjacencies(self, rectangles: List[Rectangle], layout_spec: Dict) -> List[ConstraintViolation]:
        """Check adjacency constraints"""
        
        violations = []
        
        # Build adjacency map
        adjacency_map = {}
        for rect1 in rectangles:
            adjacent = []
            for rect2 in rectangles:
                if rect1.room_id != rect2.room_id and rect1.is_adjacent(rect2):
                    adjacent.append(rect2.room_type)
            adjacency_map[rect1.room_type] = adjacent
        
        # Check required adjacencies
        for room_type, required_adjacent in self.REQUIRED_ADJACENCIES.items():
            rects_of_type = [r for r in rectangles if r.room_type == room_type]
            
            for rect in rects_of_type:
                actual_adjacent = adjacency_map.get(rect.room_type, [])
                
                for required in required_adjacent:
                    if required not in actual_adjacent:
                        violations.append(ConstraintViolation(
                            type=ConstraintViolationType.ADJACENCY,
                            severity=SeverityLevel.WARNING,
                            message=f"{rect.room_id} should be adjacent to {required}",
                            affected_rooms=[rect.room_id],
                            details={
                                'required_adjacent': required,
                                'actual_adjacent': actual_adjacent
                            },
                            auto_fixable=False  # Requires layout restructuring
                        ))
        
        # Check prohibited adjacencies
        for room_type, prohibited_adjacent in self.PROHIBITED_ADJACENCIES.items():
            rects_of_type = [r for r in rectangles if r.room_type == room_type]
            
            for rect in rects_of_type:
                actual_adjacent = adjacency_map.get(rect.room_type, [])
                
                for prohibited in prohibited_adjacent:
                    if prohibited in actual_adjacent:
                        violations.append(ConstraintViolation(
                            type=ConstraintViolationType.ADJACENCY,
                            severity=SeverityLevel.ERROR,
                            message=f"{rect.room_id} should NOT be adjacent to {prohibited}",
                            affected_rooms=[rect.room_id],
                            details={
                                'prohibited_adjacent': prohibited,
                                'actual_adjacent': actual_adjacent
                            },
                            auto_fixable=False
                        ))
        
        return violations
    
    def _check_connectivity(self, rectangles: List[Rectangle], layout_spec: Dict) -> List[ConstraintViolation]:
        """Check that all rooms are accessible (connected)"""
        
        violations = []
        
        # Build connectivity graph from doors
        connections = set()
        
        for room in layout_spec.get('rooms', []):
            room_id = room.get('label', '')
            # Assume rooms with doors are connected to adjacent rooms
            if room.get('doors'):
                connections.add(room_id)
        
        # For now, simplified check: all rooms should have at least one door
        for room in layout_spec.get('rooms', []):
            room_id = room.get('label', '')
            doors = room.get('doors', [])
            
            if not doors and room.get('type') not in ['balcony']:  # Balconies don't need doors
                violations.append(ConstraintViolation(
                    type=ConstraintViolationType.ACCESS,
                    severity=SeverityLevel.ERROR,
                    message=f"{room_id} has no door (inaccessible)",
                    affected_rooms=[room_id],
                    details={'doors': len(doors)},
                    auto_fixable=True
                ))
        
        return violations
    
    def _check_ventilation(self, layout_spec: Dict) -> List[ConstraintViolation]:
        """Check ventilation requirements"""
        
        violations = []
        
        for room in layout_spec.get('rooms', []):
            room_type = room.get('type', '')
            room_id = room.get('label', '')
            
            if room_type in self.VENTILATION_REQUIRED:
                windows = room.get('windows', [])
                has_exhaust = room.get('has_exhaust', False)
                
                if not windows and not has_exhaust:
                    violations.append(ConstraintViolation(
                        type=ConstraintViolationType.VENTILATION,
                        severity=SeverityLevel.WARNING,
                        message=f"{room_id} lacks ventilation (no window or exhaust)",
                        affected_rooms=[room_id],
                        details={
                            'windows': len(windows),
                            'has_exhaust': has_exhaust
                        },
                        auto_fixable=True
                    ))
        
        return violations
    
    def _check_building_codes(self, rectangles: List[Rectangle]) -> List[ConstraintViolation]:
        """Check basic building code compliance"""
        
        violations = []
        
        # Check aspect ratios (rooms shouldn't be too narrow/long)
        for rect in rectangles:
            aspect_ratio = max(rect.width, rect.height) / min(rect.width, rect.height)
            
            if aspect_ratio > 3.0:  # Room is more than 3:1 ratio
                violations.append(ConstraintViolation(
                    type=ConstraintViolationType.BUILDING_CODE,
                    severity=SeverityLevel.WARNING,
                    message=f"{rect.room_id}: Unusual aspect ratio {aspect_ratio:.1f}:1",
                    affected_rooms=[rect.room_id],
                    details={
                        'aspect_ratio': aspect_ratio,
                        'dimensions': (rect.width, rect.height)
                    },
                    auto_fixable=True
                ))
        
        return violations
    
    def _apply_corrections(self, layout_spec: Dict, violations: List[ConstraintViolation]) -> Dict:
        """Apply automatic corrections to layout"""
        
        corrected = copy.deepcopy(layout_spec)
        
        for violation in violations:
            if not violation.auto_fixable:
                continue
            
            if violation.type == ConstraintViolationType.OVERLAP:
                corrected = self._fix_overlap(corrected, violation)
            
            elif violation.type == ConstraintViolationType.MINIMUM_AREA:
                corrected = self._fix_minimum_area(corrected, violation)
            
            elif violation.type == ConstraintViolationType.MAXIMUM_AREA:
                corrected = self._fix_maximum_area(corrected, violation)
            
            elif violation.type == ConstraintViolationType.DIMENSIONS:
                corrected = self._fix_dimensions(corrected, violation)
            
            elif violation.type == ConstraintViolationType.ACCESS:
                corrected = self._fix_access(corrected, violation)
            
            elif violation.type == ConstraintViolationType.VENTILATION:
                corrected = self._fix_ventilation(corrected, violation)
        
        return corrected
    
    def _fix_overlap(self, layout_spec: Dict, violation: ConstraintViolation) -> Dict:
        """Fix overlapping rooms by moving them apart"""
        
        room1_id, room2_id = violation.affected_rooms
        
        # Find rooms
        room1 = next((r for r in layout_spec['rooms'] if r.get('label') == room1_id), None)
        room2 = next((r for r in layout_spec['rooms'] if r.get('label') == room2_id), None)
        
        if room1 and room2:
            # Move room2 to the right of room1
            room1_right = room1['position']['x'] + room1['width']
            room2['position']['x'] = room1_right + 0.5  # 0.5 ft gap
            
            self.corrections_applied.append(f"Moved {room2_id} to avoid overlap with {room1_id}")
        
        return layout_spec
    
    def _fix_minimum_area(self, layout_spec: Dict, violation: ConstraintViolation) -> Dict:
        """Fix room below minimum area by increasing size"""
        
        room_id = violation.affected_rooms[0]
        room = next((r for r in layout_spec['rooms'] if r.get('label') == room_id), None)
        
        if room:
            min_area = violation.details['minimum_area']
            current_area = violation.details['current_area']
            
            # Increase proportionally
            scale = np.sqrt(min_area / current_area)
            room['width'] *= scale
            room['height'] *= scale
            
            self.corrections_applied.append(f"Increased {room_id} size to meet minimum area")
        
        return layout_spec
    
    def _fix_maximum_area(self, layout_spec: Dict, violation: ConstraintViolation) -> Dict:
        """Fix room exceeding maximum area"""
        
        room_id = violation.affected_rooms[0]
        room = next((r for r in layout_spec['rooms'] if r.get('label') == room_id), None)
        
        if room:
            max_area = violation.details['maximum_area']
            current_area = violation.details['current_area']
            
            # Decrease proportionally
            scale = np.sqrt(max_area / current_area)
            room['width'] *= scale
            room['height'] *= scale
            
            self.corrections_applied.append(f"Reduced {room_id} size to meet maximum area")
        
        return layout_spec
    
    def _fix_dimensions(self, layout_spec: Dict, violation: ConstraintViolation) -> Dict:
        """Fix room with dimensions below minimum"""
        
        room_id = violation.affected_rooms[0]
        room = next((r for r in layout_spec['rooms'] if r.get('label') == room_id), None)
        
        if room:
            min_width, min_height = violation.details['minimum']
            room['width'] = max(room['width'], min_width)
            room['height'] = max(room['height'], min_height)
            
            self.corrections_applied.append(f"Adjusted {room_id} dimensions to meet minimums")
        
        return layout_spec
    
    def _fix_access(self, layout_spec: Dict, violation: ConstraintViolation) -> Dict:
        """Fix inaccessible room by adding door"""
        
        room_id = violation.affected_rooms[0]
        room = next((r for r in layout_spec['rooms'] if r.get('label') == room_id), None)
        
        if room:
            if 'doors' not in room:
                room['doors'] = []
            
            # Add a door on the left side
            room['doors'].append({
                'x': room['position']['x'],
                'y': room['position']['y'] + room['height']/2,
                'width': 3.0
            })
            
            self.corrections_applied.append(f"Added door to {room_id} for access")
        
        return layout_spec
    
    def _fix_ventilation(self, layout_spec: Dict, violation: ConstraintViolation) -> Dict:
        """Fix ventilation by adding window or exhaust"""
        
        room_id = violation.affected_rooms[0]
        room = next((r for r in layout_spec['rooms'] if r.get('label') == room_id), None)
        
        if room:
            room_type = room.get('type', '')
            
            if room_type in ['bathroom', 'toilet']:
                # Add exhaust for bathrooms
                room['has_exhaust'] = True
                self.corrections_applied.append(f"Added exhaust to {room_id}")
            else:
                # Add window for other rooms
                if 'windows' not in room:
                    room['windows'] = []
                
                room['windows'].append({
                    'x': room['position']['x'] + room['width']/2,
                    'y': room['position']['y'] + room['height'],
                    'width': 4.0,
                    'height': 4.0
                })
                
                self.corrections_applied.append(f"Added window to {room_id} for ventilation")
        
        return layout_spec


# Integration functions
def validate_layout(layout_spec: Dict, auto_correct: bool = True) -> Tuple[bool, Dict, List[ConstraintViolation]]:
    """
    Validate and optionally correct a layout specification
    
    Args:
        layout_spec: Layout specification dictionary
        auto_correct: Automatically fix violations
        
    Returns:
        (is_valid, corrected_layout, violations)
    """
    
    engine = ConstraintEngine(
        max_iterations=10,
        auto_correct=auto_correct,
        strict_mode=False
    )
    
    return engine.validate_and_correct(layout_spec)


# Example usage
if __name__ == "__main__":
    
    # Example layout with intentional violations
    test_layout = {
        'layout_type': 'apartment',
        'rooms': [
            {
                'type': 'living_room',
                'label': 'Living Room',
                'width': 8,  # Too small
                'height': 7,  # Too small
                'position': {'x': 0, 'y': 0},
                'doors': [],  # No door - violation
                'windows': []  # No window - violation
            },
            {
                'type': 'bedroom',
                'label': 'Bedroom',
                'width': 12,
                'height': 10,
                'position': {'x': 5, 'y': 0},  # Overlaps with living room
                'doors': [{'x': 5, 'y': 5}],
                'windows': [{'x': 11, 'y': 10}]
            }
        ]
    }
    
    # Validate and correct
    is_valid, corrected, violations = validate_layout(test_layout, auto_correct=True)
    
    print(f"\nFinal Result: {'VALID' if is_valid else 'INVALID'}")
    print(f"Violations remaining: {len(violations)}")