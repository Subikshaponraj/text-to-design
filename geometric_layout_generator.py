"""
Geometric Floor Plan Generator
Creates geometrically correct, precise CAD layouts from specifications
Exports directly to DXF with accurate dimensions and measurements
"""

import numpy as np
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

try:
    import ezdxf
    from ezdxf import colors
    from ezdxf.enums import TextEntityAlignment
    from ezdxf.gfxattribs import GfxAttribs
except ImportError:
    print("Warning: ezdxf not installed. Run: pip install ezdxf")
    ezdxf = None

try:
    import svgwrite
except ImportError:
    print("Warning: svgwrite not installed")
    svgwrite = None


class RoomType(Enum):
    """Room types with standard dimensions"""
    LIVING_ROOM = "living_room"
    MASTER_BEDROOM = "master_bedroom"
    BEDROOM = "bedroom"
    KITCHEN = "kitchen"
    BATHROOM = "bathroom"
    TOILET = "toilet"
    BALCONY = "balcony"
    DINING = "dining"
    CORRIDOR = "corridor"
    ENTRANCE = "entrance"
    STUDY = "study"
    UTILITY = "utility"


@dataclass
class Dimension:
    """Represents dimensions in feet and inches"""
    feet: float
    inches: float = 0
    
    @property
    def total_feet(self) -> float:
        """Get total dimension in feet"""
        return self.feet + (self.inches / 12.0)
    
    @property
    def total_inches(self) -> float:
        """Get total dimension in inches"""
        return (self.feet * 12.0) + self.inches
    
    def __str__(self):
        return f"{int(self.feet)}'-{int(self.inches)}\""


@dataclass
class Point:
    """2D point in CAD coordinates"""
    x: float
    y: float
    
    def __add__(self, other):
        return Point(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Point(self.x - other.x, self.y - other.y)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass
class Wall:
    """Wall definition with start and end points"""
    start: Point
    end: Point
    thickness: float = 0.5  # feet
    
    @property
    def length(self) -> float:
        """Calculate wall length"""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        return np.sqrt(dx*dx + dy*dy)


@dataclass
class Door:
    """Door definition"""
    position: Point
    width: float = 3.0  # feet
    swing_angle: float = 90.0  # degrees
    wall_side: str = "left"  # left, right, top, bottom


@dataclass
class Window:
    """Window definition"""
    position: Point
    width: float = 4.0  # feet
    height: float = 4.0  # feet
    wall_side: str = "exterior"


@dataclass
class Room:
    """Room with geometric definition"""
    id: str
    type: RoomType
    bottom_left: Point
    width: float  # feet
    height: float  # feet
    doors: List[Door]
    windows: List[Window]
    label: Optional[str] = None
    
    @property
    def area(self) -> float:
        """Calculate room area in square feet"""
        return self.width * self.height
    
    @property
    def top_right(self) -> Point:
        return Point(self.bottom_left.x + self.width, 
                    self.bottom_left.y + self.height)
    
    @property
    def center(self) -> Point:
        return Point(self.bottom_left.x + self.width/2,
                    self.bottom_left.y + self.height/2)


class GeometricFloorPlanGenerator:
    """
    Generates geometrically correct floor plans with precise measurements
    """
    
    # Standard room dimensions (feet)
    STANDARD_DIMENSIONS = {
        RoomType.LIVING_ROOM: (12, 15),
        RoomType.MASTER_BEDROOM: (12, 14),
        RoomType.BEDROOM: (10, 12),
        RoomType.KITCHEN: (8, 10),
        RoomType.BATHROOM: (5, 8),
        RoomType.TOILET: (4, 5),
        RoomType.BALCONY: (4, 10),
        RoomType.DINING: (10, 12),
        RoomType.CORRIDOR: (3, 10),
        RoomType.ENTRANCE: (4, 6),
        RoomType.STUDY: (8, 10),
        RoomType.UTILITY: (4, 6),
    }
    
    WALL_THICKNESS = 0.5  # feet
    DOOR_WIDTH = 3.0  # feet
    WINDOW_WIDTH = 4.0  # feet
    WINDOW_HEIGHT = 4.0  # feet
    
    def __init__(self):
        self.rooms: List[Room] = []
        self.walls: List[Wall] = []
        self.global_doors: List[Door] = []
        self.global_windows: List[Window] = []
        
    def parse_layout_specification(self, spec: Dict) -> None:
        """
        Parse a layout specification and create geometric layout
        
        Spec format:
        {
            "layout_type": "apartment",
            "rooms": [
                {
                    "type": "living_room",
                    "width": 15,  # feet
                    "height": 12,  # feet
                    "position": {"x": 0, "y": 0},
                    "doors": [...],
                    "windows": [...]
                }
            ]
        }
        """
        
        rooms_spec = spec.get('rooms', [])
        
        # If no positions specified, auto-layout
        if not any('position' in r for r in rooms_spec):
            self._auto_layout_rooms(rooms_spec)
        else:
            self._create_rooms_from_spec(rooms_spec)
        
        # Generate walls from room boundaries
        self._generate_walls()
    
    def _create_rooms_from_spec(self, rooms_spec: List[Dict]):
        """Create rooms from specification with explicit positions"""
        
        for idx, room_spec in enumerate(rooms_spec):
            room_type_str = room_spec.get('type', 'bedroom')
            room_type = RoomType(room_type_str)
            
            # Get dimensions
            width = room_spec.get('width')
            height = room_spec.get('height')
            
            # Use standard dimensions if not specified
            if not width or not height:
                width, height = self.STANDARD_DIMENSIONS[room_type]
            
            # Get position
            pos = room_spec.get('position', {'x': 0, 'y': 0})
            bottom_left = Point(pos['x'], pos['y'])
            
            # Create doors
            doors = []
            for door_spec in room_spec.get('doors', []):
                door_pos = Point(door_spec.get('x', 0), door_spec.get('y', 0))
                doors.append(Door(
                    position=door_pos,
                    width=door_spec.get('width', self.DOOR_WIDTH),
                    swing_angle=door_spec.get('swing_angle', 90.0),
                    wall_side=door_spec.get('wall_side', 'left')
                ))
            
            # Create windows
            windows = []
            for win_spec in room_spec.get('windows', []):
                win_pos = Point(win_spec.get('x', 0), win_spec.get('y', 0))
                windows.append(Window(
                    position=win_pos,
                    width=win_spec.get('width', self.WINDOW_WIDTH),
                    height=win_spec.get('height', self.WINDOW_HEIGHT),
                    wall_side=win_spec.get('wall_side', 'exterior')
                ))
            
            # Create room
            room = Room(
                id=f"room_{idx}",
                type=room_type,
                bottom_left=bottom_left,
                width=width,
                height=height,
                doors=doors,
                windows=windows,
                label=room_spec.get('label', room_type.value.replace('_', ' ').title())
            )
            
            self.rooms.append(room)
    
    def _auto_layout_rooms(self, rooms_spec: List[Dict]):
        """Automatically layout rooms in a grid pattern"""
        
        current_x = 0
        current_y = 0
        max_height_in_row = 0
        row_width = 0
        max_row_width = 30  # feet
        
        for idx, room_spec in enumerate(rooms_spec):
            room_type_str = room_spec.get('type', 'bedroom')
            room_type = RoomType(room_type_str)
            
            # Get dimensions
            width = room_spec.get('width')
            height = room_spec.get('height')
            
            if not width or not height:
                width, height = self.STANDARD_DIMENSIONS[room_type]
            
            # Check if we need to start a new row
            if row_width + width > max_row_width and idx > 0:
                current_x = 0
                current_y += max_height_in_row + self.WALL_THICKNESS
                row_width = 0
                max_height_in_row = 0
            
            # Create room at current position
            bottom_left = Point(current_x, current_y)
            
            # Add default door and window
            doors = [Door(
                position=Point(current_x + width/2, current_y),
                width=self.DOOR_WIDTH
            )]
            
            windows = [Window(
                position=Point(current_x + width/2, current_y + height),
                width=self.WINDOW_WIDTH
            )]
            
            room = Room(
                id=f"room_{idx}",
                type=room_type,
                bottom_left=bottom_left,
                width=width,
                height=height,
                doors=doors,
                windows=windows,
                label=room_type.value.replace('_', ' ').title()
            )
            
            self.rooms.append(room)
            
            # Update position
            current_x += width + self.WALL_THICKNESS
            row_width += width + self.WALL_THICKNESS
            max_height_in_row = max(max_height_in_row, height)
    
    def _generate_walls(self):
        """Generate walls from room boundaries"""
        
        self.walls = []
        
        for room in self.rooms:
            bl = room.bottom_left
            tr = room.top_right
            
            # Bottom wall
            self.walls.append(Wall(
                start=Point(bl.x, bl.y),
                end=Point(tr.x, bl.y),
                thickness=self.WALL_THICKNESS
            ))
            
            # Right wall
            self.walls.append(Wall(
                start=Point(tr.x, bl.y),
                end=Point(tr.x, tr.y),
                thickness=self.WALL_THICKNESS
            ))
            
            # Top wall
            self.walls.append(Wall(
                start=Point(tr.x, tr.y),
                end=Point(bl.x, tr.y),
                thickness=self.WALL_THICKNESS
            ))
            
            # Left wall
            self.walls.append(Wall(
                start=Point(bl.x, tr.y),
                end=Point(bl.x, bl.y),
                thickness=self.WALL_THICKNESS
            ))
    
    def export_to_dxf(self, output_path: str, scale: float = 1.0):
        """
        Export geometric layout to DXF with precise measurements
        
        Args:
            output_path: Output DXF file path
            scale: Scale factor (1.0 = 1 unit = 1 foot)
        """
        
        if ezdxf is None:
            print("Error: ezdxf not installed. Run: pip install ezdxf")
            return
        
        # Create DXF document
        doc = ezdxf.new('R2010', setup=True)
        msp = doc.modelspace()
        
        # Create layers
        doc.layers.add('WALLS', color=colors.WHITE, linetype='CONTINUOUS')
        doc.layers.add('DOORS', color=colors.CYAN, linetype='CONTINUOUS')
        doc.layers.add('WINDOWS', color=colors.BLUE, linetype='CONTINUOUS')
        doc.layers.add('DIMENSIONS', color=colors.GREEN, linetype='CONTINUOUS')
        doc.layers.add('ANNOTATIONS', color=colors.YELLOW, linetype='CONTINUOUS')
        doc.layers.add('ROOM_LABELS', color=colors.RED, linetype='CONTINUOUS')
        
        # Draw walls with thickness (as rectangles)
        for wall in self.walls:
            # Calculate perpendicular offset for wall thickness
            dx = wall.end.x - wall.start.x
            dy = wall.end.y - wall.start.y
            length = wall.length
            
            if length > 0:
                # Perpendicular unit vector
                perp_x = -dy / length
                perp_y = dx / length
                
                # Offset by half thickness on each side
                offset = wall.thickness / 2
                
                # Create wall as filled polyline
                p1 = Point(wall.start.x - perp_x * offset, wall.start.y - perp_y * offset)
                p2 = Point(wall.end.x - perp_x * offset, wall.end.y - perp_y * offset)
                p3 = Point(wall.end.x + perp_x * offset, wall.end.y + perp_y * offset)
                p4 = Point(wall.start.x + perp_x * offset, wall.start.y + perp_y * offset)
                
                msp.add_lwpolyline(
                    [(p1.x * scale, p1.y * scale),
                     (p2.x * scale, p2.y * scale),
                     (p3.x * scale, p3.y * scale),
                     (p4.x * scale, p4.y * scale)],
                    close=True,
                    dxfattribs={'layer': 'WALLS'}
                )
        
        # Draw rooms and add labels
        for room in self.rooms:
            bl = room.bottom_left
            tr = room.top_right
            
            # Add room boundary (lighter line)
            msp.add_lwpolyline(
                [(bl.x * scale, bl.y * scale),
                 (tr.x * scale, bl.y * scale),
                 (tr.x * scale, tr.y * scale),
                 (bl.x * scale, tr.y * scale)],
                close=True,
                dxfattribs={'layer': 'ANNOTATIONS', 'color': colors.YELLOW}
            )
            
            # Add room label at center
            center = room.center
            label_text = f"{room.label}\n{room.width:.1f}' x {room.height:.1f}'\n{room.area:.0f} sq ft"
            
            msp.add_mtext(
                label_text,
                dxfattribs={
                    'layer': 'ROOM_LABELS',
                    'char_height': 0.8 * scale,
                    'style': 'OpenSans'
                }
            ).set_location(
                insert=(center.x * scale, center.y * scale),
                attachment_point=5  # Middle center
            )
            
            # Add dimension lines
            # Width dimension (bottom)
            self._add_dimension_line(
                msp, 
                Point(bl.x * scale, bl.y * scale - 2),
                Point(tr.x * scale, bl.y * scale - 2),
                f"{room.width:.1f}'"
            )
            
            # Height dimension (left)
            self._add_dimension_line(
                msp,
                Point(bl.x * scale - 2, bl.y * scale),
                Point(bl.x * scale - 2, tr.y * scale),
                f"{room.height:.1f}'"
            )
            
            # Draw doors
            for door in room.doors:
                self._draw_door(msp, door, scale)
            
            # Draw windows
            for window in room.windows:
                self._draw_window(msp, window, scale)
        
        # Add title block
        self._add_title_block(msp, scale)
        
        # Save DXF
        doc.saveas(output_path)
        print(f"✓ DXF exported: {output_path}")
        print(f"  - Total rooms: {len(self.rooms)}")
        print(f"  - Total walls: {len(self.walls)}")
        print(f"  - Scale: 1 unit = 1 foot")
    
    def _draw_door(self, msp, door: Door, scale: float):
        """Draw door symbol in DXF"""
        pos = door.position
        width = door.width
        
        # Draw door arc (90 degree swing)
        msp.add_arc(
            center=(pos.x * scale, pos.y * scale),
            radius=width * scale,
            start_angle=0,
            end_angle=door.swing_angle,
            dxfattribs={'layer': 'DOORS'}
        )
        
        # Draw door line
        end_x = pos.x + width * np.cos(np.radians(door.swing_angle))
        end_y = pos.y + width * np.sin(np.radians(door.swing_angle))
        
        msp.add_line(
            (pos.x * scale, pos.y * scale),
            (end_x * scale, end_y * scale),
            dxfattribs={'layer': 'DOORS'}
        )
    
    def _draw_window(self, msp, window: Window, scale: float):
        """Draw window symbol in DXF"""
        pos = window.position
        w = window.width / 2
        h = 0.3  # Window thickness in plan
        
        # Draw window as rectangle with cross lines
        points = [
            ((pos.x - w) * scale, (pos.y - h) * scale),
            ((pos.x + w) * scale, (pos.y - h) * scale),
            ((pos.x + w) * scale, (pos.y + h) * scale),
            ((pos.x - w) * scale, (pos.y + h) * scale)
        ]
        
        msp.add_lwpolyline(
            points,
            close=True,
            dxfattribs={'layer': 'WINDOWS'}
        )
        
        # Cross lines
        msp.add_line(
            ((pos.x - w) * scale, pos.y * scale),
            ((pos.x + w) * scale, pos.y * scale),
            dxfattribs={'layer': 'WINDOWS'}
        )
    
    def _add_dimension_line(self, msp, start: Point, end: Point, text: str):
        """Add dimension line with text"""
        # Draw dimension line
        msp.add_line(
            start.to_tuple(),
            end.to_tuple(),
            dxfattribs={'layer': 'DIMENSIONS'}
        )
        
        # Add extension lines
        ext_length = 1.0
        if abs(end.x - start.x) > abs(end.y - start.y):  # Horizontal
            msp.add_line((start.x, start.y), (start.x, start.y + ext_length), dxfattribs={'layer': 'DIMENSIONS'})
            msp.add_line((end.x, end.y), (end.x, end.y + ext_length), dxfattribs={'layer': 'DIMENSIONS'})
        else:  # Vertical
            msp.add_line((start.x, start.y), (start.x + ext_length, start.y), dxfattribs={'layer': 'DIMENSIONS'})
            msp.add_line((end.x, end.y), (end.x + ext_length, end.y), dxfattribs={'layer': 'DIMENSIONS'})
        
        # Add dimension text
        mid_x = (start.x + end.x) / 2
        mid_y = (start.y + end.y) / 2
        
        msp.add_text(
            text,
            dxfattribs={
                'layer': 'DIMENSIONS',
                'height': 0.6
            }
        ).set_placement((mid_x, mid_y), align=TextEntityAlignment.MIDDLE_CENTER)
    
    def _add_title_block(self, msp, scale: float):
        """Add title block to drawing"""
        # Calculate total bounds
        if not self.rooms:
            return
        
        min_x = min(r.bottom_left.x for r in self.rooms)
        max_x = max(r.top_right.x for r in self.rooms)
        min_y = min(r.bottom_left.y for r in self.rooms)
        
        # Title block position
        title_y = (min_y - 5) * scale
        
        msp.add_text(
            "FLOOR PLAN - ARCHITECTURAL DRAWING",
            dxfattribs={
                'layer': 'ANNOTATIONS',
                'height': 1.0 * scale,
                'style': 'OpenSans'
            }
        ).set_placement((min_x * scale, title_y), align=TextEntityAlignment.BOTTOM_LEFT)
        
        msp.add_text(
            f"Scale: 1:{int(1/scale)} | All dimensions in feet",
            dxfattribs={
                'layer': 'ANNOTATIONS',
                'height': 0.5 * scale
            }
        ).set_placement((min_x * scale, title_y - 1.5), align=TextEntityAlignment.BOTTOM_LEFT)
    
    def export_to_svg(self, output_path: str, scale: float = 20.0):
        """Export to SVG format"""
        
        if svgwrite is None:
            print("Error: svgwrite not installed")
            return
        
        # Calculate bounds
        if not self.rooms:
            return
        
        min_x = min(r.bottom_left.x for r in self.rooms) - 5
        max_x = max(r.top_right.x for r in self.rooms) + 5
        min_y = min(r.bottom_left.y for r in self.rooms) - 8
        max_y = max(r.top_right.y for r in self.rooms) + 5
        
        width = (max_x - min_x) * scale
        height = (max_y - min_y) * scale
        
        dwg = svgwrite.Drawing(output_path, size=(f'{width}px', f'{height}px'))
        
        # Define viewBox (flip Y axis for SVG)
        dwg.viewbox(min_x * scale, -max_y * scale, width, height)
        
        # Draw walls
        for wall in self.walls:
            dwg.add(dwg.line(
                start=(wall.start.x * scale, -wall.start.y * scale),
                end=(wall.end.x * scale, -wall.end.y * scale),
                stroke='black',
                stroke_width=wall.thickness * scale
            ))
        
        # Draw rooms
        for room in self.rooms:
            bl = room.bottom_left
            tr = room.top_right
            
            # Room outline
            dwg.add(dwg.rect(
                insert=(bl.x * scale, -tr.y * scale),
                size=(room.width * scale, room.height * scale),
                fill='none',
                stroke='gray',
                stroke_width=0.5
            ))
            
            # Room label
            center = room.center
            dwg.add(dwg.text(
                room.label,
                insert=(center.x * scale, -center.y * scale),
                text_anchor='middle',
                font_size=f'{0.8 * scale}px',
                fill='black'
            ))
        
        dwg.save()
        print(f"✓ SVG exported: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    # Example: Create a 2BHK apartment with precise measurements
    
    generator = GeometricFloorPlanGenerator()
    
    # Define layout specification with precise dimensions
    layout_spec = {
        "layout_type": "apartment",
        "rooms": [
            {
                "type": "living_room",
                "width": 15,
                "height": 12,
                "position": {"x": 0, "y": 0},
                "label": "Living Room",
                "windows": [{"x": 7.5, "y": 12}],
                "doors": [{"x": 0, "y": 6}]
            },
            {
                "type": "master_bedroom",
                "width": 14,
                "height": 12,
                "position": {"x": 15.5, "y": 0},
                "label": "Master Bedroom",
                "windows": [{"x": 22.5, "y": 12}],
                "doors": [{"x": 15.5, "y": 6}]
            },
            {
                "type": "bedroom",
                "width": 12,
                "height": 10,
                "position": {"x": 0, "y": 12.5},
                "label": "Bedroom 2",
                "windows": [{"x": 6, "y": 22.5}],
                "doors": [{"x": 0, "y": 17}]
            },
            {
                "type": "kitchen",
                "width": 10,
                "height": 8,
                "position": {"x": 12.5, "y": 12.5},
                "label": "Kitchen",
                "windows": [{"x": 17.5, "y": 20.5}],
                "doors": [{"x": 12.5, "y": 16}]
            },
            {
                "type": "bathroom",
                "width": 7,
                "height": 5,
                "position": {"x": 22.5, "y": 12.5},
                "label": "Bathroom",
                "windows": [{"x": 26, "y": 17.5}],
                "doors": [{"x": 22.5, "y": 15}]
            }
        ]
    }
    
    # Parse specification and create geometric layout
    generator.parse_layout_specification(layout_spec)
    
    # Export to DXF
    generator.export_to_dxf("geometric_floor_plan.dxf", scale=1.0)
    
    # Export to SVG
    generator.export_to_svg("geometric_floor_plan.svg", scale=20.0)
    
    print("\nGeometric floor plan generated successfully!")
    print("Open geometric_floor_plan.dxf in AutoCAD or any CAD software")