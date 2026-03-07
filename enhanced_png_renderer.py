"""
Enhanced Geometric Floor Plan Generator with PNG Rendering
Generates geometrically correct floor plans and renders them as PNG images
with precise measurements and annotations
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import os

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Warning: Pillow not installed. Run: pip install Pillow")
    Image = None

try:
    import ezdxf
    from ezdxf import colors
    from ezdxf.enums import TextEntityAlignment
except ImportError:
    ezdxf = None

try:
    import svgwrite
except ImportError:
    svgwrite = None

from geometric_layout_generator import (
    GeometricFloorPlanGenerator, 
    Point, 
    Wall, 
    Door, 
    Window, 
    Room,
    RoomType
)


class PNGFloorPlanRenderer:
    """
    Renders geometric floor plans as high-quality PNG images
    with dimensions, labels, and measurements
    """
    
    def __init__(
        self,
        width: int = 2400,
        height: int = 2400,
        dpi: int = 300,
        scale: float = 40.0  # pixels per foot
    ):
        """
        Initialize PNG renderer
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            dpi: Dots per inch for print quality
            scale: Pixels per foot (40 = good detail)
        """
        self.width = width
        self.height = height
        self.dpi = dpi
        self.scale = scale
        
        # Colors (RGB)
        self.COLORS = {
            'background': (255, 255, 255),       # White
            'wall': (0, 0, 0),                   # Black
            'wall_fill': (240, 240, 240),        # Light gray
            'door': (139, 69, 19),               # Brown
            'window': (70, 130, 180),            # Steel blue
            'dimension_line': (0, 128, 0),       # Green
            'room_label': (0, 0, 0),             # Black
            'grid': (230, 230, 230),             # Very light gray
            'room_fill': (250, 250, 245),        # Ivory
        }
        
        # Line widths (pixels)
        self.LINE_WIDTHS = {
            'wall': 4,
            'door': 3,
            'window': 2,
            'dimension': 2,
            'grid': 1,
        }
    
    def render(
        self,
        generator: GeometricFloorPlanGenerator,
        output_path: str,
        show_dimensions: bool = True,
        show_grid: bool = True,
        show_room_areas: bool = True,
        show_title: bool = True
    ):
        """
        Render floor plan to PNG image
        
        Args:
            generator: GeometricFloorPlanGenerator instance
            output_path: Output PNG file path
            show_dimensions: Show dimension annotations
            show_grid: Show background grid
            show_room_areas: Show room area labels
            show_title: Show title block
        """
        
        if Image is None:
            print("Error: Pillow not installed. Run: pip install Pillow")
            return
        
        # Calculate bounds
        if not generator.rooms:
            print("Error: No rooms in layout")
            return
        
        min_x = min(r.bottom_left.x for r in generator.rooms) - 5
        max_x = max(r.top_right.x for r in generator.rooms) + 5
        min_y = min(r.bottom_left.y for r in generator.rooms) - 8
        max_y = max(r.top_right.y for r in generator.rooms) + 5
        
        # Calculate image dimensions
        layout_width = max_x - min_x
        layout_height = max_y - min_y
        
        # Adjust scale if needed to fit
        scale_x = (self.width - 200) / (layout_width * self.scale)
        scale_y = (self.height - 200) / (layout_height * self.scale)
        scale_factor = min(scale_x, scale_y, 1.0)
        
        final_scale = self.scale * scale_factor
        
        # Create image
        img = Image.new('RGB', (self.width, self.height), self.COLORS['background'])
        draw = ImageDraw.Draw(img)
        
        # Load font
        try:
            font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 
                                          int(24 * scale_factor))
            font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                                           int(18 * scale_factor))
            font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                                          int(14 * scale_factor))
            font_tiny = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 
                                         int(12 * scale_factor))
        except:
            # Fallback to default font
            font_large = ImageFont.load_default()
            font_medium = ImageFont.load_default()
            font_small = ImageFont.load_default()
            font_tiny = ImageFont.load_default()
        
        # Calculate offset to center layout
        offset_x = (self.width - layout_width * final_scale) / 2
        offset_y = (self.height - layout_height * final_scale) / 2
        
        def to_img_coords(x: float, y: float) -> Tuple[int, int]:
            """Convert floor plan coordinates to image coordinates"""
            img_x = int((x - min_x) * final_scale + offset_x)
            img_y = int((y - min_y) * final_scale + offset_y)
            return (img_x, img_y)
        
        # Draw grid if requested
        if show_grid:
            self._draw_grid(draw, min_x, max_x, min_y, max_y, final_scale, offset_x, offset_y)
        
        # Draw room fills first (background)
        for room in generator.rooms:
            bl = to_img_coords(room.bottom_left.x, room.bottom_left.y)
            tr = to_img_coords(room.top_right.x, room.top_right.y)
            
            # Ensure coordinates are in correct order (top-left, bottom-right)
            x1, y1 = min(bl[0], tr[0]), min(bl[1], tr[1])
            x2, y2 = max(bl[0], tr[0]), max(bl[1], tr[1])
            
            draw.rectangle([(x1, y1), (x2, y2)], fill=self.COLORS['room_fill'], outline=None)
        
        # Draw walls with thickness
        for wall in generator.walls:
            self._draw_wall(draw, wall, to_img_coords, final_scale)
        
        # Draw doors and windows
        for room in generator.rooms:
            for door in room.doors:
                self._draw_door(draw, door, to_img_coords, final_scale)
            for window in room.windows:
                self._draw_window(draw, window, to_img_coords, final_scale)
        
        # Draw room labels and areas
        for room in generator.rooms:
            center = to_img_coords(room.center.x, room.center.y)
            
            # Room name
            label_text = room.label
            bbox = draw.textbbox((0, 0), label_text, font=font_medium)
            text_width = bbox[2] - bbox[0]
            draw.text(
                (center[0] - text_width//2, center[1] - 30),
                label_text,
                fill=self.COLORS['room_label'],
                font=font_medium
            )
            
            if show_room_areas:
                # Room dimensions
                dim_text = f"{room.width:.1f}' × {room.height:.1f}'"
                bbox = draw.textbbox((0, 0), dim_text, font=font_small)
                text_width = bbox[2] - bbox[0]
                draw.text(
                    (center[0] - text_width//2, center[1]),
                    dim_text,
                    fill=self.COLORS['room_label'],
                    font=font_small
                )
                
                # Room area
                area_text = f"{room.area:.0f} sq ft"
                bbox = draw.textbbox((0, 0), area_text, font=font_tiny)
                text_width = bbox[2] - bbox[0]
                draw.text(
                    (center[0] - text_width//2, center[1] + 25),
                    area_text,
                    fill=(100, 100, 100),
                    font=font_tiny
                )
        
        # Draw dimension lines if requested
        if show_dimensions:
            for room in generator.rooms:
                self._draw_room_dimensions(
                    draw, room, to_img_coords, final_scale, font_tiny
                )
        
        # Draw title block if requested
        if show_title:
            self._draw_title_block(draw, generator, font_large, font_small)
        
        # Save image
        img.save(output_path, dpi=(self.dpi, self.dpi))
        print(f"✓ PNG rendered: {output_path}")
        print(f"  Resolution: {self.width}x{self.height} pixels")
        print(f"  DPI: {self.dpi}")
        print(f"  Scale: {final_scale:.1f} pixels per foot")
    
    def _draw_grid(
        self, 
        draw, 
        min_x: float, 
        max_x: float, 
        min_y: float, 
        max_y: float,
        scale: float,
        offset_x: float,
        offset_y: float
    ):
        """Draw background grid"""
        
        grid_spacing = 5  # feet
        
        # Vertical lines
        x = int(min_x / grid_spacing) * grid_spacing
        while x <= max_x:
            x1 = int((x - min_x) * scale + offset_x)
            y1 = int(offset_y)
            y2 = int((max_y - min_y) * scale + offset_y)
            draw.line([(x1, y1), (x1, y2)], fill=self.COLORS['grid'], width=1)
            x += grid_spacing
        
        # Horizontal lines
        y = int(min_y / grid_spacing) * grid_spacing
        while y <= max_y:
            y1 = int((y - min_y) * scale + offset_y)
            x1 = int(offset_x)
            x2 = int((max_x - min_x) * scale + offset_x)
            draw.line([(x1, y1), (x2, y1)], fill=self.COLORS['grid'], width=1)
            y += grid_spacing
    
    def _draw_wall(self, draw, wall: Wall, to_img_coords, scale: float):
        """Draw wall with thickness"""
        
        start = to_img_coords(wall.start.x, wall.start.y)
        end = to_img_coords(wall.end.x, wall.end.y)
        
        # Calculate perpendicular offset for thickness
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = np.sqrt(dx*dx + dy*dy)
        
        if length > 0:
            perp_x = -dy / length
            perp_y = dx / length
            
            offset = wall.thickness * scale / 2
            
            # Four corners of wall
            p1 = (int(start[0] - perp_x * offset), int(start[1] - perp_y * offset))
            p2 = (int(end[0] - perp_x * offset), int(end[1] - perp_y * offset))
            p3 = (int(end[0] + perp_x * offset), int(end[1] + perp_y * offset))
            p4 = (int(start[0] + perp_x * offset), int(start[1] + perp_y * offset))
            
            # Draw filled polygon for wall
            draw.polygon([p1, p2, p3, p4], fill=self.COLORS['wall_fill'], outline=self.COLORS['wall'])
    
    def _draw_door(self, draw, door: Door, to_img_coords, scale: float):
        """Draw door symbol"""
        
        pos = to_img_coords(door.position.x, door.position.y)
        width = door.width * scale
        angle_rad = np.radians(door.swing_angle)
        
        # Door arc
        bbox = [
            pos[0] - width, pos[1] - width,
            pos[0] + width, pos[1] + width
        ]
        draw.arc(bbox, start=0, end=door.swing_angle, fill=self.COLORS['door'], width=self.LINE_WIDTHS['door'])
        
        # Door line
        end_x = int(pos[0] + width * np.cos(angle_rad))
        end_y = int(pos[1] + width * np.sin(angle_rad))
        draw.line([pos, (end_x, end_y)], fill=self.COLORS['door'], width=self.LINE_WIDTHS['door'])
    
    def _draw_window(self, draw, window: Window, to_img_coords, scale: float):
        """Draw window symbol"""
        
        pos = to_img_coords(window.position.x, window.position.y)
        w = window.width * scale / 2
        h = 3  # Window thickness in pixels
        
        # Window rectangle
        draw.rectangle(
            [pos[0] - w, pos[1] - h, pos[0] + w, pos[1] + h],
            fill=self.COLORS['window'],
            outline=self.COLORS['window']
        )
        
        # Window panes (cross lines)
        draw.line([(pos[0] - w, pos[1]), (pos[0] + w, pos[1])], 
                 fill=(255, 255, 255), width=1)
    
    def _draw_room_dimensions(self, draw, room: Room, to_img_coords, scale: float, font):
        """Draw dimension lines for room"""
        
        bl = to_img_coords(room.bottom_left.x, room.bottom_left.y)
        tr = to_img_coords(room.top_right.x, room.top_right.y)
        
        # Ensure correct ordering
        x1, y1 = min(bl[0], tr[0]), min(bl[1], tr[1])
        x2, y2 = max(bl[0], tr[0]), max(bl[1], tr[1])
        
        offset = 30  # pixels
        
        # Bottom dimension (width)
        y_pos = y2 + offset
        draw.line([(x1, y_pos), (x2, y_pos)], 
                 fill=self.COLORS['dimension_line'], width=self.LINE_WIDTHS['dimension'])
        
        # Extension lines
        draw.line([(x1, y2), (x1, y_pos + 5)], 
                 fill=self.COLORS['dimension_line'], width=1)
        draw.line([(x2, y2), (x2, y_pos + 5)], 
                 fill=self.COLORS['dimension_line'], width=1)
        
        # Dimension text
        dim_text = f"{room.width:.1f}'"
        bbox = draw.textbbox((0, 0), dim_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_x = (x1 + x2) // 2 - text_width // 2
        draw.text((text_x, y_pos + 5), dim_text, fill=self.COLORS['dimension_line'], font=font)
        
        # Left dimension (height)
        x_pos = x1 - offset
        draw.line([(x_pos, y1), (x_pos, y2)], 
                 fill=self.COLORS['dimension_line'], width=self.LINE_WIDTHS['dimension'])
        
        # Extension lines
        draw.line([(x1, y1), (x_pos - 5, y1)], 
                 fill=self.COLORS['dimension_line'], width=1)
        draw.line([(x1, y2), (x_pos - 5, y2)], 
                 fill=self.COLORS['dimension_line'], width=1)
        
        # Dimension text (rotated)
        dim_text = f"{room.height:.1f}'"
        bbox = draw.textbbox((0, 0), dim_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_y = (y1 + y2) // 2 - text_width // 2
        
        # Draw text vertically
        from PIL import Image as PILImage
        text_img = PILImage.new('RGBA', (100, 20), (255, 255, 255, 0))
        text_draw = ImageDraw.Draw(text_img)
        text_draw.text((0, 0), dim_text, fill=self.COLORS['dimension_line'], font=font)
        text_img = text_img.rotate(90, expand=True)
        
        # Paste rotated text
        try:
            draw._image.paste(text_img, (x_pos - 25, text_y), text_img)
        except:
            # Fallback: draw text horizontally if rotation fails
            draw.text((x_pos - 25, (y1 + y2) // 2), dim_text, fill=self.COLORS['dimension_line'], font=font)
    
    def _draw_title_block(self, draw, generator: GeometricFloorPlanGenerator, font_large, font_small):
        """Draw title block"""
        
        # Calculate total area
        total_area = sum(room.area for room in generator.rooms)
        
        # Title
        title = "ARCHITECTURAL FLOOR PLAN"
        bbox = draw.textbbox((0, 0), title, font=font_large)
        text_width = bbox[2] - bbox[0]
        draw.text((self.width // 2 - text_width // 2, 30), title, fill=(0, 0, 0), font=font_large)
        
        # Subtitle
        subtitle = f"Total Area: {total_area:.0f} sq ft  |  Rooms: {len(generator.rooms)}  |  All dimensions in feet"
        bbox = draw.textbbox((0, 0), subtitle, font=font_small)
        text_width = bbox[2] - bbox[0]
        draw.text((self.width // 2 - text_width // 2, 65), subtitle, fill=(100, 100, 100), font=font_small)
        
        # Bottom note
        note = "Geometrically accurate floor plan with precise measurements"
        bbox = draw.textbbox((0, 0), note, font=font_small)
        text_width = bbox[2] - bbox[0]
        draw.text((self.width // 2 - text_width // 2, self.height - 40), note, fill=(100, 100, 100), font=font_small)


# Integrate with existing generator
class EnhancedGeometricFloorPlanGenerator(GeometricFloorPlanGenerator):
    """
    Enhanced generator that includes PNG rendering
    """
    
    def export_to_png(
        self,
        output_path: str,
        width: int = 2400,
        height: int = 2400,
        dpi: int = 300,
        show_dimensions: bool = True,
        show_grid: bool = True,
        show_room_areas: bool = True
    ):
        """
        Export floor plan to PNG image with measurements
        
        Args:
            output_path: Output PNG file path
            width: Image width in pixels
            height: Image height in pixels
            dpi: DPI for print quality
            show_dimensions: Show dimension lines
            show_grid: Show background grid
            show_room_areas: Show room area labels
        """
        
        renderer = PNGFloorPlanRenderer(width=width, height=height, dpi=dpi)
        renderer.render(
            self,
            output_path,
            show_dimensions=show_dimensions,
            show_grid=show_grid,
            show_room_areas=show_room_areas
        )


# Example usage
if __name__ == "__main__":
    
    # Example layout
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
    
    # Create generator
    generator = EnhancedGeometricFloorPlanGenerator()
    generator.parse_layout_specification(layout_spec)
    
    # Export to all formats
    generator.export_to_png("floor_plan.png", show_dimensions=True, show_grid=True)
    generator.export_to_dxf("floor_plan.dxf")
    generator.export_to_svg("floor_plan.svg")
    
    print("\nAll formats generated successfully!")
    print("- floor_plan.png (with measurements)")
    print("- floor_plan.dxf (for AutoCAD)")
    print("- floor_plan.svg (for web)")