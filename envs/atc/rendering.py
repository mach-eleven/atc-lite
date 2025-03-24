# minimal-atc-rl/envs/atc/rendering.py
import pyglet  # Pyglet is a cross-platform windowing and multimedia library for Python

# Import the base Geom class from the custom rendering module
from .my_rendering import Geom, FilledPolygon
# Import color definitions from the themes module
from .themes import ColorScheme


class Label(Geom):
    """
    A specialized geometry class for rendering text labels in the simulation.
    
    This class extends the base Geom class to provide text rendering capabilities.
    It's used for displaying information like aircraft callsigns, altitude, and speed
    in the simulation visualization.
    """
    def __init__(self, text, x, y, bold=True):
        """
        Initialize a new text label.
        
        :param text: The text content to display
        :param x: X-coordinate of the label position
        :param y: Y-coordinate of the label position
        :param bold: Whether to render the text in bold (default: True, but not used due to compatibility)
        """
        # Initialize the parent Geom class
        super().__init__()
        # Store the text content
        self.text = text
        # Store the position coordinates
        self.x = x
        self.y = y
        # Store the bold setting (not used in rendering due to compatibility)
        self.bold = bold

    def _render(self):
        """
        Render the text label using Pyglet's text rendering capabilities.
        
        This method is called by the parent class's render() method during the
        rendering cycle. It creates and draws a Pyglet text label with the
        specified properties.
        """
        # Create a Pyglet text label with the configured properties
        # Not using the bold parameter since it's not compatible with this version
        label = pyglet.text.Label(
            self.text,                  # The text to display
            font_name='Arial',          # Font family
            font_size=16,               # Fixed larger font size
            x=self.x, y=self.y,         # Position coordinates
            anchor_x="left",            # Horizontal anchor at left side
            anchor_y="top",             # Vertical anchor at top side
            color=(255, 255, 255, 255)  # Bright white text for visibility
        )
        # Draw the label to the current OpenGL context
        label.draw()
        
        
class FuelGauge(Geom):
    """
    A simplified fuel gauge for maximum compatibility.
    """
    def __init__(self, x, y, width, height, fuel_percentage):
        """Initialize a new fuel gauge."""
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.fuel_percentage = max(0, min(100, fuel_percentage))
        
    def _render(self):
        """Render a simple fuel gauge with filled rectangles only."""
        # Background rectangle (black)
        bg_rect = pyglet.shapes.Rectangle(
            x=self.x, 
            y=self.y, 
            width=self.width, 
            height=self.height, 
            color=(0, 0, 0)
        )
        bg_rect.draw()
        
        # Foreground filled portion based on fuel percentage
        if self.fuel_percentage > 0:
            fill_width = (self.fuel_percentage / 100) * self.width
            
            # Color based on fuel level
            if self.fuel_percentage > 66:
                color = (0, 255, 0)  # Green
            elif self.fuel_percentage > 33:
                color = (255, 255, 0)  # Yellow
            else:
                color = (255, 0, 0)  # Red
            
            fg_rect = pyglet.shapes.Rectangle(
                x=self.x, 
                y=self.y, 
                width=fill_width, 
                height=self.height, 
                color=color
            )
            fg_rect.draw()