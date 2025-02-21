# minimal-atc-rl/envs/atc/rendering.py
import pyglet  # Pyglet is a cross-platform windowing and multimedia library for Python

# Import the base Geom class from the custom rendering module
from .my_rendering import Geom
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
        :param bold: Whether to render the text in bold (default: True)
        """
        # Initialize the parent Geom class
        super().__init__()
        # Store the text content
        self.text = text
        # Store the position coordinates
        self.x = x
        self.y = y
        # Store the bold setting
        self.bold = bold

    def _render(self):
        """
        Render the text label using Pyglet's text rendering capabilities.
        
        This method is called by the parent class's render() method during the
        rendering cycle. It creates and draws a Pyglet text label with the
        specified properties.
        """
        # Create a Pyglet text label with the configured properties
        label = pyglet.text.Label(
            self.text,                  # The text to display
            font_name='Arial',          # Font family
            font_size=12,               # Font size in points
            weight='bold' if self.bold else 'normal',  # Bold or normal weight
            x=self.x, y=self.y,         # Position coordinates
            anchor_x="left",            # Horizontal anchor at left side
            anchor_y="top",             # Vertical anchor at top side
            color=ColorScheme.label     # Text color from theme
        )
        # Draw the label to the current OpenGL context
        label.draw()