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


class FuelGauge(Geom):
    """
    A specialized geometry class for rendering a fuel gauge in the simulation.
    
    This class extends the base Geom class to provide a visual representation of
    aircraft fuel levels. It displays a color-coded bar that changes from green
    to yellow to red as fuel decreases.
    """
    def __init__(self, x, y, width, height, fuel_percentage, label_text):
        """
        Initialize a new fuel gauge.
        
        :param x: X-coordinate of the gauge position
        :param y: Y-coordinate of the gauge position
        :param width: Width of the gauge in pixels
        :param height: Height of the gauge in pixels
        :param fuel_percentage: Fuel level as a percentage (0-100)
        :param label_text: Text label for the gauge
        """
        # Initialize the parent Geom class
        super().__init__()
        # Store position and dimensions
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        # Store the fuel percentage
        self.fuel_percentage = max(0, min(100, fuel_percentage))
        # Store the label text
        self.label_text = label_text
        
    def _render(self):
        """
        Render the fuel gauge using Pyglet's shape rendering capabilities.
        
        This method is called by the parent class's render() method during the
        rendering cycle. It draws a background bar, a foreground bar representing
        the fuel level, and a text label.
        """
        # Draw background (empty gauge)
        background = pyglet.shapes.Rectangle(
            x=self.x, 
            y=self.y, 
            width=self.width, 
            height=self.height, 
            color=(50, 50, 50)
        )
        background.draw()
        
        # Draw foreground (filled portion of gauge)
        if self.fuel_percentage > 0:
            # Calculate fill width based on percentage
            fill_width = (self.fuel_percentage / 100) * self.width
            
            # Determine color based on fuel level
            if self.fuel_percentage > 66:
                # Green for high fuel
                color = (0, 200, 0)
            elif self.fuel_percentage > 33:
                # Yellow for medium fuel
                color = (240, 240, 0)
            else:
                # Red for low fuel
                color = (200, 0, 0)
            
            foreground = pyglet.shapes.Rectangle(
                x=self.x, 
                y=self.y, 
                width=fill_width, 
                height=self.height, 
                color=color
            )
            foreground.draw()
        
        # Add border around gauge
        border = pyglet.shapes.Rectangle(
            x=self.x, 
            y=self.y, 
            width=self.width, 
            height=self.height, 
            color=(255, 255, 255)
        )
        border.opacity = 128
        border.draw()
        
        # Create label
        label = pyglet.text.Label(
            f"{self.label_text}: {self.fuel_percentage:.1f}%",
            font_name='Arial',
            font_size=10,
            weight='normal',
            x=self.x, 
            y=self.y - 15,
            anchor_x="left",
            anchor_y="top",
            color=ColorScheme.label
        )
        label.draw()