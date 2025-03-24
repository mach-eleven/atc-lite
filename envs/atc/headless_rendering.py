# Create a new file: envs/atc/headless_rendering.py

"""
Headless rendering implementation for ATC simulator.
This module provides a no-graphics alternative to the regular rendering
system for training reinforcement learning models.
"""

class HeadlessViewer:
    """
    A headless implementation of the Viewer class that performs no actual rendering.
    Used for training reinforcement learning models without visual output.
    """
    def __init__(self, width, height, display=None):
        """
        Initialize a headless viewer with dummy dimensions.
        
        Args:
            width: Dummy window width (not used)
            height: Dummy window height (not used)
            display: Dummy display parameter (not used)
        """
        self.width = width
        self.height = height
        self.isopen = True
        self.geoms = []
        self.onetime_geoms = []
    
    def close(self):
        """Close the viewer (simply sets isopen to False)."""
        self.isopen = False
    
    def set_bounds(self, left, right, bottom, top):
        """Dummy method to maintain API compatibility."""
        pass
    
    def add_geom(self, geom):
        """Add a geometry to the list (no-op in headless mode)."""
        self.geoms.append(geom)
    
    def add_onetime(self, geom):
        """Add a one-time geometry to the list (no-op in headless mode)."""
        self.onetime_geoms.append(geom)
    
    def render(self, return_rgb_array=False):
        """
        Dummy render method that maintains API compatibility.
        
        Args:
            return_rgb_array: If True, returns a blank array
            
        Returns:
            Empty numpy array if return_rgb_array is True, else isopen status
        """
        # Clear onetime geometries list
        self.onetime_geoms = []
        
        if return_rgb_array:
            import numpy as np
            # Return a blank array of appropriate shape
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        return self.isopen
    
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        """Dummy circle drawing method."""
        pass
    
    def draw_polygon(self, v, filled=True, **attrs):
        """Dummy polygon drawing method."""
        pass
    
    def draw_polyline(self, v, **attrs):
        """Dummy polyline drawing method."""
        pass
    
    def draw_line(self, start, end, **attrs):
        """Dummy line drawing method."""
        pass
    
    def get_array(self):
        """
        Return a dummy array for API compatibility.
        
        Returns:
            Empty numpy array of appropriate shape
        """
        import numpy as np
        return np.zeros((self.height, self.width, 3), dtype=np.uint8)
    
    def __del__(self):
        """Destructor that calls close()."""
        self.close()


# Dummy equivalents for geometry classes
class HeadlessGeom:
    """Base class for headless geometry implementations."""
    def __init__(self):
        self.attrs = []
    
    def render(self):
        """Dummy render method."""
        pass
    
    def add_attr(self, attr):
        """Dummy attribute addition."""
        pass
    
    def set_color(self, r, g, b):
        """Dummy color setting."""
        pass
        
    def set_color_opacity(self, r, g, b, a):
        """Dummy color with opacity setting."""
        pass


class HeadlessFilledPolygon(HeadlessGeom):
    """Headless implementation of FilledPolygon."""
    def __init__(self, v):
        super().__init__()


class HeadlessPolyLine(HeadlessGeom):
    """Headless implementation of PolyLine."""
    def __init__(self, v, close, linewidth=1):
        super().__init__()
        self.v = v
        self.close = close
        self.linewidth = linewidth
    
    def set_linewidth(self, x):
        """Dummy linewidth setting."""
        self.linewidth = x


class HeadlessLine(HeadlessGeom):
    """Headless implementation of Line."""
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), attrs=None):
        super().__init__()
        self.start = start
        self.end = end
        self.linewidth = 1


# Dummy factory functions
def make_circle(radius=10, res=30, filled=True):
    """Create a dummy circle."""
    return HeadlessFilledPolygon([])


def make_polygon(v, filled=True):
    """Create a dummy polygon."""
    return HeadlessFilledPolygon(v)


def make_polyline(v):
    """Create a dummy polyline."""
    return HeadlessPolyLine(v, False)


# Dummy label and fuel gauge classes
class Label(HeadlessGeom):
    """Headless implementation of Label."""
    def __init__(self, text, x, y, bold=True):
        super().__init__()
        self.text = text
        self.x = x
        self.y = y


class FuelGauge(HeadlessGeom):
    """Headless implementation of FuelGauge."""
    def __init__(self, x, y, width, height, fuel_percentage):
        super().__init__()
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.fuel_percentage = fuel_percentage