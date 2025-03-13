"""
2D rendering framework

This module provides a custom rendering framework for the ATC simulation.
"""
import os
import sys

from gymnasium import error  # Import error handling from Gymnasium (OpenAI Gym successor)

# Try to import pyglet, which is required for rendering
try:
    import pyglet
except ImportError as e:
    raise ImportError(
        """
    Cannot import pyglet.
    HINT: you can install pyglet directly via 'pip install pyglet'.
    But if you really just want to install all Gym dependencies and not have to think about it,
    'pip install -e .[all]' or 'pip install gym[all]' will do it.
    """
    )

import math
import numpy as np

# Constant to convert radians to degrees
RAD2DEG = 57.29577951308232  # Value of 180/pi


def get_display(spec):
    """
    Convert a display specification (such as :0) into an actual Display object.
    
    Pyglet only supports multiple Displays on Linux.
    
    :param spec: Display specification (None or string like ":0")
    :return: Pyglet Display object
    """
    if spec is None:
        return pyglet.display.get_display()
        # returns already available pyglet_display,
        # if there is no pyglet display available then it creates one
    elif isinstance(spec, str):
        return pyglet.display.Display(spec)
    else:
        raise error.Error(
            "Invalid display specification: {}. (Must be a string like :0 or None.)".format(
                spec
            )
        )


def get_window(width, height, display, **kwargs):
    """
    Create a pyglet window from the display specification provided.
    
    :param width: Window width in pixels
    :param height: Window height in pixels
    :param display: Pyglet Display object
    :param kwargs: Additional arguments to pass to Window constructor
    :return: Configured Pyglet Window object
    """
    screen = display.get_screens()  # Get available screens
    config = screen[0].get_best_config()  # Select the first screen's best configuration
    context = config.create_context(None)  # Create graphics context

    # Create and return the window with specified parameters
    return pyglet.window.Window(
        width=width,
        height=height,
        display=display,
        config=config,
        context=context,
        **kwargs
    )


class Viewer(object):
    """
    Main rendering window that manages the display and rendering cycle.
    Handles creation of rendering window, maintaining geometry lists,
    and converting rendered frames to array format for RL algorithms.
    """
    def __init__(self, width, height, display=None):
        """
        Initialize the viewer with specified dimensions.
        
        :param width: Window width in pixels
        :param height: Window height in pixels
        :param display: Display specification (optional)
        """
        # Get display and create window
        display = get_display(display)

        # Store window dimensions
        self.width = width
        self.height = height
        
        # Create the window
        self.window = get_window(width=width, height=height, display=display)
        
        # Set callback for window close event
        self.window.on_close = self.window_closed_by_user
        
        # Track if window is open
        self.isopen = True
        
        # List for persistent geometries (shown until explicitly removed)
        self.geoms = []
        
        # List for one-time geometries (cleared after each render)
        self.onetime_geoms = []
        
        # Commented out: OpenGL transform that was used in the original version
        # self.transform = Transform()

        # Commented out: OpenGL blend functions, replaced with Pyglet's built-in functionality
        # glEnable(GL_BLEND)
        # glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

    def close(self):
        """
        Close the rendering window.
        Checks sys.meta_path to avoid error during Python shutdown.
        """
        if self.isopen and sys.meta_path:
            # Check sys.meta_path to avoid 'ImportError: sys.meta_path is None, Python is likely shutting down'
            self.window.close()
            self.isopen = False

    def window_closed_by_user(self):
        """
        Event handler when user closes the window.
        """
        self.isopen = False

    def set_bounds(self, left, right, bottom, top):
        """
        Set the bounds for world coordinates to screen coordinates mapping.
        
        :param left: Left world coordinate
        :param right: Right world coordinate
        :param bottom: Bottom world coordinate
        :param top: Top world coordinate
        """
        assert right > left and top > bottom
        
        # Calculate scale factors for world-to-screen conversion
        scalex = self.width / (right - left)
        scaley = self.height / (top - bottom)
        
        # Commented out: Original transform creation
        # self.transform = Transform(
        #     translation=(-left * scalex, -bottom * scaley), scale=(scalex, scaley)
        # )

    def add_geom(self, geom):
        """
        Add a geometry to the persistent list.
        
        :param geom: Geometry object to add
        """
        self.geoms.append(geom)

    def add_onetime(self, geom):
        """
        Add a geometry to the one-time list (rendered once then cleared).
        
        :param geom: Geometry object to add
        """
        self.onetime_geoms.append(geom)

    def render(self, return_rgb_array=False):
        """
        Render all geometries in the viewer and optionally return as an RGB array.
        
        :param return_rgb_array: If True, return the rendered scene as an RGB array
        :return: RGB array if requested, otherwise whether window is still open
        """
        # Clear the window
        self.window.clear()
        
        # Make this window the current context
        self.window.switch_to()
        
        # Process window events (mouse, keyboard, etc.)
        self.window.dispatch_events()
        
        # Commented out: OpenGL transform enable
        # self.transform.enable()
        
        # Render all persistent geometries
        for geom in self.geoms:
            geom.render()
            
        # Render all one-time geometries
        for geom in self.onetime_geoms:
            geom.render()
            
        # Commented out: OpenGL transform disable
        # self.transform.disable()
        
        # If requested, get rendered frame as RGB array
        arr = None
        if return_rgb_array:
            # Get the color buffer from Pyglet
            buffer = pyglet.image.get_buffer_manager().get_color_buffer()
            image_data = buffer.get_image_data()
            
            # Convert buffer data to numpy array
            arr = np.frombuffer(image_data.get_data(), dtype=np.uint8)
            
            # In https://github.com/openai/gym-http-api/issues/2, we
            # discovered that someone using Xmonad on Arch was having
            # a window of size 598 x 398, though a 600 x 400 window
            # was requested. (Guess Xmonad was preserving a pixel for
            # the boundary.) So we use the buffer height/width rather
            # than the requested one.
            arr = arr.reshape(buffer.height, buffer.width, 4)
            
            # Flip array vertically and remove alpha channel
            arr = arr[::-1, :, 0:3]
            
        # Swap buffers (display the rendered frame)
        self.window.flip()
        
        # Clear one-time geometries
        self.onetime_geoms = []
        
        # Return array if requested, otherwise return whether window is open
        return arr if return_rgb_array else self.isopen

    # Convenience methods to add common shapes
    def draw_circle(self, radius=10, res=30, filled=True, **attrs):
        """
        Convenience method to draw a circle.
        
        :param radius: Circle radius
        :param res: Resolution (number of segments)
        :param filled: Whether to fill the circle
        :param attrs: Additional attributes (color, linewidth, etc.)
        :return: The created geometry object
        """
        # TODO: Change - marked for future refactoring
        geom = make_circle(radius=radius, res=res, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polygon(self, v, filled=True, **attrs):
        """
        Convenience method to draw a polygon.
        
        :param v: List of vertices
        :param filled: Whether to fill the polygon
        :param attrs: Additional attributes (color, linewidth, etc.)
        :return: The created geometry object
        """
        # TODO: Change - marked for future refactoring
        geom = make_polygon(v=v, filled=filled)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_polyline(self, v, **attrs):
        """
        Convenience method to draw a polyline (connected line segments).
        
        :param v: List of vertices
        :param attrs: Additional attributes (color, linewidth, etc.)
        :return: The created geometry object
        """
        # TODO: Change - marked for future refactoring
        geom = make_polyline(v=v)
        _add_attrs(geom, attrs)
        self.add_onetime(geom)
        return geom

    def draw_line(self, start, end, **attrs):
        """
        Convenience method to draw a line.
        
        :param start: Start point (x, y)
        :param end: End point (x, y)
        :param attrs: Additional attributes (color, linewidth, etc.)
        :return: The created geometry object
        """
        # TODO: Change - marked for future refactoring
        geom = Line(start, end, attrs)
        self.add_onetime(geom)
        return geom

    def get_array(self):
        """
        Get the current display as an RGB array.
        
        :return: RGB array representation of the current display
        """
        # Swap buffers to render any pending changes
        self.window.flip()
        
        # Get image data from buffer
        image_data = (
            pyglet.image.get_buffer_manager().get_color_buffer().get_image_data()
        )
        
        # Swap buffers again to restore original display
        self.window.flip()
        
        # Convert image data to numpy array
        arr = np.fromstring(image_data.get_data(), dtype=np.uint8, sep="")
        arr = arr.reshape(self.height, self.width, 4)
        
        # Flip vertically and remove alpha channel
        return arr[::-1, :, 0:3]

    def __del__(self):
        """
        Destructor to ensure window is closed when the viewer is deleted.
        """
        self.close()


def _add_attrs(geom, attrs):
    """
    Add attributes to a geometry.
    
    :param geom: Geometry object
    :param attrs: Dictionary of attributes
    """
    # TODO: Remove - marked for future removal
    if "color" in attrs:
        geom.set_color(*attrs["color"])
    if "linewidth" in attrs:
        geom.set_linewidth(attrs["linewidth"])


class Geom(object):
    """
    Base class for all geometry objects.
    Handles color and common rendering functionality.
    """
    def __init__(self):
        """
        Initialize geometry with default black color.
        """
        self._color = Color((0, 0, 0, 255))  # Default to black with full opacity
        self.attrs = [self._color]  # List of attributes (currently just color)

    def render(self):
        """
        Render the geometry by calling the implementation-specific _render method.
        """
        self._render()

    def _render(self):
        """
        Implementation-specific rendering method (must be overridden by subclasses).
        """
        raise NotImplementedError

    def add_attr(self, attr):
        """
        Add an attribute to the geometry.
        
        :param attr: Attribute to add
        """
        # TODO: remove - marked for future removal
        self.attrs.append(attr)

    def set_color(self, r, g, b):
        """
        Set the RGB color of the geometry.
        
        :param r: Red component (0-255)
        :param g: Green component (0-255)
        :param b: Blue component (0-255)
        """
        self._color.vec4 = (r, g, b, 255)  # Set RGB with full opacity

    def set_color_opacity(self, r, g, b, a):
        """
        Set the RGBA color of the geometry.
        
        :param r: Red component (0-255)
        :param g: Green component (0-255)
        :param b: Blue component (0-255)
        :param a: Alpha component (0-255)
        """
        self._color.vec4 = (r, g, b, a)


class Attr(object):
    """
    Base class for geometry attributes (color, transform, etc.).
    """
    def enable(self):
        """
        Enable the attribute (must be overridden by subclasses).
        """
        raise NotImplementedError

    def disable(self):
        """
        Disable the attribute (default implementation does nothing).
        """
        pass


class Transform(Attr):
    """
    Transform attribute for geometries (translation, rotation, scale).
    """
    # TODO: Change - marked for future refactoring
    def __init__(self, translation=(0.0, 0.0), rotation=0.0, scale=(1, 1)):
        """
        Initialize transform with default identity values.
        
        :param translation: (x, y) translation
        :param rotation: Rotation angle in radians
        :param scale: (x, y) scale factors
        """
        self.set_translation(*translation)
        self.set_rotation(rotation)
        self.set_scale(*scale)

    def enable(self):
        """
        Enable this transform.
        
        Original OpenGL code is commented out, as this implementation
        now relies on Pyglet's built-in transformations.
        """
        # Original OpenGL code:
        # glPushMatrix()
        # glTranslatef(
        #     self.translation[0], self.translation[1], 0
        # )  # translate to GL loc ppint
        # glRotatef(RAD2DEG * self.rotation, 0, 0, 1.0)
        # glScalef(self.scale[0], self.scale[1], 1)
        pass

    def disable(self):
        """
        Disable this transform.
        
        Original OpenGL code is commented out, as this implementation
        now relies on Pyglet's built-in transformations.
        """
        # Original OpenGL code:
        # glPopMatrix()
        pass

    def set_translation(self, newx, newy):
        """
        Set the translation component.
        
        :param newx: New x translation
        :param newy: New y translation
        """
        self.translation = (float(newx), float(newy))

    def set_rotation(self, new):
        """
        Set the rotation component.
        
        :param new: New rotation angle in radians
        """
        self.rotation = float(new)

    def set_scale(self, newx, newy):
        """
        Set the scale component.
        
        :param newx: New x scale factor
        :param newy: New y scale factor
        """
        self.scale = (float(newx), float(newy))


class Color(Attr):
    """
    Color attribute for geometries.
    """
    def __init__(self, vec4):
        """
        Initialize with RGBA color.
        
        :param vec4: (r, g, b, a) color tuple
        """
        self.vec4 = vec4


class LineStyle(Attr):
    """
    Line style attribute for geometries (dashed, solid, etc.).
    
    Original OpenGL code is commented out, as this implementation
    now relies on Pyglet's built-in line styles.
    """
    def __init__(self, style):
        """
        Initialize with line style value.
        
        :param style: OpenGL line style bit pattern
        """
        self.style = style

    # Original OpenGL code:
    # def enable(self):
    #     glEnable(GL_LINE_STIPPLE)
    #     glLineStipple(1, self.style)

    # def disable(self):
    #     glDisable(GL_LINE_STIPPLE)


class Point(Geom):
    """
    Point geometry (rendered as a small circle).
    """
    def __init__(self):
        """
        Initialize point geometry.
        """
        Geom.__init__(self)

    def _render(self):
        """
        Render the point as a small circle.
        """
        point = pyglet.shapes.Circle(0, 0, 1, color=self._color.vec4)
        point.draw()


class FilledPolygon(Geom):
    """
    Filled polygon geometry.
    """
    def __init__(self, v):
        """
        Initialize with vertices.
        
        :param v: List of (x, y) vertices
        """
        Geom.__init__(self)
        self.v = v

    def _render(self):
        """
        Render the filled polygon using Pyglet shapes.
        """
        # Convert vertices to float to avoid possible type issues
        self.v = [(float(x), float(y)) for x, y in self.v]
        
        # Create and draw Pyglet polygon
        poly = pyglet.shapes.Polygon(*self.v, color=self._color.vec4)
        poly.draw()


def make_circle(radius=10, res=30, filled=True):
    """
    Factory function to create a circle geometry.
    
    :param radius: Circle radius
    :param res: Resolution (number of segments)
    :param filled: Whether to fill the circle
    :return: Circle geometry (FilledPolygon or PolyLine)
    """
    # Generate points around a circle
    points = []
    for i in range(res):
        ang = 2 * math.pi * i / res
        points.append((math.cos(ang) * radius, math.sin(ang) * radius))
        
    # Return filled or outline circle
    if filled:
        return FilledPolygon(points)
    else:
        return PolyLine(points, True)


def make_polygon(v, filled=True):
    """
    Factory function to create a polygon geometry.
    
    :param v: List of (x, y) vertices
    :param filled: Whether to fill the polygon
    :return: Polygon geometry (FilledPolygon or PolyLine)
    """
    if filled:
        return FilledPolygon(v)
    else:
        return PolyLine(v, True)


def make_polyline(v):
    """
    Factory function to create a polyline geometry.
    
    :param v: List of (x, y) vertices
    :return: PolyLine geometry
    """
    return PolyLine(v, False)


def make_capsule(length, width):
    """
    Factory function to create a capsule geometry (rectangle with semicircles at ends).
    
    :param length: Capsule length
    :param width: Capsule width
    :return: Compound geometry
    """
    l, r, t, b = 0, length, width / 2, -width / 2
    
    # Create the rectangular part
    box = make_polygon([(l, b), (l, t), (r, t), (r, b)])
    
    # Create the left semicircle
    circ0 = make_circle(width / 2)
    
    # Create the right semicircle and translate it to the right end
    circ1 = make_circle(width / 2)
    circ1.add_attr(Transform(translation=(length, 0)))
    
    # Combine shapes into a compound geometry
    geom = Compound([box, circ0, circ1])
    return geom


class Compound(Geom):
    """
    Compound geometry (group of geometries treated as one).
    """
    def __init__(self, gs):
        """
        Initialize with list of geometries.
        
        :param gs: List of geometry objects
        """
        Geom.__init__(self)
        self.gs = gs
        
        # Remove color attributes from child geometries to allow parent color to apply
        for g in self.gs:
            g.attrs = [a for a in g.attrs if not isinstance(a, Color)]

    def _render(self):
        """
        Render all child geometries.
        """
        for g in self.gs:
            g.render()


class PolyLine(Geom):
    """
    Polyline geometry (connected line segments).
    """
    def __init__(self, v, close, linewidth=1):
        """
        Initialize with vertices.
        
        :param v: List of (x, y) vertices
        :param close: Whether to close the path (connect last vertex to first)
        :param linewidth: Line width in pixels
        """
        Geom.__init__(self)
        self.v = v
        self.close = close
        self.linewidth = linewidth
        # self.linestyle = LineStyle(linestyle)  # Original line style support

    def _render(self):
        """
        Render the polyline using Pyglet shapes.
        """
        # Convert vertices to float to avoid possible type issues
        try:
            self.v = [(float(x), float(y)) for x, y in self.v]
        except TypeError as e:
            print(self.v)
            raise e
        
        # Create and draw Pyglet MultiLine
        ml = pyglet.shapes.MultiLine(*self.v, closed=self.close, color=self._color.vec4, thickness=self.linewidth)
        ml.draw()

    def set_linewidth(self, x):
        """
        Set the line width.
        
        :param x: New line width in pixels
        """
        self.linewidth = x


class Line(Geom):
    """
    Line geometry (single line segment).
    """
    def __init__(self, start=(0.0, 0.0), end=(0.0, 0.0), attrs=None):
        """
        Initialize with start and end points.
        
        :param start: Start point (x, y)
        :param end: End point (x, y)
        :param attrs: Additional attributes (color, linewidth, etc.)
        """
        Geom.__init__(self)
        self.start = start
        self.end = end
        self.linewidth = 1
        
        # Process attributes if provided
        if attrs is not None:
            if isinstance(attrs, dict):
                if "linewidth" in attrs:
                    self.linewidth = attrs["linewidth"]
                if "color" in attrs:
                    self.set_color(*attrs["color"])
            else:
                self.attrs = attrs

    def _render(self):
        """
        Render the line using Pyglet shapes.
        """
        # Create and draw Pyglet Line
        # Note: 'width' parameter was renamed to 'thickness' in newer Pyglet versions
        l = pyglet.shapes.Line(
            self.start[0], self.start[1], 
            self.end[0], self.end[1], 
            color=self._color.vec4, 
            thickness=self.linewidth  # Changed from 'width' to 'thickness'
        )
        l.draw()