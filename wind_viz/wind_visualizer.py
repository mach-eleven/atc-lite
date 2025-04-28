#!/usr/bin/env python
"""
Wind Visualizer - Displays wind patterns across the airspace for a given scenario.

Usage:
    python wind_visualizer.py --scenario <scenario_name> --badness <0-10>

Example:
    python wind_visualizer.py --scenario LOWW --badness 7
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import colorsys

from envs.atc.scenarios import LOWW, SuperSimple, SupaSupa
from envs.atc import model

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Visualize wind patterns across an airspace.')
    parser.add_argument('--scenario', type=str, default='LOWW', 
                      choices=['LOWW', 'SuperSimple', 'SupaSupa'],
                      help='Scenario to visualize')
    parser.add_argument('--badness', type=int, default=5, choices=range(0, 11),
                      help='Wind badness level (0-10)')
    parser.add_argument('--resolution', type=float, default=0.5,
                      help='Sampling resolution in nautical miles')
    parser.add_argument('--altitude', type=float, default=6000,
                      help='Aircraft altitude in feet for wind calculation')
    return parser.parse_args()

def get_scenario(name):
    """Get the specified scenario instance."""
    if name == 'LOWW':
        return LOWW()
    elif name == 'SuperSimple':
        return SuperSimple()
    elif name == 'SupaSupa':
        return SupaSupa()
    else:
        raise ValueError(f"Unknown scenario: {name}")

def sample_winds(scenario, badness, altitude, resolution):
    """Sample wind vectors across the airspace."""
    airspace = scenario.airspace
    bbox = airspace.get_bounding_box()
    
    # Define grid
    x_min, y_min, x_max, y_max = bbox
    x = np.arange(x_min, x_max, resolution)
    y = np.arange(y_min, y_max, resolution)
    X, Y = np.meshgrid(x, y)
    
    # Initialize wind component arrays
    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    wind_speed = np.zeros_like(X)
    mva_types = np.zeros_like(X, dtype=int)
    
    # Sample wind vectors
    for i in range(len(y)):
        for j in range(len(x)):
            try:
                mva = airspace.find_mva(x[j], y[i])
                wind_x, wind_y = model.get_wind_speed(x[j], y[i], altitude, mva.mva_type, badness)
                U[i, j] = wind_x
                V[i, j] = wind_y
                wind_speed[i, j] = np.sqrt(wind_x**2 + wind_y**2)
                
                # Convert MVA type to integer for colormapping
                if mva.mva_type == model.MvaType.GENERIC:
                    mva_types[i, j] = 0
                elif mva.mva_type == model.MvaType.MOUNTAINOUS:
                    mva_types[i, j] = 1
                elif mva.mva_type == model.MvaType.OCEANIC:
                    mva_types[i, j] = 2
                elif mva.mva_type == model.MvaType.WEATHER:
                    mva_types[i, j] = 3
            except ValueError:
                # Point outside airspace
                pass
    
    return X, Y, U, V, wind_speed, mva_types

def render_mva_polygons(ax, scenario):
    """Render MVA polygons with transparent colors based on type."""
    patches_list = []
    colors = []
    
    # Type-based colors with transparency
    type_colors = {
        model.MvaType.GENERIC: (0.2, 0.2, 0.2, 0.1),     # Gray
        model.MvaType.MOUNTAINOUS: (0.6, 0.3, 0.0, 0.1), # Brown
        model.MvaType.OCEANIC: (0.0, 0.3, 0.8, 0.1),     # Blue
        model.MvaType.WEATHER: (0.7, 0.0, 0.7, 0.1)      # Purple
    }
    
    for mva in scenario.airspace.mvas:
        # Extract polygon coordinates
        coords = list(mva.area.exterior.coords)
        polygon = patches.Polygon(coords, closed=True)
        patches_list.append(polygon)
        colors.append(type_colors.get(mva.mva_type, (0.2, 0.2, 0.2, 0.1)))
    
    # Add the polygon collection to the plot
    p = PatchCollection(patches_list, alpha=0.4, facecolors=colors, edgecolors='gray', linewidths=0.5)
    ax.add_collection(p)
    
    # Add MVA type labels at the center of each polygon
    for mva in scenario.airspace.mvas:
        centroid = mva.area.centroid
        ax.text(centroid.x, centroid.y, mva.mva_type.value,
                fontsize=8, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.3'))

def visualize_wind(scenario, badness, altitude, resolution):
    """Create a visualization of wind patterns."""
    # Sample wind data
    X, Y, U, V, wind_speed, mva_types = sample_winds(scenario, badness, altitude, resolution)
    
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))
    
    # First subplot: Wind vectors colored by speed
    render_mva_polygons(ax1, scenario)
    
    # Plot the runway
    runway_x, runway_y = scenario.runway.x, scenario.runway.y
    phi_rad = np.radians(scenario.runway.phi_from_runway)
    dx, dy = 1.5 * np.sin(phi_rad), 1.5 * np.cos(phi_rad)
    ax1.plot([runway_x - dx, runway_x + dx], [runway_y - dy, runway_y + dy], 'k-', linewidth=3)
    
    # Plot wind vectors
    skip = max(1, int(5 / resolution))  # Skip some arrows for clarity
    q = ax1.quiver(X[::skip, ::skip], Y[::skip, ::skip], 
                  U[::skip, ::skip], V[::skip, ::skip],
                  wind_speed[::skip, ::skip],
                  cmap='ocean', scale=50, width=0.002)
    
    # Add colorbar for wind speed
    cbar1 = fig.colorbar(q, ax=ax1)
    cbar1.set_label('Wind Speed (knots)')
    
    # Add scenario info
    ax1.set_title(f'Wind Vectors - {scenario.__class__.__name__} - Badness: {badness} - Altitude: {altitude}ft')
    ax1.set_xlabel('X (nautical miles)')
    ax1.set_ylabel('Y (nautical miles)')
    ax1.grid(alpha=0.3)
    
    # Second subplot: Streamplot visualization
    render_mva_polygons(ax2, scenario)
    
    # Plot the runway
    ax2.plot([runway_x - dx, runway_x + dx], [runway_y - dy, runway_y + dy], 'k-', linewidth=3)
    
    # Create streamplot
    strm = ax2.streamplot(X, Y, U, V, color=wind_speed, cmap='ocean', linewidth=1.5, density=1.5, arrowsize=1.2)
    
    # Add colorbar for wind speed
    cbar2 = fig.colorbar(strm.lines, ax=ax2)
    cbar2.set_label('Wind Speed (knots)')
    
    # Add legend for MVA types
    mva_types_legend = {
        'Generic': (0.2, 0.2, 0.2),
        'Mountainous': (0.6, 0.3, 0.0),
        'Oceanic': (0.0, 0.3, 0.8),
        'Weather': (0.7, 0.0, 0.7)
    }
    
    legend_elements = [patches.Patch(facecolor=color + (0.6,), edgecolor='gray',
                                     label=name)
                       for name, color in mva_types_legend.items()]
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Add streamplot title
    ax2.set_title(f'Wind Flow - {scenario.__class__.__name__} - Badness: {badness} - Altitude: {altitude}ft')
    ax2.set_xlabel('X (nautical miles)')
    ax2.set_ylabel('Y (nautical miles)')
    ax2.grid(alpha=0.3)
    
    # Make axes proportional and set limits to match the airspace
    ax1.set_aspect('equal')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'wind_visualization_{scenario.__class__.__name__}_badness{badness}.png', dpi=300)
    plt.show()

def main():
    """Main function."""
    args = parse_arguments()
    
    # Get the scenario
    scenario = get_scenario(args.scenario)
    
    # Visualize wind patterns
    visualize_wind(scenario, args.badness, args.altitude, args.resolution)

if __name__ == "__main__":
    main()