from matplotlib import colormaps
# minimal-atc-rl/envs/atc/themes.py

# Hello, this file contains the color scheme for the ATC simulation visualization. We shall use sexy colors!!!!!

class ColorScheme:
    background_inactive = [29, 69, 76]
    background_active = [84, 121, 128]
    lines_info = [69, 173, 168]
    mva = lines_info
    runway = lines_info
    airplane = [157, 224, 173]
    label = (157, 224, 173, 255)
    wind = [255, 255, 255, 80]
    wind_arrow_base = [0, 255, 0, 80]
    mva_height_colormap = colormaps.get_cmap('viridis')
    # Generic MVA areas (cyan with 40% opacity)
    generic_mva_color = [0, 255, 255, 80]  # Cyan, moderate opacity
    # Mountainous areas (earthy red with 60% opacity)
    mountainous_mva_color = [180, 60, 30, 150]  # Reddish-brown, higher opacity for mountains
    # Weather zones (blue with 30% opacity)
    weather_mva_color = [30, 144, 255, 150]  # Dodger blue, lower opacity for weather
    # Oceanic areas (deep blue with 50% opacity)s
    oceanic_mva_color = [0, 105, 148, 150]  # Deep blue, medium opacity
