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
    wind = [255, 255, 255, 120]
    wind_arrow_base = [0, 255, 0, 120]
    mva_height_colormap = colormaps.get_cmap('viridis')