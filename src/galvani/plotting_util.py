from tueplots.constants.color import palettes

def get_next_tue_plot_color(idx, mod=1.0):
    """continuous tue_plot color selector"""
    try:
        return palettes.tue_plot[idx]*mod
    except IndexError:
        return get_next_tue_plot_color(idx-len(palettes.tue_plot), mod*1.2)