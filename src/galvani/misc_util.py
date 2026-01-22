from tueplots.constants.color import palettes
import json
import os


def get_next_tue_plot_color(idx, mod=1.0):
    """continuous tue_plot color selector"""
    try:
        return palettes.tue_plot[idx]*mod
    except IndexError:
        return get_next_tue_plot_color(idx-len(palettes.tue_plot), mod*1.2)


def get_config(path='cfg/config.json'):
    config = None
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    assert config != None
    return config


def save_config(config, path):
    with open(path, "w", encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)



def add_subfolder_to_save_prefix(inf_config, subfolder):
    _path_parts = inf_config["output_prefix"].split(os.sep)
    [_path_parts.append(f) for f in subfolder.split(os.sep)]
    new_save_prefix = os.sep.join(_path_parts)
    os.makedirs(new_save_prefix, exist_ok=True)
    return f"{new_save_prefix}{os.sep}"