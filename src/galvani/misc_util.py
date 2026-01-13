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


def create_save_path_from_prefix(prefix):
    """
    given path/to/prefix_, creates path/to/ if it does not already exist 
    """
    save_path = "/".join(prefix.split("/")[:-1])
    os.makedirs(save_path, exist_ok=True)


def add_subfolder_to_save_prefix(args, subfolder):
    _file_prefix = args.output_prefix.split(os.sep)[-1]
    _path_parts = args.output_prefix.split(os.sep)[:-1]
    [_path_parts.append(f) for f in subfolder.split(os.sep)]
    _path_parts.append(_file_prefix)
    new_save_prefix = os.sep.join(_path_parts)
    create_save_path_from_prefix(new_save_prefix)
    return new_save_prefix