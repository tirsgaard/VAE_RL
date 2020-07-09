__all__ = ['get_vislogger']

from .spaces_vis import SpacesVis
def get_vislogger(cfg):
    if cfg.model == 'SPACES':
        return SpacesVis()
    return None
