from .spaces.spaces import Spaces

__all__ = ['get_model']

def get_model(cfg):
    """
    Also handles loading checkpoints, data parallel and so on
    :param cfg:
    :return:
    """
    
    model = None
    if cfg.model == 'SPACES':
        model = Spaces(cfg)
        
    return model
