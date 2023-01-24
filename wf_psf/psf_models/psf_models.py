"""
:file: wf_psf/psf_models/psf_models.py

:date: 18/01/23
:author: jpollack

"""
from wf_psf.tf_mccd_psf_field import build_mccd_spatial_dic
import tensorflow as tf

PSF_CLASS = {}

class PsfModelError(Exception):
    pass

def register_psfclass(psf_class):
    for id in psf_class.ids :
        PSFCLASSES[id] = psf_class
    return psfclass


class TFPSFModel(tf.keras.Model):
    """
    A generic PSF model parent class 
    """
    pass

class DataDrivenPSFModel(tf.keras.Model):
    """
    A generic DataDriven model parent class to
    handle common parameters for the MCCD and graph 
    PSF models.

    - obs_stars
    - obs_pos
    - x_lims=args['x_lims'],
    - y_lims=args['y_lims'],
    - d_max=args['d_max_nonparam'],
    - graph_features=args['graph_features']
    """
    def __init__(self) -> None:
        self.obs_stars=outputs
        self.obs_pos=tf_train_pos
        self.x_lims=x_lims
        self.y_lims=y_lims
        self.d_max=d_max_nonparam
        self.graph_features=graph_features

    def get_spatial_dict(self):
        poly_dict, graph_dict = build_mccd_spatial_dic_v2(
            self.obs_stars,
            self.obs_pos,
            self.x_lims,
            self.y_lims,
            self.d_max,
            self.graph_features
        return [poly_dict, graph_dict]

@register_psfclass
class MCCDModel(DataDrivenPSFModel):
    id = ('mccd',)

    def __init__(self, spatial_dict):
        pass

@register_psfclass
class GraphModel(DataDrivenPSFModel):
    id = ('graph',)

    def __init__(self):
        pass