
"""Model loading utilities for product-matching pipeline.

This module lazily downloads (on first use) and caches two pretrained
models:

1. Vision encoder – `facebook/dinov2-base` (DINOv2, ViT-style)
2. Text encoder   – `BAAI/bge-medium-en` (general embedding model)

Both are exposed via simple helper functions that return numpy arrays for
easy downstream use. No fine-tuning, minimal defaults.
"""
from __future__ import annotations 

import logging 
from functools import lru_cache 
from typing import List 

import numpy as np 
import torch 
from PIL import Image 
from transformers import AutoImageProcessor ,AutoModel ,AutoTokenizer 

LOGGER =logging .getLogger (__name__ )




VISION_MODEL_NAME ="facebook/dinov2-base"



TEXT_MODEL_NAME ="BAAI/bge-small-en"


_VISION_DIM =768 

_TEXT_DIM =384 





def _get_device ()->torch .device :
    return torch .device ("cuda"if torch .cuda .is_available ()else "cpu")






@lru_cache (maxsize =1 )
def _load_vision_components ():
    """Load and cache vision *processor* and *model*."""
    device =_get_device ()
    LOGGER .info ("Loading vision encoder (%s) to %s",VISION_MODEL_NAME ,device )

    processor =AutoImageProcessor .from_pretrained (VISION_MODEL_NAME )
    model =AutoModel .from_pretrained (VISION_MODEL_NAME )
    model .to (device )
    model .eval ()

    return processor ,model ,device 


def encode_image (image :Image .Image )->np .ndarray :
    """Return a DINOv2 embedding for *image* as a 1-D numpy array."""
    processor ,model ,device =_load_vision_components ()

    with torch .no_grad ():
        inputs =processor (images =image ,return_tensors ="pt").to (device )
        outputs =model (**inputs )

        embedding =outputs .last_hidden_state [:,0 ]
        embedding =torch .nn .functional .normalize (embedding ,dim =-1 )
        return embedding .cpu ().numpy ()[0 ]





@lru_cache (maxsize =1 )
def _load_text_components ():
    """Load tokenizer & model. Raises if model cannot be downloaded/loaded."""
    device =_get_device ()
    LOGGER .info ("Loading text encoder (%s) to %s",TEXT_MODEL_NAME ,device )

    tokenizer =AutoTokenizer .from_pretrained (TEXT_MODEL_NAME )
    model =AutoModel .from_pretrained (TEXT_MODEL_NAME )
    model .to (device )
    model .eval ()
    return tokenizer ,model ,device 


def encode_text (texts :List [str ]|str )->np .ndarray :
    """Return L2-normalised embeddings for *texts* (single string or list).

    Returns
    -------
    np.ndarray
        Embeddings with shape (N, dim) where N = len(texts).
    """
    if isinstance (texts ,str ):
        texts =[texts ]

    tokenizer ,model ,device =_load_text_components ()

    with torch .no_grad ():
        encoded =tokenizer (texts ,padding =True ,truncation =True ,return_tensors ="pt").to (device )
        outputs =model (**encoded )
        embeddings =outputs .last_hidden_state [:,0 ]
        embeddings =torch .nn .functional .normalize (embeddings ,dim =-1 )
        return embeddings .cpu ().numpy ()