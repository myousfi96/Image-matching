
"""Minimal in-process Qdrant wrapper with separate image/text vectors.

This module provides a zero-configuration, in-process Qdrant setup using
`qdrant-client`'s local mode with separate vector spaces for image and text embeddings:

1. `get_client()` – returns the singleton Qdrant client (local, file-based)
2. `ensure_collection()` – creates the collection with image/text vector configs
3. `upsert_product_embeddings()` – insert image embeddings with metadata
4. `search_image_embeddings()` – cosine similarity search on image vectors

Minimal approach: Image-only search for product matching.
"""
from __future__ import annotations 

import os 
from typing import List ,Sequence ,Dict ,Any 
from pathlib import Path 

import numpy as np 
from qdrant_client import QdrantClient 
from qdrant_client .http import models as qmodels 




COLLECTION_NAME =os .getenv ("QDRANT_COLLECTION","products")
QDRANT_HOST =os .getenv ("QDRANT_HOST")



_DEFAULT_PATH =Path (__file__ ).resolve ().parent /".qdrant_data"
STORAGE_PATH =os .getenv ("QDRANT_PATH",str (_DEFAULT_PATH ))

_client :QdrantClient |None =None 






def get_client ()->QdrantClient :
    """Return a singleton local QdrantClient instance."""
    global _client 
    if _client is None :
        if QDRANT_HOST :
            _client =QdrantClient (host =QDRANT_HOST ,port =6333 )
        else :
            _client =QdrantClient (path =STORAGE_PATH )
    return _client 


def ensure_collection ()->None :
    """Create or update the collection with separate image/text vector configs."""
    client =get_client ()
    if COLLECTION_NAME in [c .name for c in client .get_collections ().collections ]:
        return 

    client .create_collection (
    collection_name =COLLECTION_NAME ,
    vectors_config ={
    "image":qmodels .VectorParams (size =768 ,distance =qmodels .Distance .COSINE ),
    "text":qmodels .VectorParams (size =384 ,distance =qmodels .Distance .COSINE ),
    },
    )


def upsert_product_embeddings (
image_embeddings :Sequence [np .ndarray ],
payloads :Sequence [dict ]|None =None ,
text_embeddings :Sequence [np .ndarray ]|None =None ,
ids :Sequence [int ]|None =None ,
)->None :
    """Insert or update product embeddings with separate image/text vectors.
    
    For product matching, we primarily use image embeddings.
    Text embeddings are optional for future enhancements.
    """
    if not image_embeddings :
        return 

    ensure_collection ()
    client =get_client ()


    if ids is None :
        current_count =client .count (collection_name =COLLECTION_NAME ,exact =True ).count 
        ids =list (range (current_count ,current_count +len (image_embeddings )))

    payloads =payloads if payloads is not None else [{}for _ in image_embeddings ]


    points =[]
    for i ,(pid ,img_emb ,payload )in enumerate (zip (ids ,image_embeddings ,payloads )):
        vector_data ={"image":img_emb .tolist ()}


        if text_embeddings and i <len (text_embeddings ):
            vector_data ["text"]=text_embeddings [i ].tolist ()

        points .append (qmodels .PointStruct (
        id =pid ,
        vector =vector_data ,
        payload =payload 
        ))

    client .upsert (collection_name =COLLECTION_NAME ,points =points )


def search_embeddings (query :np .ndarray ,top_k :int =5 )->List [qmodels .ScoredPoint ]:
    """Search using image embeddings (primary search method)."""
    ensure_collection ()
    client =get_client ()


    return client .search (
    collection_name =COLLECTION_NAME ,
    query_vector =("image",query .tolist ()),
    limit =top_k 
    )


def search_image_embeddings (query :np .ndarray ,top_k :int =5 )->List [qmodels .ScoredPoint ]:
    """Explicit image embedding search."""
    return search_embeddings (query ,top_k )


def search_text_embeddings (query :np .ndarray ,top_k :int =5 )->List [qmodels .ScoredPoint ]:
    """Text embedding search (for future use)."""
    ensure_collection ()
    client =get_client ()

    return client .search (
    collection_name =COLLECTION_NAME ,
    query_vector =("text",query .tolist ()),
    limit =top_k 
    )






def upsert_embeddings (
embeddings :Sequence [np .ndarray ],
payloads :Sequence [dict ]|None =None ,
ids :Sequence [int ]|None =None ,
)->None :
    """Backward compatibility: treat all embeddings as image embeddings."""
    upsert_product_embeddings (embeddings ,payloads ,None ,ids )


