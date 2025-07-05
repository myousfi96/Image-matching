"""Minimal MongoDB helper using `pymongo`.

This module spins up a real MongoDB instance and exposes
convenient helper functions for the two required collections:

1. `products` – stores basic product metadata
2. `logs`     – stores simple log entries

The goal is zero-configuration usage across the code-base.
"""
from __future__ import annotations 

import os 
from datetime import datetime 
from typing import Any ,Dict ,List ,Optional 

from bson .objectid import ObjectId 
from pymongo import MongoClient 
from pymongo .collection import Collection 
from dotenv import load_dotenv 


products :Optional [Collection ]=None 
logs :Optional [Collection ]=None 

def connect_to_mongo ():
    """Connect to MongoDB and initialize collections."""
    global products ,logs 

    load_dotenv ()

    MONGO_HOST =os .getenv ("MONGO_HOST","localhost")
    MONGO_URI =f"mongodb://{MONGO_HOST }:27017/"
    _client =MongoClient (MONGO_URI )
    _db =_client [os .getenv ("MONGO_DB_NAME","product_matching")]

    products =_db ["products"]
    logs =_db ["logs"]


    products .create_index ("category")
    print ("MongoDB connected and collections initialized.")





def add_product(name: str, category: str, image_url: str) -> str:
    """Insert a product document and return its generated _id."""
    if products is None :
        raise RuntimeError ("Database not connected. Call connect_to_mongo first.")
    doc = {
        "name": name,
        "category": category,
        "image_url": image_url,
    }
    result =products .insert_one (doc )
    return str (result .inserted_id )


def get_product (product_id :str )->Optional [Dict [str ,Any ]]:
    """Retrieve a product by its `_id` (as a string)."""
    if products is None :
        raise RuntimeError ("Database not connected. Call connect_to_mongo first.")
    return products .find_one ({"_id":ObjectId (product_id )})


def list_products (limit :int =20 )->List [Dict [str ,Any ]]:
    """Return up to *limit* products (defaults to 20)."""
    if products is None :
        raise RuntimeError ("Database not connected. Call connect_to_mongo first.")
    cursor =products .find ().limit (limit )
    return list (cursor )


def add_log (level :str ,message :str )->None :
    """Add a log entry to the `logs` collection."""
    if logs is None :
        raise RuntimeError ("Database not connected. Call connect_to_mongo first.")
    logs .insert_one ({"level":level ,"message":message ,"timestamp":datetime .utcnow ()})

