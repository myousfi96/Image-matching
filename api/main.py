"""FastAPI Backend for Product Matching System

This FastAPI server acts as an intermediary between the Streamlit frontend
and the Triton inference server. It provides REST endpoints for:
- Health checks
- Image embedding extraction
- Product matching
- System stats

Architecture: Streamlit → FastAPI → Triton
"""
from __future__ import annotations 

import io 
import logging 
from typing import List ,Dict ,Any ,Optional 
from pathlib import Path 
import base64 
import os 

from fastapi import FastAPI ,UploadFile ,File ,HTTPException ,status 
from fastapi .middleware .cors import CORSMiddleware 
from fastapi .responses import JSONResponse 
from pydantic import BaseModel 
import numpy as np 
from PIL import Image 
import tritonclient .http as httpclient 
from tritonclient .http import InferInput ,InferRequestedOutput 
from fastapi .staticfiles import StaticFiles 
from transformers import AutoTokenizer 

from .vector_db import search_embeddings ,search_text_embeddings 
from .database import get_product ,list_products ,connect_to_mongo 
from .log_utils import add_log ,list_logs 


TRITON_HOST =os .environ .get ("TRITON_HOST","localhost")
TRITON_SERVER_URL =f"{TRITON_HOST }:8000"
DINOV2_MODEL_NAME =os .environ .get ("DINOV2_MODEL_NAME","dinov2")
BGE_MODEL_NAME =os .environ .get ("BGE_MODEL_NAME","bge")
TOP_K_RESULTS =5 

logger =logging .getLogger (__name__ )


app =FastAPI (
title ="Product Matching API",
description ="REST API for product matching using computer vision",
version ="1.0.0",
docs_url ="/docs",
redoc_url ="/redoc"
)

app .mount ("/data",StaticFiles (directory ="data"),name ="data")

app .add_middleware (
CORSMiddleware ,
allow_origins =["*"],
allow_credentials =True ,
allow_methods =["*"],
allow_headers =["*"],
)


class ProductMatch(BaseModel):
    id: str
    name: str
    category: str
    image_url: str
    similarity_score: float


class MatchingResponse (BaseModel ):
    success :bool 
    matches :List [ProductMatch ]
    total_matches :int 
    processing_time_ms :float 
    message :Optional [str ]=None 


class TextSearchRequest (BaseModel ):
    query :str 


class HealthResponse (BaseModel ):
    status :str 
    triton_online :bool 
    models_ready :Dict [str ,bool ]
    message :Optional [str ]=None 


def get_triton_client ()->httpclient .InferenceServerClient :
    """Get Triton inference client."""
    return httpclient .InferenceServerClient (url =TRITON_SERVER_URL )


def preprocess_image (image :Image .Image )->np .ndarray :
    """Preprocess image for DINOv2 model."""
    if image .mode !='RGB':
        image =image .convert ('RGB')

    image =image .resize ((224 ,224 ),Image .Resampling .LANCZOS )
    img_array =np .array (image ).astype (np .float32 )/255.0 
    img_array =img_array .transpose (2 ,0 ,1 )
    img_array =np .expand_dims (img_array ,axis =0 )

    return img_array 


def extract_image_embeddings (image :Image .Image )->np .ndarray :
    """Extract embeddings from image using Triton DINOv2 model."""
    try :
        client =get_triton_client ()

        img_array =preprocess_image (image )

        inputs =[]
        inputs .append (InferInput ("input0",img_array .shape ,"FP32"))
        inputs [0 ].set_data_from_numpy (img_array )

        outputs =[]
        outputs .append (InferRequestedOutput ("output"))


        response =client .infer (
        model_name =DINOV2_MODEL_NAME ,
        inputs =inputs ,
        outputs =outputs 
        )


        embeddings =response .as_numpy ("output")
        if embeddings is None :
            raise HTTPException (
            status_code =status .HTTP_500_INTERNAL_SERVER_ERROR ,
            detail ="Failed to get embeddings from Triton response"
            )


        if embeddings .shape ==(1 ,257 ,768 ):
            embeddings =embeddings [0 ,0 ,:]
        else :
            embeddings =embeddings .flatten ()


        add_log ("INFO",f"Extracted image embeddings, shape: {embeddings .shape }")

        return embeddings 

    except Exception as e :
        logger .error (f"Failed to extract image embeddings: {e }")
        raise HTTPException (
        status_code =status .HTTP_500_INTERNAL_SERVER_ERROR ,
        detail =f"Embedding extraction failed: {str (e )}"
        )


def extract_text_embeddings (text :str )->np .ndarray :
    """Extract embeddings from text using Triton BGE model."""
    try :
        client =get_triton_client ()


        tokenizer =AutoTokenizer .from_pretrained ("BAAI/bge-small-en")

        encoded =tokenizer ([text ],padding ="max_length",truncation =True ,return_tensors ="np")

        input_ids =encoded ["input_ids"].astype (np .int32 )
        attention_mask =encoded ["attention_mask"].astype (np .int32 )


        inputs =[
        InferInput ("input0",input_ids .shape ,"INT32"),
        InferInput ("input1",attention_mask .shape ,"INT32")
        ]
        inputs [0 ].set_data_from_numpy (input_ids )
        inputs [1 ].set_data_from_numpy (attention_mask )

        outputs =[InferRequestedOutput ("output")]

        response =client .infer (
        model_name =BGE_MODEL_NAME ,
        inputs =inputs ,
        outputs =outputs 
        )

        embeddings =response .as_numpy ("output")
        if embeddings is None :
            raise HTTPException (
            status_code =status .HTTP_500_INTERNAL_SERVER_ERROR ,
            detail ="Failed to get embeddings from Triton response"
            )



        if len (embeddings .shape )==3 :
            embeddings =embeddings [:,0 ,:]


        norm =np .linalg .norm (embeddings ,axis =1 ,keepdims =True )
        normalized_embeddings =embeddings /norm 

        add_log ("INFO",f"Extracted text embeddings, shape: {normalized_embeddings .shape }")

        return normalized_embeddings .flatten ()

    except Exception as e :
        logger .error (f"Failed to extract text embeddings: {e }")
        raise HTTPException (
        status_code =status .HTTP_500_INTERNAL_SERVER_ERROR ,
        detail =f"Text embedding extraction failed: {str (e )}"
        )


@app .get ("/",response_model =Dict [str ,str ])
async def root ():
    """Root endpoint with API information."""
    return {
    "message":"Product Matching API",
    "version":"1.0.0",
    "docs":"/docs",
    "health":"/health"
    }


@app .get ("/health",response_model =HealthResponse )
async def health_check ():
    """Health check endpoint."""
    try :
        client =get_triton_client ()
        triton_online =client .is_server_ready ()

        models_ready ={"dinov2":False ,"bge":False }

        if triton_online :
            try :

                dinov2_ready =client .is_model_ready (DINOV2_MODEL_NAME )
                bge_ready =client .is_model_ready (BGE_MODEL_NAME )

                models_ready ["dinov2"]=dinov2_ready 
                models_ready ["bge"]=bge_ready 

            except Exception as e :
                logger .warning (f"Failed to check model readiness: {e }")

        overall_status ="healthy"if triton_online and all (models_ready .values ())else "degraded"

        return HealthResponse (
        status =overall_status ,
        triton_online =triton_online ,
        models_ready =models_ready ,
        message ="All systems operational"if overall_status =="healthy"else "Some components unavailable"
        )

    except Exception as e :
        logger .error (f"Health check failed: {e }")
        return HealthResponse (
        status ="error",
        triton_online =False ,
        models_ready ={"dinov2":False ,"bge":False },
        message =f"Health check failed: {str (e )}"
        )


@app .post ("/match",response_model =MatchingResponse )
async def match_products (file :UploadFile =File (...)):
    """Match products based on uploaded image."""
    import time 
    start_time =time .time ()

    if not file or not file .content_type or not file .content_type .startswith ('image/'):
        raise HTTPException (
        status_code =status .HTTP_400_BAD_REQUEST ,
        detail ="File must be an image"
        )

    try :

        image_bytes =await file .read ()
        image =Image .open (io .BytesIO (image_bytes ))

        add_log ("INFO",f"Processing image upload: {file .filename }")


        embeddings =extract_image_embeddings (image )


        search_results =search_embeddings (embeddings ,top_k =TOP_K_RESULTS )
        print (search_results )


        matched_products =[]
        if search_results :
            for result in search_results :
                if not result or not result .payload :
                    continue 
                product_id =result .payload .get ("product_id")
                if not product_id :
                    continue 

                product =get_product (str (product_id ))
                if product :
                    product_match =ProductMatch (
                    id =str (product ["_id"]),
                    name =product .get ("name","Unknown"),
                    category =product .get ("category","Unknown"),
                    image_url =product .get ("image_url",""),
                    similarity_score =float (result .score )
                    )
                    matched_products .append (product_match )

        processing_time =(time .time ()-start_time )*1000 

        add_log ("INFO",f"Found {len (matched_products )} matches in {processing_time :.2f}ms")

        return MatchingResponse (
        success =True ,
        matches =matched_products ,
        total_matches =len (matched_products ),
        processing_time_ms =round (processing_time ,2 ),
        message =f"Found {len (matched_products )} similar products"
        )

    except HTTPException :
        raise 
    except Exception as e :
        processing_time =(time .time ()-start_time )*1000 
        error_msg =f"Product matching failed: {str (e )}"
        add_log ("ERROR",error_msg )
        logger .error (error_msg )

        return MatchingResponse (
        success =False ,
        matches =[],
        total_matches =0 ,
        processing_time_ms =round (processing_time ,2 ),
        message =error_msg 
        )


@app .post ("/search_by_text",response_model =MatchingResponse )
async def search_by_text (request :TextSearchRequest ):
    """Search products by text query."""
    import time 
    start_time =time .time ()

    try :
        query =request .query 
        if not query :
            raise HTTPException (
            status_code =status .HTTP_400_BAD_REQUEST ,
            detail ="Query text cannot be empty"
            )

        add_log ("INFO",f"Processing text search for: '{query }'")


        embeddings =extract_text_embeddings (query )


        search_results =search_text_embeddings (embeddings ,top_k =TOP_K_RESULTS )


        matched_products =[]
        if search_results :
            for result in search_results :
                if not result or not result .payload :
                    continue 
                product_id =result .payload .get ("product_id")
                if not product_id :
                    continue 

                product =get_product (str (product_id ))
                if product :
                    product_match =ProductMatch (
                    id =str (product ["_id"]),
                    name =product .get ("name","Unknown"),
                    category =product .get ("category","Unknown"),
                    image_url =product .get ("image_url",""),
                    similarity_score =float (result .score )
                    )
                    matched_products .append (product_match )

        processing_time =(time .time ()-start_time )*1000 

        add_log ("INFO",f"Found {len (matched_products )} matches for query '{query }' in {processing_time :.2f}ms")

        return MatchingResponse (
        success =True ,
        matches =matched_products ,
        total_matches =len (matched_products ),
        processing_time_ms =round (processing_time ,2 ),
        message =f"Found {len (matched_products )} similar products for query '{query }'"
        )

    except HTTPException :
        raise 
    except Exception as e :
        processing_time =(time .time ()-start_time )*1000 
        error_msg =f"Text search failed: {str (e )}"
        add_log ("ERROR",error_msg )
        logger .error (error_msg )

        return MatchingResponse (
        success =False ,
        matches =[],
        total_matches =0 ,
        processing_time_ms =round (processing_time ,2 ),
        message =error_msg 
        )


@app .get ("/products",response_model =List [Dict [str ,Any ]])
async def get_products (limit :int =20 ):
    """Get list of products in the database."""
    try :
        products =list_products (limit =limit )

        for product in products :
            if "_id"in product :
                product ["_id"]=str (product ["_id"])
        return products 
    except Exception as e :
        logger .error (f"Failed to get products: {e }")
        raise HTTPException (
        status_code =status .HTTP_500_INTERNAL_SERVER_ERROR ,
        detail =f"Failed to get products: {str (e )}"
        )


@app .get ("/logs",response_model =List [Dict [str ,Any ]])
async def get_logs (limit :int =50 ):
    """Get recent logs."""
    try :
        logs =list_logs (limit =limit )

        for log in logs :
            if "timestamp"in log and hasattr (log ["timestamp"],'isoformat'):
                log ["timestamp"]=log ["timestamp"].isoformat ()

            if "_id"in log :
                log ["_id"]=str (log ["_id"])
        return logs 
    except Exception as e :
        logger .error (f"Failed to get logs: {e }")
        raise HTTPException (
        status_code =status .HTTP_500_INTERNAL_SERVER_ERROR ,
        detail =f"Failed to get logs: {str (e )}"
        )






@app .on_event ("startup")
async def startup_event ():
    """Connect to DB and log startup event."""
    connect_to_mongo ()
    add_log ("INFO","FastAPI server started")
    logger .info ("FastAPI server started")


@app .on_event ("shutdown")
async def shutdown_event ():
    """Log shutdown event."""
    add_log ("INFO","FastAPI server shutdown")
    logger .info ("FastAPI server shutdown")






@app .exception_handler (Exception )
async def global_exception_handler (request ,exc ):
    """Global exception handler."""
    error_msg =f"Unexpected error: {str (exc )}"
    add_log ("ERROR",error_msg )
    logger .error (error_msg )

    return JSONResponse (
    status_code =status .HTTP_500_INTERNAL_SERVER_ERROR ,
    content ={"detail":error_msg }
    )






if __name__ =="__main__":
    import uvicorn 

    uvicorn .run (
    "main:app",
    host ="0.0.0.0",
    port =8080 ,
    reload =True ,
    log_level ="info"
    )