"""Streamlit Frontend with FastAPI Backend

Modified to use FastAPI as intermediary:
Architecture: Streamlit → FastAPI → Triton

- File uploader for input images
- Display top-k matches with metadata
- Show similarity scores
- System health monitoring via FastAPI
- Error handling with troubleshooting

Single page, no session state, clean styling
"""
import streamlit as st 
import pandas as pd 
from PIL import Image 
import numpy as np 
from typing import List ,Dict ,Any ,Optional 
import io 
import requests 
import json 
import time 
import os 






FASTAPI_URL =os .getenv ("API_URL","http://localhost:8080")
PUBLIC_FASTAPI_URL =os .getenv ("PUBLIC_API_URL",FASTAPI_URL )


st .set_page_config (
page_title ="Product Matching System",
page_icon ="🔍",
layout ="wide",
initial_sidebar_state ="expanded"
)





def call_api (endpoint :str ,method :str ="GET",files :Optional [Dict ]=None ,params :Optional [Dict ]=None )->Dict :
    """Make API call to FastAPI backend."""
    try :
        url =f"{FASTAPI_URL }{endpoint }"

        if method =="GET":
            response =requests .get (url ,params =params ,timeout =30 )
        elif method =="POST":
            if files :
                response =requests .post (url ,files =files ,timeout =120 )
            else :
                response =requests .post (url ,json =params ,timeout =30 )
        else :
            raise ValueError (f"Unsupported HTTP method: {method }")

        response .raise_for_status ()
        return response .json ()

    except requests .exceptions .RequestException as e :
        st .error (f"API Error: {str (e )}")
        return {"error":str (e )}
    except Exception as e :
        st .error (f"Unexpected error: {str (e )}")
        return {"error":str (e )}


def check_api_health ()->Dict :
    """Check FastAPI backend health."""
    return call_api ("/health")


def match_products_api (uploaded_file )->Dict :
    """Match products via FastAPI backend."""
    files ={"file":(uploaded_file .name ,uploaded_file .getvalue (),uploaded_file .type )}
    return call_api ("/match",method ="POST",files =files )


def search_by_text_api (query :str )->Dict :
    """Search products by text query via FastAPI backend."""
    return call_api ("/search_by_text",method ="POST",params ={"query":query })






def display_product_card (product :Dict [str ,Any ],rank :int )->None :
    """Display a single product match as a card."""
    col1 ,col2 =st .columns ([1 ,3 ])

    with col1 :
        image_url =product .get ("image_url")
        if image_url :

            full_image_url =f"{PUBLIC_FASTAPI_URL }/data/{image_url }"
            st .image (
            full_image_url ,
            caption =f"Rank #{rank }",
            use_container_width =True ,
            )
        else :

            st .image (
            f"https://via.placeholder.com/150x150?text=No+Image",
            caption =f"Rank #{rank }",
            width =150 ,
            )

    with col2 :
        st .subheader (f"#{rank }: {product ['name']}")
        st .write (f"**Category:** {product ['category']}")
        st .write (f"**Similarity Score:** {product ['similarity_score']:.3f}")
        st .write (f"**Product ID:** {product ['id']}")


        st .progress (product ['similarity_score'])

    st .divider ()


def display_error (error_msg :str )->None :
    """Display error message with styling."""
    st .error (f"❌ **Error:** {error_msg }")
    st .info ("💡 **Troubleshooting:**")
    st .write ("- Make sure FastAPI server is running on localhost:8080")
    st .write ("- Check that Triton server is running on localhost:8000")
    st .write ("- Verify the database has been populated with products")
    st .write ("- Check that the image format is supported (JPEG, PNG, etc.)")


def health_check_section ():
    """Display system health check section."""
    st .sidebar .header ("🏥 Health Check")

    if st .sidebar .button ("🔍 Check System Status"):
        try :
            health_response =check_api_health ()

            if "error"in health_response :
                st .sidebar .error (f"❌ Health Check Failed: {health_response ['error']}")
                return 


            status =health_response .get ("status","unknown")
            if status =="healthy":
                st .sidebar .success ("✅ System Status: Healthy")
            elif status =="degraded":
                st .sidebar .warning ("⚠️ System Status: Degraded")
            else :
                st .sidebar .error ("❌ System Status: Error")


            triton_online =health_response .get ("triton_online",False )
            if triton_online :
                st .sidebar .success ("✅ Triton Server: Online")
            else :
                st .sidebar .error ("❌ Triton Server: Offline")


            models_ready =health_response .get ("models_ready",{})
            for model_name ,ready in models_ready .items ():
                if ready :
                    st .sidebar .success (f"✅ {model_name .upper ()} Model: Ready")
                else :
                    st .sidebar .error (f"❌ {model_name .upper ()} Model: Not Ready")


            message =health_response .get ("message")
            if message :
                st .sidebar .info (f"ℹ️ {message }")

        except Exception as e :
            st .sidebar .error (f"❌ Health Check Failed: {e }")






def main ():
    """Main Streamlit application."""


    st .title ("🔍 Product Matching System")
    st .markdown ("""
    Upload an image to find similar products in our database. 
    The system uses computer vision to match products based on visual similarity.
    
    **Architecture:** Streamlit → FastAPI → Triton Inference Server
    """)


    try :
        api_info =call_api ("/")
        if "error"not in api_info :
            st .success (f"✅ Connected to {api_info .get ('message','API')} v{api_info .get ('version','1.0')}")
        else :
            st .error (f"❌ Cannot connect to FastAPI backend: {api_info ['error']}")
    except :
        st .error ("❌ Cannot connect to FastAPI backend at http://localhost:8080")


    tab1 ,tab2 =st .tabs (["🖼️ Image Search","✍️ Text Search"])

    with tab1 :

        st .header ("📤 Upload Product Image")
        uploaded_file =st .file_uploader (
        "Choose an image file",
        type =['jpg','jpeg','png','bmp','gif'],
        help ="Upload a product image to find similar items"
        )

        if uploaded_file is not None :

            col1 ,col2 =st .columns ([2 ,3 ])

            with col1 :
                st .subheader ("📸 Uploaded Image")
                image =Image .open (uploaded_file )
                st .image (image ,caption ="Query Image",use_container_width =True )


                st .write (f"**Size:** {image .size }")
                st .write (f"**Mode:** {image .mode }")
                st .write (f"**Format:** {image .format }")
                st .write (f"**File Size:** {len (uploaded_file .getvalue ())} bytes")

            with col2 :
                st .subheader ("🔄 Processing...")


                if st .button ("🚀 Find Similar Products",type ="primary"):
                    try :

                        with st .spinner ("Extracting features and searching..."):

                            result =match_products_api (uploaded_file )


                        if "error"in result :
                            display_error (result ["error"])
                            return 


                        if not result .get ("success",False ):
                            display_error (result .get ("message","Unknown error occurred"))
                            return 


                        matches =result .get ("matches",[])
                        processing_time =result .get ("processing_time_ms",0 )

                        if matches :
                            st .success (f"✅ Found {len (matches )} similar products in {processing_time :.2f}ms!")


                            st .header ("🎯 Matching Results")


                            for i ,product in enumerate (matches ,1 ):
                                display_product_card (product ,i )

                        else :
                            st .warning ("⚠️ No similar products found in the database.")
                            st .info ("Try uploading a different image or check if the database is populated.")

                    except Exception as e :
                        display_error (str (e ))

        else :

            st .info ("👆 Please upload an image to start matching products.")

    with tab2 :
        st .header ("✏️ Search by Text Description")
        text_query =st .text_input (
        "Enter a description to search for products",
        placeholder ="e.g., 'a red dress with white polka dots'",
        help ="Describe the product you want to find."
        )

        if st .button ("🔍 Search by Text",type ="primary"):
            if text_query :
                try :
                    with st .spinner ("Encoding text and searching for products..."):
                        result =search_by_text_api (text_query )

                    if "error"in result :
                        display_error (result ["error"])
                        return 

                    if not result .get ("success",False ):
                        display_error (result .get ("message","Unknown error occurred"))
                        return 

                    matches =result .get ("matches",[])
                    processing_time =result .get ("processing_time_ms",0 )

                    if matches :
                        st .success (f"✅ Found {len (matches )} products for '{text_query }' in {processing_time :.2f}ms!")
                        st .header ("🎯 Matching Results")
                        for i ,product in enumerate (matches ,1 ):
                            display_product_card (product ,i )
                    else :
                        st .warning (f"⚠️ No products found matching '{text_query }'.")
                except Exception as e :
                    display_error (str (e ))
            else :
                st .warning ("Please enter a text query to search.")

    st .divider ()

    health_check_section ()






if __name__ =="__main__":
    main ()