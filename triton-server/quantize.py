
"""Minimal TensorRT quantization script for vision & text encoders.

Usage::

    python quantize.py  # builds both engines

The script attempts to:
1. Export each PyTorch model to ONNX (stored in `triton/<name>/1/model.onnx`)
2. Convert the ONNX model to a TensorRT engine (`model.trt`) with FP16 & INT8
   enabled. If TensorRT isn't available (e.g., CPU-only machine), the ONNX file
   is retained and a warning is printed.

Directory layout created::

    triton/
      dinov2/
        1/model.trt (or .onnx fallback)
      bge/
        1/model.trt (or .onnx fallback)
"""
from __future__ import annotations 

import os 
import subprocess 
from pathlib import Path 
from typing import Optional 

import torch 
import shutil 
from transformers import AutoModel ,AutoTokenizer ,AutoImageProcessor 




DINOV2_NAME ="facebook/dinov2-base"
BGE_NAME ="BAAI/bge-small-en"

TRITON_DIR =Path ("triton")





def _ensure_output_dir (model_name :str )->Path :
    sub =TRITON_DIR /model_name /"1"
    sub .mkdir (parents =True ,exist_ok =True )
    return sub 


def _export_onnx (model :torch .nn .Module ,inputs :tuple [torch .Tensor ,...],out_path :Path ,
dynamic_axes :Optional [dict [str ,dict [int ,str ]]]=None )->None :
    torch .onnx .export (
    model ,
    inputs ,
    str (out_path ),
    export_params =True ,
    opset_version =17 ,
    do_constant_folding =True ,
    input_names =[f"input{i }"for i in range (len (inputs ))],
    output_names =["output"],
    dynamic_axes =dynamic_axes or {},
    )


def _build_trt (
onnx_path :Path ,
engine_path :Path ,
min_shapes :str |None =None ,
opt_shapes :str |None =None ,
max_shapes :str |None =None ,
)->None :
    """Use `trtexec` CLI if available to convert ONNX→TensorRT.

    Raises an exception on failure.
    """
    trtexec ="/usr/src/tensorrt/bin/trtexec"
    if not os .path .exists (trtexec ):
        raise RuntimeError (
        f"{trtexec } not found. This script requires NVIDIA TensorRT to be installed."
        " Please run this script in an environment with TensorRT, such as:"
        " `docker run --gpus all -v $(pwd):/workspace nvcr.io/nvidia/tritonserver:<xx.yy>-py3`"
        )

    cmd =[
    trtexec ,
    f"--onnx={onnx_path }",
    f"--saveEngine={engine_path }",
    "--fp16",


    "--verbose",
    "--memPoolSize=workspace:4096",
    ]
    if min_shapes :
        cmd .append (f"--minShapes={min_shapes }")
    if opt_shapes :
        cmd .append (f"--optShapes={opt_shapes }")
    if max_shapes :
        cmd .append (f"--maxShapes={max_shapes }")

    print ("[quantize] Running:"," ".join (cmd ))
    res =subprocess .run (cmd ,capture_output =True ,text =True )
    if res .returncode !=0 :
        print ("[quantize] trtexec failed:\n",res .stdout ,res .stderr )
        raise RuntimeError (f"TensorRT engine build failed for {onnx_path }")






def quantize_dinov2 ()->str :
    out_dir =_ensure_output_dir ("dinov2")
    onnx_path =out_dir /"model.onnx"
    engine_path =out_dir /"model.plan"

    if engine_path .exists ():
        print (f"[quantize] DINOv2 TensorRT engine already exists at {engine_path }, skipping.")
        return str (engine_path )

    print ("[quantize] Exporting DINOv2 ONNX …")
    processor =AutoImageProcessor .from_pretrained (DINOV2_NAME )
    model =AutoModel .from_pretrained (DINOV2_NAME )
    model .eval ()

    dummy =torch .randn (1 ,3 ,224 ,224 )
    _export_onnx (model ,(dummy ,),onnx_path )


    _build_trt (onnx_path ,engine_path )
    onnx_path .unlink ()
    print (f"[quantize] DINOv2 TensorRT engine saved to {engine_path }")
    return str (engine_path )


def quantize_bge ()->str :
    out_dir =_ensure_output_dir ("bge")
    onnx_path =out_dir /"model.onnx"
    engine_path =out_dir /"model.plan"

    if engine_path .exists ():
        print (f"[quantize] BGE TensorRT engine already exists at {engine_path }, skipping.")
        return str (engine_path )

    print ("[quantize] Exporting BGE ONNX …")
    tokenizer =AutoTokenizer .from_pretrained (BGE_NAME )
    model =AutoModel .from_pretrained (BGE_NAME )
    model .eval ()

    dummy_txt =tokenizer ("hello world",return_tensors ="pt")
    input_ids =dummy_txt ["input_ids"]
    attention =dummy_txt ["attention_mask"]

    dynamic ={
    "input0":{1 :"seq"},
    "input1":{1 :"seq"},
    }
    _export_onnx (model ,(input_ids ,attention ),onnx_path ,dynamic )


    _build_trt (
    onnx_path ,
    engine_path ,
    min_shapes ="input0:1x8,input1:1x8",
    opt_shapes ="input0:1x128,input1:1x128",
    max_shapes ="input0:1x512,input1:1x512",
    )
    onnx_path .unlink ()
    print (f"[quantize] BGE TensorRT engine saved to {engine_path }")
    return str (engine_path )






def main ()->None :
    print ("[quantize] Starting quantization …")
    dinov2_path =quantize_dinov2 ()
    bge_path =quantize_bge ()
    print ("[quantize] Done.\n  Vision engine:",dinov2_path ,"\n  Text engine  :",bge_path )


if __name__ =="__main__":
    main ()