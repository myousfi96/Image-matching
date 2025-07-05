# populate_db.py
"""Populate Qdrant & MongoDB mock with product embeddings & metadata.

Assumes that `prepare_dataset.py` has already been executed and generated:
    data/images/*.jpg
    data/metadata.json

The script performs the following:
1. Loads metadata (up to 200 items)
2. For each item:
   • Loads product image → vision embedding via encode_image()
   • Generates simple text embedding from category via encode_text()
3. Stores separate image/text embeddings in Qdrant (`vector_db.py`) with payload
4. Inserts product document into MongoDB (`database.py`)

Run:
    python populate_db.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

# Import helpers from sibling module within the same directory
from model_loader import encode_image, encode_text
from api.vector_db import upsert_product_embeddings
from api.database import add_product, add_log, connect_to_mongo

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")
METADATA_FILE = DATA_DIR / "metadata.json"
MAX_ITEMS = 200  # populate at most this many products
BATCH_SIZE = 32  # batch upsert into Qdrant for efficiency


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_metadata() -> List[dict]:
    if not METADATA_FILE.exists():
        raise FileNotFoundError("metadata.json not found – run prepare_dataset.py first")
    with METADATA_FILE.open("r") as f:
        data = json.load(f)
    return data[:MAX_ITEMS]


# ---------------------------------------------------------------------------
# Main population logic
# ---------------------------------------------------------------------------

def populate() -> None:
    connect_to_mongo()
    items = _load_metadata()
    print(f"Populating DB with {len(items)} items…")

    batch_image_embeddings: List[np.ndarray] = []
    batch_text_embeddings: List[np.ndarray] = []
    batch_payloads: List[dict] = []
    product_ids: List[str] = []

    for idx, item in enumerate(items, start=1):
        # Build image path relative to data/images/
        image_path = Path(item.get("image_path", ""))
        if not image_path.is_absolute():
            image_path = DATA_DIR / "images" / item["filename"]
        if not image_path.exists():
            add_log("ERROR", f"Missing image file: {image_path}")
            continue

        try:
            # 1. Insert product into MongoDB mock first to get product_id
            product_id = add_product(
                name=item["filename"],
                category=item["category"],
                price=0.0,  # price unknown for STL-10; placeholder
                image_url=str(image_path.relative_to(DATA_DIR)),
            )

            # 2. Vision embedding (768 dimensions)
            img = Image.open(image_path).convert("RGB")
            img_emb = encode_image(img).astype(np.float32)

            # 3. Text embedding (1024+ dimensions)
            txt_emb = encode_text(item["category"]).astype(np.float32)[0]

            # 4. Prepare payload for Qdrant with product_id reference
            payload = {
                "product_id": product_id,
                "filename": item["filename"],
                "category": item["category"],
            }

            batch_image_embeddings.append(img_emb)
            batch_text_embeddings.append(txt_emb)
            batch_payloads.append(payload)
            product_ids.append(product_id)

            # Upsert in batches for efficiency
            if len(batch_image_embeddings) >= BATCH_SIZE:
                upsert_product_embeddings(
                    image_embeddings=batch_image_embeddings,
                    text_embeddings=batch_text_embeddings,
                    payloads=batch_payloads
                )
                print(f"Upserted {len(batch_image_embeddings)} products with separate embeddings …")
                batch_image_embeddings.clear()
                batch_text_embeddings.clear()
                batch_payloads.clear()
                product_ids.clear()

        except Exception as exc:          # noqa: BLE001 – broad on purpose
            msg = f"Failed processing {image_path}: {exc}"
            add_log("ERROR", msg)
            print(msg)
            continue

    # Insert any remaining vectors
    if batch_image_embeddings:
        upsert_product_embeddings(
            image_embeddings=batch_image_embeddings,
            text_embeddings=batch_text_embeddings,
            payloads=batch_payloads
        )

    add_log("INFO", f"Database population complete. Total items: {len(items)}")
    print("Population finished. ✔")
    print(f"📊 Stored {len(items)} products with separate image (768d) and text (1024d) embeddings")


if __name__ == "__main__":
    populate() 