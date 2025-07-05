
"""Logging utilities for MongoDB."""
from __future__ import annotations 

from datetime import datetime 
from typing import Any ,Dict ,List 

def add_log (level :str ,msg :str )->None :
    """Insert a log entry with current UTC timestamp."""
    from .database import logs 

    if logs is None :
        print (f"ERROR: Database not connected. Could not log: {msg }")
        return 

    entry ={
    "timestamp":datetime .utcnow (),
    "level":level .upper (),
    "msg":msg ,
    }
    logs .insert_one (entry )


def list_logs (limit :int =50 )->List [Dict [str ,Any ]]:
    """Return the *limit* most recent log entries (defaults to 50)."""
    from .database import logs 

    if logs is None :
        return [{"level":"ERROR","msg":"Database not connected."}]

    cursor =logs .find ().sort ("timestamp",-1 ).limit (limit )
    return list (cursor )