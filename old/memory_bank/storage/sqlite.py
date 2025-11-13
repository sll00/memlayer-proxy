import sqlite3
from pathlib import Path

# This will be our local database engine.
# We will define the schema for memories, entities, etc., here.
# For now, a simple `memories` table is enough for the fast path.

CREATE_MEMORIES_TABLE = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    embedding BLOB NOT NULL,
    timestamp REAL NOT NULL,
    user_id TEXT,
    metadata TEXT
);
"""

class SQLiteStorage:
    def __init__(self, storage_path: str):
        self.db_path = Path(storage_path) / "memory_bank.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        # TODO: Load sqlite-vss extension
        self.conn.execute(CREATE_MEMORIES_TABLE)
        print(f"Memory Bank initialized at: {self.db_path}")

    # We will add methods like `add_memory`, `search_memories`, etc.