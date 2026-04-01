import sqlite3, os

DB_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "app.db")
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# ── Drop two columns from fight (rebuild method — SQLite limitation) ──
cur.executescript("""
    BEGIN;
    ALTER TABLE fight RENAME TO fight_old;
    CREATE TABLE fight (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER REFERENCES user(id),
        timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
        clip_path   VARCHAR(255),
        fight_score FLOAT
    );
    INSERT INTO fight (id, user_id, timestamp, clip_path, fight_score)
    SELECT id, user_id, timestamp, clip_path, fight_score FROM fight_old;
    DROP TABLE fight_old;
    COMMIT;
""")
print("✅ fight table updated")

# ── Create fire table ──
cur.execute("""
    CREATE TABLE IF NOT EXISTS fire (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER REFERENCES user(id),
        timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
        clip_path   VARCHAR(255),
        fight_score FLOAT
    )
""")
print("✅ fire table created")

# ── Create weapon table ──
cur.execute("""
    CREATE TABLE IF NOT EXISTS weapon (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id     INTEGER REFERENCES user(id),
        timestamp   DATETIME DEFAULT CURRENT_TIMESTAMP,
        clip_path   VARCHAR(255),
        fight_score FLOAT
    )
""")
print("✅ weapon table created")

conn.commit()
conn.close()
print("Done. All existing data preserved.")