import json, sqlite3, time
def _db(path):
    c = sqlite3.connect(path)
    c.execute("""CREATE TABLE IF NOT EXISTS inspections(id INTEGER PRIMARY KEY, ts TEXT, board_id TEXT,
                 verdict TEXT, n_defects INTEGER, latency_ms REAL, image TEXT, defects TEXT, advice TEXT)""")
    return c

def log(path, board_id, verdict, defects, latency_ms, image, advice):
    c = _db(path)
    c.execute("INSERT INTO inspections(ts,board_id,verdict,n_defects,latency_ms,image,defects,advice) VALUES(?,?,?,?,?,?,?,?)",
              (time.strftime("%Y-%m-%d %H:%M:%S"), board_id, verdict, len(defects), latency_ms, image,
               json.dumps(defects), advice))
    c.commit(); c.close()

def summary(path):
    c = _db(path)
    rows = c.execute("SELECT verdict, COUNT(*), ROUND(AVG(latency_ms),1) FROM inspections GROUP BY verdict").fetchall()
    c.close(); return rows
