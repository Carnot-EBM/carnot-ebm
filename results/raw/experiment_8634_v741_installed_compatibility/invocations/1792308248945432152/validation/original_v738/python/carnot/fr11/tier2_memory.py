import sqlite3
import os

class Tier2ThresholdMemory:
    def __init__(self, db_path="data/fr11_tier2_memory.db"):
        self.db_path = db_path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        self._init_db()
        self.schema_version = "v1.0"

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS domain_thresholds (
                    domain_key TEXT PRIMARY KEY,
                    threshold_delta REAL NOT NULL,
                    n_examples INTEGER NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            conn.commit()
        finally:
            conn.close()

    def update_domain_delta(self, domain_key: str, examples: list[float], labels: list[int]):
        """
        Given examples (raw scores) and labels (0 or 1), compute threshold shift delta.
        We'll use a simple approach: find the mean score of the examples and compare it
        to the ideal center (0.5).
        """
        if len(examples) < 32:
            raise ValueError("Need at least 32 examples per domain to reliably estimate delta")
            
        # simple estimation: we want the threshold that best separates positives and negatives.
        # Alternatively, we just align the mean of the scores to 0.5
        mean_score = sum(examples) / len(examples)
        delta = mean_score - 0.5
        
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO domain_thresholds (domain_key, threshold_delta, n_examples)
                VALUES (?, ?, ?)
                ON CONFLICT(domain_key) DO UPDATE SET
                    threshold_delta=excluded.threshold_delta,
                    n_examples=excluded.n_examples,
                    updated_at=CURRENT_TIMESTAMP
            ''', (domain_key, delta, len(examples)))
            conn.commit()
        finally:
            conn.close()

    def get_domain_delta(self, domain_key: str) -> float:
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT threshold_delta FROM domain_thresholds WHERE domain_key = ?", (domain_key,))
            row = cursor.fetchone()
            if row:
                return row[0]
            return 0.0
        finally:
            conn.close()

    def apply_delta(self, domain_key: str, raw_score: float) -> float:
        delta = self.get_domain_delta(domain_key)
        return raw_score - delta
