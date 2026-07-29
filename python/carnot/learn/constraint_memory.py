import sqlite3
import os
from typing import List, Tuple, Optional
import datetime
from carnot.paths import repo_path


class ConstraintMemoryCache:
    """
    Tier 2 constraint memory cache.
    Persists verified facts across sessions to a SQLite database.
    """

    def __init__(self, db_path: str | None = None):
        # Resolved at CALL time via the central resolver rather than a hardcoded
        # absolute default, which made every clone share one developer's database.
        # See python/carnot/paths.py.
        if db_path is None:
            db_path = str(repo_path("data", "constraint_memory.db"))
        self.db_path = db_path
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS verified_facts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    fact_text TEXT NOT NULL,
                    violation_count INTEGER DEFAULT 0,
                    precision_rate REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    UNIQUE(domain, fact_text)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS constraint_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    domain TEXT NOT NULL,
                    template_pattern TEXT NOT NULL,
                    success_rate REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    UNIQUE(domain, template_pattern)
                )
            """)
            conn.commit()

    def store_violation(self, domain: str, fact_text: str, was_real_error: bool) -> int:
        """
        Records a violation for a specific fact.
        Updates violation count and precision rate (where precision = correct violations / total violations).
        Creates a template pattern if the fact is new.
        Returns the fact ID.
        """
        now = datetime.datetime.now(datetime.UTC).isoformat()
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Check if fact exists
            cursor.execute(
                "SELECT id, violation_count, precision_rate FROM verified_facts WHERE domain = ? AND fact_text = ?",
                (domain, fact_text),
            )
            row = cursor.fetchone()

            if row:
                fact_id, v_count, p_rate = row
                new_count = v_count + 1

                # If was_real_error is true, that means the violation was a correct catch (precision).
                # Current successful catches = v_count * p_rate.
                successes = v_count * p_rate
                if was_real_error:
                    successes += 1

                new_precision = successes / new_count

                cursor.execute(
                    "UPDATE verified_facts SET violation_count = ?, precision_rate = ? WHERE id = ?",
                    (new_count, new_precision, fact_id),
                )
                conn.commit()
                return fact_id
            else:
                new_count = 1
                new_precision = 1.0 if was_real_error else 0.0
                cursor.execute(
                    "INSERT INTO verified_facts (domain, fact_text, violation_count, precision_rate, created_at) VALUES (?, ?, ?, ?, ?)",
                    (domain, fact_text, new_count, new_precision, now),
                )
                fact_id = cursor.lastrowid

                # Also store a template pattern based on the fact text
                template_pattern = f"Template for: {fact_text}"
                try:
                    cursor.execute(
                        "INSERT INTO constraint_templates (domain, template_pattern, success_rate, created_at) VALUES (?, ?, ?, ?)",
                        (domain, template_pattern, new_precision, now),
                    )
                except sqlite3.IntegrityError:
                    pass

            conn.commit()
            return fact_id

    def update_precision(self, fact_id: int, was_correct: bool):
        """
        Updates precision of an existing fact.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT violation_count, precision_rate FROM verified_facts WHERE id = ?",
                (fact_id,),
            )
            row = cursor.fetchone()
            if row:
                v_count, p_rate = row
                new_count = v_count + 1
                successes = v_count * p_rate
                if was_correct:
                    successes += 1
                new_precision = successes / new_count
                cursor.execute(
                    "UPDATE verified_facts SET violation_count = ?, precision_rate = ? WHERE id = ?",
                    (new_count, new_precision, fact_id),
                )
            conn.commit()

    def query_templates(self, domain: str) -> List[str]:
        """
        Returns top-5 templates by success_rate for domain.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT template_pattern FROM constraint_templates WHERE domain = ? ORDER BY success_rate DESC LIMIT 5",
                (domain,),
            )
            rows = cursor.fetchall()
            return [row[0] for row in rows]

    def get_all_facts(self) -> List[Tuple]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, domain, fact_text, violation_count, precision_rate FROM verified_facts"
            )
            return cursor.fetchall()

    def clear(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM verified_facts")
            cursor.execute("DELETE FROM constraint_templates")
            conn.commit()
