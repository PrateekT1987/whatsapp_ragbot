"""Structured data extraction from PDFs using pdfplumber.

Parses table-based PDFs into SQLite for exact SQL queries.
"""

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pdfplumber

from config import config

logger = logging.getLogger(__name__)


class TableExtractor:
    """Extract structured tables from PDFs and store in SQLite."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self) -> None:
        """Create the structured data tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS members (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    s_no INTEGER,
                    name TEXT NOT NULL,
                    mobile TEXT,
                    address TEXT,
                    district TEXT,
                    pin TEXT,
                    raw_address TEXT,
                    source_file TEXT NOT NULL,
                    page_number INTEGER NOT NULL,
                    row_number INTEGER NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_members_name ON members(name)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_members_mobile ON members(mobile)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_members_district ON members(district)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_members_pin ON members(pin)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_members_source ON members(source_file)
            """)

    def _parse_address(self, raw: str) -> Tuple[str, str, str]:
        """Extract district and pin from raw address.

        Returns: (clean_address, district, pin)
        """
        if not raw:
            return "", "", ""

        # Try to find pin code (6 digits)
        pin_match = re.search(r"\b(\d{6})\b", raw)
        pin = pin_match.group(1) if pin_match else ""

        # Known districts in Karnataka (add more as needed)
        districts = [
            "BAGALKOT",
            "BANGALORE",
            "BANGALORE RURAL",
            "BELGAUM",
            "BELLARY",
            "BIDAR",
            "BIJAPUR",
            "CHAMARAJANAGAR",
            "CHIKKABALLAPUR",
            "CHIKMAGALUR",
            "CHITRADURGA",
            "DAKSHINA KANNADA",
            "DAVANGERE",
            "DHARWAD",
            "GADAG",
            "GULBARGA",
            "HASSAN",
            "HAVERI",
            "KALABURAGI",
            "KODAGU",
            "KOLAR",
            "KOPPAL",
            "MANDYA",
            "MYSORE",
            "RAICHUR",
            "RAMANAGARA",
            "SHIMOGA",
            "TUMKUR",
            "UDUPI",
            "UTTARA KANNADA",
            "YADGIR",
        ]

        district = ""
        upper_raw = raw.upper()
        for d in districts:
            if d in upper_raw:
                district = d.title()
                break

        # Clean address: remove the district/pin artifacts
        clean = raw
        if pin:
            clean = clean.replace(pin, "")
        if district:
            clean = (
                clean.replace(district.upper(), "")
                .replace(district.lower(), "")
                .replace(district.title(), "")
            )
        clean = re.sub(r"\s+", " ", clean).strip(" ,;\n")

        return clean, district, pin

    def extract_from_pdf(self, pdf_path: Path, force_reload: bool = False) -> int:
        """Extract all tables from a PDF and store in SQLite.

        Returns number of rows inserted.
        """
        if not force_reload:
            # Check if already processed
            with sqlite3.connect(self.db_path) as conn:
                count = conn.execute(
                    "SELECT COUNT(*) FROM members WHERE source_file = ?",
                    (pdf_path.name,),
                ).fetchone()[0]
                if count > 0:
                    logger.info(
                        f"Table data for {pdf_path.name} already exists ({count} rows)"
                    )
                    return 0
        else:
            # A reload must replace, not append: drop this file's old rows first.
            with sqlite3.connect(self.db_path) as conn:
                removed = conn.execute(
                    "DELETE FROM members WHERE source_file = ?", (pdf_path.name,)
                ).rowcount
            if removed:
                logger.info(
                    "Re-extracting %s: removed %d old row(s)", pdf_path.name, removed
                )

        logger.info(f"Extracting tables from {pdf_path.name}...")
        inserted = 0

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                if not tables:
                    continue

                for table in tables:
                    if len(table) < 2:  # header only
                        continue

                    # Expect header: S.no, Member Name, Mobile No, Registered Address, Photo
                    header = [str(c).strip().lower() if c else "" for c in table[0]]
                    if "s.no" not in header and "s no" not in " ".join(header):
                        continue  # Not our target table

                    # Find column indices
                    name_idx = mobile_idx = addr_idx = -1
                    for i, h in enumerate(header):
                        h = h.replace(".", "").replace(" ", "")
                        if "member" in h and "name" in h:
                            name_idx = i
                        elif "mobile" in h:
                            mobile_idx = i
                        elif "address" in h:
                            addr_idx = i

                    if name_idx == -1 or addr_idx == -1:
                        continue

                    for row_num, row in enumerate(table[1:], 1):
                        if not row or len(row) <= max(name_idx, mobile_idx, addr_idx):
                            continue

                        name = str(row[name_idx]).strip() if row[name_idx] else ""
                        mobile = (
                            str(row[mobile_idx]).strip()
                            if mobile_idx >= 0 and row[mobile_idx]
                            else ""
                        )
                        raw_addr = str(row[addr_idx]).strip() if row[addr_idx] else ""

                        if not name or name.lower() in ("member name", "name"):
                            continue

                        # Parse address
                        clean_addr, district, pin = self._parse_address(raw_addr)

                        # Clean mobile
                        mobile = re.sub(r"\D", "", mobile)
                        if mobile in ("TBD", "", "0"):
                            mobile = ""

                        with sqlite3.connect(self.db_path) as conn:
                            conn.execute(
                                """
                                INSERT INTO members
                                (s_no, name, mobile, address, district, pin, raw_address, source_file, page_number, row_number)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                                (
                                    row_num,
                                    name,
                                    mobile or None,
                                    clean_addr,
                                    district,
                                    pin,
                                    raw_addr,
                                    pdf_path.name,
                                    page_num,
                                    row_num,
                                ),
                            )
                            inserted += 1

        logger.info(f"Inserted {inserted} rows from {pdf_path.name}")
        return inserted

    def query(self, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute a read-only SQL query."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    def count_by_district(self, district: str) -> int:
        """Count members in a district."""
        with sqlite3.connect(self.db_path) as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM members WHERE district LIKE ?", (f"%{district}%",)
            ).fetchone()[0]

    def search_by_name(self, name: str) -> List[Dict[str, Any]]:
        """Search members by name (partial match)."""
        return self.query(
            """
            SELECT s_no, name, mobile, address, district, pin
            FROM members
            WHERE name LIKE ?
            ORDER BY name
            LIMIT 20
        """,
            (f"%{name}%",),
        )

    def search_by_mobile(self, mobile: str) -> Optional[Dict[str, Any]]:
        """Search by mobile number."""
        mobile_digits = re.sub(r"\D", "", mobile)
        rows = self.query(
            """
            SELECT s_no, name, mobile, address, district, pin
            FROM members
            WHERE mobile = ?
            LIMIT 1
        """,
            (mobile_digits,),
        )
        return rows[0] if rows else None

    def get_stats(self) -> Dict[str, Any]:
        """Get table statistics."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute("SELECT COUNT(*) FROM members").fetchone()[0]
            districts = conn.execute("""
                SELECT district, COUNT(*) as cnt
                FROM members
                WHERE district IS NOT NULL AND district != ''
                GROUP BY district
                ORDER BY cnt DESC
            """).fetchall()
            files = conn.execute("""
                SELECT source_file, COUNT(*) as cnt
                FROM members
                GROUP BY source_file
            """).fetchall()

            return {
                "total_rows": total,
                "districts": {d[0]: d[1] for d in districts},
                "files": {f[0]: f[1] for f in files},
            }


# Global instance - lazy initialization
_table_extractor: Optional[TableExtractor] = None


def get_table_extractor() -> TableExtractor:
    """Get or create the table extractor with current config."""
    global _table_extractor
    if _table_extractor is None:
        _table_extractor = TableExtractor(config.STRUCTURED_DB_PATH)
    return _table_extractor
