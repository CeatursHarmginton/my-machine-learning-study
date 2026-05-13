"""
Glossary / Terminology Manager for consistent novel translation.

Manages per-project glossaries with CRUD, import/export, and enforcement.
"""
import csv
import io
import json
from pathlib import Path
from typing import Optional

from config import GLOSSARY_DIR


class GlossaryManager:
    """
    Manages translation glossaries for consistent terminology.

    Each glossary entry has:
    - source: Original term (e.g., "林凡")
    - target: Target translation (e.g., "Lâm Phàm")
    - category: character | place | skill | title | item | organization | other
    - note: Optional context note
    """

    CATEGORIES = [
        "character", "place", "skill", "title",
        "item", "organization", "other",
    ]

    def __init__(self, project_name: str = "default"):
        self.project_name = project_name
        self.filepath = GLOSSARY_DIR / f"{project_name}.json"
        self.entries: list[dict] = []
        self._load()

    def _load(self):
        """Load glossary from disk."""
        if self.filepath.exists():
            with open(self.filepath, "r", encoding="utf-8") as f:
                self.entries = json.load(f)

    def save(self):
        """Save glossary to disk."""
        GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, "w", encoding="utf-8") as f:
            json.dump(self.entries, f, ensure_ascii=False, indent=2)

    def add(
        self,
        source: str,
        target: str,
        category: str = "other",
        note: str = "",
    ) -> bool:
        """
        Add a glossary entry.

        Returns False if the source term already exists.
        """
        # Check for duplicate
        if any(e["source"] == source for e in self.entries):
            return False

        self.entries.append({
            "source": source,
            "target": target,
            "category": category,
            "note": note,
        })
        self.save()
        return True

    def update(self, index: int, source: str, target: str, category: str = "", note: str = ""):
        """Update an existing entry by index."""
        if 0 <= index < len(self.entries):
            self.entries[index]["source"] = source
            self.entries[index]["target"] = target
            if category:
                self.entries[index]["category"] = category
            if note is not None:
                self.entries[index]["note"] = note
            self.save()

    def delete(self, index: int):
        """Delete entry by index."""
        if 0 <= index < len(self.entries):
            self.entries.pop(index)
            self.save()

    def bulk_add(self, entries: list[dict]) -> int:
        """
        Add multiple entries at once. Skips duplicates.

        Returns number of entries added.
        """
        added = 0
        for entry in entries:
            if self.add(
                source=entry.get("source", ""),
                target=entry.get("target", ""),
                category=entry.get("category", "other"),
                note=entry.get("note", ""),
            ):
                added += 1
        return added

    def get_by_category(self, category: str) -> list[dict]:
        """Get entries filtered by category."""
        return [e for e in self.entries if e.get("category") == category]

    def search(self, query: str) -> list[dict]:
        """Search glossary by source or target term."""
        query_lower = query.lower()
        return [
            e for e in self.entries
            if query_lower in e["source"].lower() or
               query_lower in e["target"].lower()
        ]

    def get_relevant_entries(self, text: str) -> list[dict]:
        """
        Find glossary entries whose source terms appear in the given text.

        Used for injecting relevant terms into the translation prompt.
        """
        relevant = []
        for entry in self.entries:
            if entry["source"] in text:
                relevant.append(entry)
        return relevant

    def check_enforcement(self, source_text: str, translation: str) -> list[dict]:
        """
        Check if glossary terms in the source are correctly translated.

        Returns list of violations.
        """
        violations = []
        relevant = self.get_relevant_entries(source_text)

        for entry in relevant:
            if entry["source"] in source_text and entry["target"] not in translation:
                violations.append({
                    "source": entry["source"],
                    "expected": entry["target"],
                    "issue": f"Term '{entry['source']}' found in source but '{entry['target']}' not found in translation",
                })

        return violations

    # ===== Import / Export =====

    def export_json(self) -> str:
        """Export glossary as JSON string."""
        return json.dumps(self.entries, ensure_ascii=False, indent=2)

    def export_csv(self) -> str:
        """Export glossary as CSV string."""
        output = io.StringIO()
        writer = csv.DictWriter(
            output,
            fieldnames=["source", "target", "category", "note"],
        )
        writer.writeheader()
        writer.writerows(self.entries)
        return output.getvalue()

    def import_json(self, json_str: str) -> int:
        """Import entries from JSON string. Returns count added."""
        try:
            data = json.loads(json_str)
            if isinstance(data, list):
                return self.bulk_add(data)
        except json.JSONDecodeError:
            pass
        return 0

    def import_csv(self, csv_str: str) -> int:
        """Import entries from CSV string. Returns count added."""
        reader = csv.DictReader(io.StringIO(csv_str))
        entries = []
        for row in reader:
            if "source" in row and "target" in row:
                entries.append({
                    "source": row["source"],
                    "target": row["target"],
                    "category": row.get("category", "other"),
                    "note": row.get("note", ""),
                })
        return self.bulk_add(entries)

    def import_from_file(self, filepath: str) -> int:
        """Import from a file (JSON or CSV)."""
        path = Path(filepath)
        content = path.read_text(encoding="utf-8")

        if path.suffix.lower() == ".json":
            return self.import_json(content)
        elif path.suffix.lower() == ".csv":
            return self.import_csv(content)
        else:
            # Try JSON first, then CSV
            count = self.import_json(content)
            if count == 0:
                count = self.import_csv(content)
            return count

    # ===== Display =====

    def get_display_data(self) -> list[list]:
        """Get entries formatted for Gradio Dataframe."""
        return [
            [i, e["source"], e["target"], e.get("category", "other"), e.get("note", "")]
            for i, e in enumerate(self.entries)
        ]

    def get_stats(self) -> dict:
        """Get glossary statistics."""
        cat_counts = {}
        for e in self.entries:
            cat = e.get("category", "other")
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        return {
            "total_entries": len(self.entries),
            "categories": cat_counts,
            "project": self.project_name,
        }

    def __len__(self):
        return len(self.entries)

    def __repr__(self):
        return f"GlossaryManager(project={self.project_name}, entries={len(self.entries)})"


def list_projects() -> list[str]:
    """List all available glossary projects."""
    GLOSSARY_DIR.mkdir(parents=True, exist_ok=True)
    return [
        p.stem for p in GLOSSARY_DIR.glob("*.json")
    ]
