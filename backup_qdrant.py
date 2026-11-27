#!/usr/bin/env python3
"""Qdrant snapshot backup/restore helper.

Usage:
  python backup_qdrant.py --create --collection rag_documents_personal
  python backup_qdrant.py --list
  python backup_qdrant.py --restore rag_documents_personal-2026-11-26-12-00-00

Requires QDRANT_URL and QDRANT_API_KEY in environment (or .env).
"""

import argparse
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")

HEADERS = {}
if QDRANT_API_KEY:
    HEADERS["api-key"] = QDRANT_API_KEY


def list_snapshots() -> list[dict]:
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{QDRANT_URL}/collections", headers=HEADERS)
        resp.raise_for_status()
        collections = resp.json()["result"]["collections"]

        snapshots = []
        for coll in collections:
            name = coll["name"]
            snap_resp = client.get(
                f"{QDRANT_URL}/collections/{name}/snapshots", headers=HEADERS
            )
            if snap_resp.status_code == 200:
                for snap in snap_resp.json()["result"]:
                    snapshots.append({"collection": name, **snap})
            else:
                snapshots.append({"collection": name, "snapshots": []})
        return snapshots


def create_snapshot(collection: str) -> str:
    with httpx.Client(timeout=60) as client:
        resp = client.post(
            f"{QDRANT_URL}/collections/{collection}/snapshots",
            headers=HEADERS,
        )
        resp.raise_for_status()
        data = resp.json()
        snapshot_name = data["result"]["name"]
        print(f"✅ Created snapshot: {snapshot_name}")
        return snapshot_name


def download_snapshot(collection: str, snapshot_name: str, dest: Path) -> None:
    with httpx.Client(timeout=300) as client:
        resp = client.get(
            f"{QDRANT_URL}/collections/{collection}/snapshots/{snapshot_name}",
            headers=HEADERS,
            follow_redirects=True,
        )
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"✅ Downloaded {len(resp.content)/1e6:.1f} MB to {dest}")


def restore_snapshot(collection: str, snapshot_path: Path) -> None:
    if not snapshot_path.exists():
        print(f"❌ Snapshot file not found: {snapshot_path}")
        sys.exit(1)

    with httpx.Client(timeout=300) as client:
        with open(snapshot_path, "rb") as f:
            files = {"snapshot": (snapshot_path.name, f, "application/octet-stream")}
            resp = client.post(
                f"{QDRANT_URL}/collections/{collection}/snapshots/upload",
                headers={k: v for k, v in HEADERS.items() if k != "api-key"},
                files=files,
                params={"api_key": QDRANT_API_KEY} if QDRANT_API_KEY else {},
            )
        resp.raise_for_status()
        print(f"✅ Restored snapshot into collection: {collection}")


def main():
    parser = argparse.ArgumentParser(description="Qdrant snapshot manager")
    parser.add_argument("--list", action="store_true", help="List all snapshots")
    parser.add_argument("--create", action="store_true", help="Create a snapshot")
    parser.add_argument(
        "--download", action="store_true", help="Download snapshot after create"
    )
    parser.add_argument(
        "--collection", help="Collection name (required for create/restore)"
    )
    parser.add_argument("--restore", help="Restore from local snapshot file")
    parser.add_argument("--output", default=".", help="Output directory for downloads")
    args = parser.parse_args()

    if args.list:
        snaps = list_snapshots()
        for s in snaps:
            print(f"  {s['collection']}: {s.get('name', 'no snapshots')}")
        return

    if args.create:
        if not args.collection:
            print("❌ --collection is required for --create")
            sys.exit(1)
        name = create_snapshot(args.collection)
        if args.download:
            dest = Path(args.output) / name
            download_snapshot(args.collection, name, dest)
        return

    if args.restore:
        if not args.collection:
            print("❌ --collection is required for --restore")
            sys.exit(1)
        restore_snapshot(args.collection, Path(args.restore))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
