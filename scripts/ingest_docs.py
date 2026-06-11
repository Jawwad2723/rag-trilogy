import argparse
import asyncio
import os
import sys
from typing import List

# Ensure src/ package is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.api.deps import get_ingestion_pipeline
from src.logger import get_logger

logger = get_logger(__name__)

def find_files(directory: str) -> List[str]:
    """Finds all support document files in the directory recursively."""
    supported_extensions = {".pdf", ".md", ".markdown", ".html", ".htm", ".txt"}
    filepaths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            if ext in supported_extensions:
                filepaths.append(os.path.join(root, file))
                
    return filepaths

async def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest document files into RAG stores.")
    parser.add_argument(
        "--dir",
        type=str,
        required=True,
        help="Path to the directory containing files to ingest."
    )
    args = parser.parse_args()

    if not os.path.exists(args.dir):
        print(f"Error: Directory '{args.dir}' does not exist.")
        sys.exit(1)

    filepaths = find_files(args.dir)
    if not filepaths:
        print(f"No supported documents found in directory '{args.dir}'.")
        sys.exit(0)

    print(f"Found {len(filepaths)} files to ingest. Starting pipeline...")
    
    pipeline = get_ingestion_pipeline()
    try:
        stats = await pipeline.ingest_files(filepaths)
        print("\nIngestion completed successfully!")
        print(f"  - Files processed: {stats['files_processed']}")
        print(f"  - Chunks created: {stats['chunks_created']}")
        print(f"  - Chunks skipped (deduplicated): {stats['chunks_skipped']}")
        print(f"  - Duration: {stats['duration_seconds']} seconds")
    except Exception as e:
        print(f"\nError occurred during ingestion: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
