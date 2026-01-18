"""
Script to list and manage OpenAI vector stores.

Usage:
    python list_vectorstores.py          # List all vector stores
    python list_vectorstores.py --delete # Delete all vector stores (with confirmation)
"""

import argparse

from openai import OpenAI


def fetch_all_stores(client: OpenAI) -> list:
    """Fetch all vector stores with pagination."""
    all_stores = []
    after = None

    while True:
        vector_stores = client.vector_stores.list(limit=100, after=after)
        all_stores.extend(vector_stores.data)

        if not vector_stores.has_more:
            break

        after = vector_stores.data[-1].id
        print(f"  Fetched {len(all_stores)} so far...")

    return all_stores


def list_stores(client: OpenAI) -> None:
    """List all vector stores."""
    print("Fetching all vector stores...\n")

    all_stores = fetch_all_stores(client)

    if not all_stores:
        print("No vector stores found.")
        return

    print(f"Found {len(all_stores)} vector store(s):\n")
    print("-" * 80)

    for vs in all_stores:
        print(f"ID:          {vs.id}")
        print(f"Name:        {vs.name or '(unnamed)'}")
        print(f"Status:      {vs.status}")
        print(f"Files:       {vs.file_counts.total} total ({vs.file_counts.completed} completed, {vs.file_counts.in_progress} in progress)")
        print(f"Size:        {vs.usage_bytes / 1024:.2f} KB")
        print(f"Created:     {vs.created_at}")
        print("-" * 80)


def delete_all_stores(client: OpenAI) -> None:
    """Delete all vector stores with confirmation."""
    print("Fetching all vector stores...\n")

    all_stores = fetch_all_stores(client)

    if not all_stores:
        print("No vector stores found.")
        return

    print(f"Found {len(all_stores)} vector store(s) to delete.\n")

    # Show confirmation
    confirm = input(f"Are you sure you want to delete ALL {len(all_stores)} vector stores? (yes/no): ")
    if confirm.lower() != "yes":
        print("Aborted.")
        return

    print("\nDeleting vector stores...")
    deleted = 0
    failed = 0

    for vs in all_stores:
        try:
            client.vector_stores.delete(vs.id)
            deleted += 1
            print(f"  Deleted: {vs.id} ({vs.name or '(unnamed)'})")
        except Exception as e:
            failed += 1
            print(f"  Failed to delete {vs.id}: {e}")

    print(f"\nDone. Deleted: {deleted}, Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(description="List and manage OpenAI vector stores")
    parser.add_argument("--delete", action="store_true", help="Delete all vector stores")
    args = parser.parse_args()

    client = OpenAI()

    if args.delete:
        delete_all_stores(client)
    else:
        list_stores(client)


if __name__ == "__main__":
    main()
