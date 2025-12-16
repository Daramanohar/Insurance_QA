"""
Delete vectors from Pinecone
"""

from pinecone_setup import PineconeManager

print("=" * 70)
print(" Delete Vectors from Pinecone")
print("=" * 70)

# Initialize
manager = PineconeManager()
manager.create_index()

# Get stats before
stats_before = manager.index.describe_index_stats()
count_before = stats_before['total_vector_count']
print(f"\nCurrent vectors in index: {count_before}")

print("\nWhat would you like to delete?")
print("1. Delete specific vector by ID")
print("2. Delete multiple vectors by IDs")
print("3. Delete ALL vectors (CAREFUL!)")
print("4. Cancel")

choice = input("\nEnter choice (1-4): ").strip()

if choice == "1":
    # Delete single vector
    vector_id = input("\nEnter vector ID to delete (e.g., qa_2042_0): ").strip()
    
    confirm = input(f"\nAre you sure you want to delete '{vector_id}'? (yes/no): ").strip().lower()
    
    if confirm == "yes":
        print(f"\nDeleting vector: {vector_id}...")
        manager.index.delete(ids=[vector_id])
        print("[OK] Deleted!")
    else:
        print("Cancelled.")

elif choice == "2":
    # Delete multiple vectors
    print("\nEnter vector IDs separated by commas:")
    print("Example: qa_2042_0, qa_4705_0, qa_6523_0")
    
    ids_input = input("\nIDs: ").strip()
    vector_ids = [id.strip() for id in ids_input.split(',')]
    
    print(f"\nYou want to delete {len(vector_ids)} vectors:")
    for vid in vector_ids:
        print(f"  - {vid}")
    
    confirm = input("\nAre you sure? (yes/no): ").strip().lower()
    
    if confirm == "yes":
        print(f"\nDeleting {len(vector_ids)} vectors...")
        manager.index.delete(ids=vector_ids)
        print("[OK] Deleted!")
    else:
        print("Cancelled.")

elif choice == "3":
    # Delete all vectors
    print("\n" + "!" * 70)
    print(" WARNING: This will delete ALL vectors in your index!")
    print("!" * 70)
    
    confirm1 = input("\nType 'DELETE ALL' to confirm: ").strip()
    
    if confirm1 == "DELETE ALL":
        confirm2 = input("Are you absolutely sure? (yes/no): ").strip().lower()
        
        if confirm2 == "yes":
            print("\nDeleting ALL vectors...")
            manager.index.delete(delete_all=True, namespace="")
            print("[OK] All vectors deleted!")
            print("\nTo repopulate, run: python pinecone_setup.py")
        else:
            print("Cancelled.")
    else:
        print("Cancelled.")

else:
    print("\nCancelled.")

# Get stats after
print("\n" + "=" * 70)
stats_after = manager.index.describe_index_stats()
count_after = stats_after['total_vector_count']

if count_before != count_after:
    print(f"Vectors before: {count_before}")
    print(f"Vectors after:  {count_after}")
    print(f"Deleted:        {count_before - count_after}")
else:
    print("No changes made.")

print("=" * 70)

