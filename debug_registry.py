import pickle

with open('checkpoints/registry.pkl', 'rb') as f:
    data = pickle.load(f)

print("Metadata:")
for fid, meta in data['metadata'].items():
    print(f"  {fid}: {meta}")

print("\nCompositions:")
for fid, comp in data['compositions'].items():
    print(f"  {fid}: {comp}")

print(f"\nloop_id: {data['loop_id']}")
print(f"next_id: {data['next_id']}")