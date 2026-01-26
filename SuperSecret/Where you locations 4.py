import json
import numpy as np

def debug_d41_extraction(weight_file="Metalearnerv16_EVOLVED.json"):
    """Debug exactly what's in the weight file"""
    
    with open(weight_file, 'r') as f:
        data = json.load(f)
    
    print("🔍 DEEP DEBUG OF D41 EXTRACTION")
    print("="*60)
    
    # 1. Check Node 12 structure
    print("\n1. NODE 12 STRUCTURE:")
    node12 = data['meta_pantheon']['12']['state_dict']
    for key in node12.keys():
        if hasattr(node12[key], '__len__'):
            print(f"   {key}: shape would be {len(node12[key])} x ...")
        else:
            print(f"   {key}: {node12[key]}")
    
    # 2. Extract project_to_latent.weight properly
    print("\n2. EXTRACTING project_to_latent.weight:")
    if 'project_to_latent.weight' in node12:
        weights_array = np.array(node12['project_to_latent.weight'])
        print(f"   Shape: {weights_array.shape}")
        
        # Check all dimensions around 41
        print("\n3. DIMENSIONS 35-45 (around D41):")
        for dim in range(35, 46):
            dim_values = weights_array[:, dim]
            print(f"   D{dim:02}: mean={np.mean(dim_values):.10f}, min={np.min(dim_values):.10f}, max={np.max(dim_values):.10f}")
        
        # What does the White Paper say about D41?
        print("\n4. WHITE PAPER COMPARISON:")
        print(f"   White Paper D41 value: -0.01282715")
        print(f"   Extracted D41 mean:    {np.mean(weights_array[:, 40]):.10f}")
        print(f"   Difference:           {abs(-0.01282715 - np.mean(weights_array[:, 40])):.10f}")
        
        # Maybe it's not dimension 41 (index 40)?
        print("\n5. SEARCHING FOR -0.0128 PATTERN:")
        best_match = None
        best_diff = float('inf')
        
        for dim in range(64):
            dim_mean = np.mean(weights_array[:, dim])
            diff = abs(dim_mean - (-0.01282715))
            if diff < best_diff:
                best_diff = diff
                best_match = (dim, dim_mean)
        
        print(f"   Closest match to -0.01282715:")
        print(f"   Dimension {best_match[0]}: {best_match[1]:.10f} (diff: {best_diff:.10f})")
        
        return weights_array
    
    return None

# Also check Node 3's principle embeddings
def check_principle_embeddings(data):
    print("\n6. NODE 3 PRINCIPLE EMBEDDINGS:")
    node3 = data['meta_pantheon']['3']['state_dict']['principle_embeddings']
    embeddings = np.array(node3)
    print(f"   Shape: {embeddings.shape}")
    print(f"   First few values: {embeddings[0][:5]}")
    
    # Maybe D41 is here?
    if embeddings.shape[1] >= 41:
        print(f"   D41 in embeddings: {embeddings[:, 40]}")

# Run debug
weights_array = debug_d41_extraction()