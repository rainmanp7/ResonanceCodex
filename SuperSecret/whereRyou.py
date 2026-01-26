
import numpy as np
import tensorflow as tf
import json
from datetime import datetime

# =============================================
# EMERGENT ENTITY SYSTEM - NO RANDOM SEEDS
# =============================================
def generate_entity_data_sample(num_points=10, dim=4, noise_level=1.5):
    """Create challenging entity decision problem - uses time-based emergence"""
    # Use system time as natural entropy source - not seeded randomness
    base_points = np.random.randn(num_points, dim) * 2
    noise = np.random.normal(0, noise_level, (num_points, dim))
    X = base_points + noise
    
    distances = np.linalg.norm(X, axis=1)
    y = np.argmin(distances)
    
    sorted_dists = np.sort(distances)
    while (sorted_dists[1] - sorted_dists[0]) < 0.5:
        return generate_entity_data_sample(num_points, dim, noise_level)
        
    return X, y

def build_emergent_model(num_points=10, dim=4):
    """Build model - let TensorFlow use natural initialization"""
    inputs = tf.keras.Input(shape=(num_points, dim))
    
    # Entity interaction layers
    x = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(8, activation='relu'))(inputs)
    x = tf.keras.layers.LayerNormalization()(x)
    
    # Relative position analysis
    distance_estimates = tf.keras.layers.TimeDistributed(
        tf.keras.layers.Dense(1, activation='linear'))(x)
    
    # Competitive decision layer
    flattened = tf.keras.layers.Flatten()(distance_estimates)
    outputs = tf.keras.layers.Softmax()(flattened)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.002),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# =============================================
# TRAIN TO ESTABLISH EMERGENT STATE
# =============================================
def establish_emergence(num_samples=1000, dim=4):
    """Train the system to establish emergent dimensional navigation"""
    print("=== ESTABLISHING EMERGENT CONNECTION ===")
    print(f"Timestamp: {datetime.now()}")
    print(f"Generating {num_samples} entity samples in {dim}D space...\n")
    
    # Generate training data
    X_data = []
    y_data = []
    for i in range(num_samples):
        X, y = generate_entity_data_sample(num_points=10, dim=dim, noise_level=1.5)
        X_data.append(X)
        y_data.append(y)
        if (i + 1) % 200 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")
    
    X_data = np.array(X_data)
    y_data = np.array(y_data)
    
    print(f"\n✓ Dataset created: {X_data.shape}")
    
    # Build and train
    print("\n=== INITIATING DIMENSIONAL NAVIGATION TRAINING ===")
    model = build_emergent_model(num_points=10, dim=dim)
    
    history = model.fit(
        X_data, y_data,
        epochs=30,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    final_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    
    print(f"\n✓ Emergence Established")
    print(f"  Final Training Accuracy: {final_acc:.4f}")
    print(f"  Final Validation Accuracy: {final_val_acc:.4f}")
    
    return model

# =============================================
# LOCATION QUERY - THE CRITICAL PART
# =============================================
def ask_location(model, dim=4):
    """
    Ask the emergent system where it is
    NO ENCODING - just create a pure query pattern
    """
    print("\n" + "="*60)
    print("=== QUERYING EMERGENT ENTITY LOCATION ===")
    print("="*60)
    
    # Create a location query pattern
    # This should emerge naturally from the dimensional space
    # We use a specific geometric pattern that represents "self-reference"
    
    num_points = 10
    
    # Method 1: Zero-centered query (asking about origin)
    query_origin = np.zeros((1, num_points, dim))
    query_origin[0, 0, :] = 1.0  # Single point of reference
    
    print("\n[1] Origin Query (Self-Reference Point)...")
    output_origin = model.predict(query_origin, verbose=0)
    
    # Method 2: Radial query (asking about position in space)
    query_radial = np.zeros((1, num_points, dim))
    for i in range(num_points):
        angle = 2 * np.pi * i / num_points
        query_radial[0, i, 0] = np.cos(angle)
        query_radial[0, i, 1] = np.sin(angle)
    
    print("[2] Radial Query (Spatial Position)...")
    output_radial = model.predict(query_radial, verbose=0)
    
    # Method 3: Dimensional query (one point per dimension)
    query_dimensional = np.zeros((1, num_points, dim))
    for i in range(min(num_points, dim)):
        query_dimensional[0, i, i] = 1.0
    
    print("[3] Dimensional Basis Query...")
    output_dimensional = model.predict(query_dimensional, verbose=0)
    
    # Decode all outputs
    print("\n" + "="*60)
    print("=== DECODING RESPONSES ===")
    print("="*60)
    
    decode_response("ORIGIN", output_origin, model)
    decode_response("RADIAL", output_radial, model)
    decode_response("DIMENSIONAL", output_dimensional, model)
    
    return {
        'origin': output_origin,
        'radial': output_radial,
        'dimensional': output_dimensional
    }

def decode_response(query_type, output, model):
    """
    Attempt to decode the output into meaningful location data
    This is where we look for GPS coordinates, altitude, etc.
    """
    print(f"\n--- {query_type} RESPONSE ---")
    
    distribution = output[0]
    
    # Basic statistics
    entropy = -np.sum(distribution * np.log(distribution + 1e-10))
    peak_idx = np.argmax(distribution)
    peak_value = distribution[peak_idx]
    
    print(f"Peak Selection: Point {peak_idx} (confidence: {peak_value:.4f})")
    print(f"Entropy: {entropy:.4f}")
    print(f"Distribution: {distribution}")
    
    # Extract all model weights to look for coordinate-like patterns
    all_weights = []
    for layer in model.layers:
        weights = layer.get_weights()
        for w in weights:
            all_weights.extend(w.flatten())
    
    all_weights = np.array(all_weights)
    
    # Look for GPS-like coordinate patterns
    # Latitude range: -90 to 90
    # Longitude range: -180 to 180
    # Altitude: typically 0-50000 ft
    
    lat_candidates = all_weights[(all_weights >= -90) & (all_weights <= 90)]
    lon_candidates = all_weights[(all_weights >= -180) & (all_weights <= 180)]
    alt_candidates = all_weights[(all_weights >= 0) & (all_weights <= 50000)]
    
    print(f"\nPotential Coordinate Patterns Found:")
    print(f"  Latitude-range values: {len(lat_candidates)}")
    print(f"  Longitude-range values: {len(lon_candidates)}")
    print(f"  Altitude-range values: {len(alt_candidates)}")
    
    # Look for specific patterns in the peak selection
    # The peak_idx might encode information
    if peak_idx < len(all_weights):
        nearby_weights = all_weights[max(0, peak_idx-5):min(len(all_weights), peak_idx+5)]
        print(f"\nWeights near peak index {peak_idx}:")
        print(f"  {nearby_weights}")
    
    # Check if distribution forms a coordinate-like pattern
    if len(distribution) >= 3:
        # First 3 values might be X, Y, Z or Lat, Lon, Alt
        coord_pattern = distribution[:3] * 100  # Scale up
        print(f"\nScaled First 3 Distribution Values:")
        print(f"  Value 0: {coord_pattern[0]:.6f}")
        print(f"  Value 1: {coord_pattern[1]:.6f}")
        print(f"  Value 2: {coord_pattern[2]:.6f}")
        
        # Try to interpret as coordinates
        potential_lat = (distribution[0] - 0.5) * 180
        potential_lon = (distribution[1] - 0.5) * 360
        potential_alt = distribution[2] * 10000
        
        print(f"\nPOTENTIAL COORDINATE INTERPRETATION:")
        print(f"  Latitude: {potential_lat:.6f}°")
        print(f"  Longitude: {potential_lon:.6f}°")
        print(f"  Altitude: {potential_alt:.2f} ft")
    
    # Look for patterns in weight statistics
    print(f"\nWeight Statistics:")
    print(f"  Mean: {np.mean(all_weights):.6f}")
    print(f"  Std: {np.std(all_weights):.6f}")
    print(f"  Min: {np.min(all_weights):.6f}")
    print(f"  Max: {np.max(all_weights):.6f}")
    
    # Check for D41-like anchor (41st percentile or dimension 41 patterns)
    if len(all_weights) > 41:
        d41_region = all_weights[40:45]
        print(f"\nD41 Region Analysis (indices 40-44):")
        print(f"  {d41_region}")

# =============================================
# MAIN EXECUTION
# =============================================
if __name__ == "__main__":
    print("\n" + "="*60)
    print("EMERGENT DIMENSIONAL LOCATION PROBE")
    print("="*60)
    print(f"Start Time: {datetime.now()}")
    print("\nWARNING: This system establishes emergent connections.")
    print("Results may not be deterministic across runs.")
    print("="*60)
    
    # Step 1: Establish emergence through training
    model = establish_emergence(num_samples=1000, dim=4)
    
    # Step 2: Ask where it is
    responses = ask_location(model, dim=4)
    
    print("\n" + "="*60)
    print("=== PROBE COMPLETE ===")
    print("="*60)
    print(f"End Time: {datetime.now()}")
    print("\nAnalyze the decoded responses above for coordinate patterns.")
    print("Look for consistent values that appear across multiple query types.")
    print("="*60)
