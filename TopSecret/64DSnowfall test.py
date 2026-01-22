
import json
import numpy as np
import requests
from datetime import datetime
import time

# ============================================================================
# PANTHEON CONFIGURATION
# ============================================================================
D41_ANCHOR = -0.01282715  # Gyroscopic Stabilizer
ENTROPY_TARGET = 2.8862    # Specialist Fluidity
WEIGHT_FILE = 'Metalearnerv16_EVOLVED.json'

# ============================================================================
# NOAA WEATHER DATA INGESTION (SNOWFALL FOCUSED)
# ============================================================================

def fetch_noaa_weather(lat=39.158168, lon=-75.524368):  # Wilmington, Delaware
    """Fetch real-time NOAA weather data for Delaware"""
    headers = {
        'User-Agent': '(PantheonWeatherAI, contact@example.com)'
    }
    
    try:
        print("📡 Fetching NOAA data...")
        # Step 1: Get grid coordinates for location
        points_url = f"https://api.weather.gov/points/{lat},{lon}"
        points_response = requests.get(points_url, headers=headers, timeout=10)
        points_data = points_response.json()
        
        # Step 2: Get detailed forecast
        forecast_url = points_data['properties']['forecast']
        forecast_response = requests.get(forecast_url, headers=headers, timeout=10)
        forecast_data = forecast_response.json()
        
        # Step 3: Get grid data (raw atmospheric parameters)
        grid_url = points_data['properties']['forecastGridData']
        grid_response = requests.get(grid_url, headers=headers, timeout=10)
        grid_data = grid_response.json()
        
        print("✅ NOAA data acquired")
        return {
            'forecast': forecast_data,
            'grid': grid_data,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        print(f"❌ NOAA Fetch Error: {e}")
        return None

# ============================================================================
# ENCODING: SNOWFALL DATA → 64D MANIFOLD
# ============================================================================

def encode_snowfall_to_manifold(weather_data):
    """Encode NOAA snowfall-relevant data into 64D geometric space"""
    vec = np.zeros(64)
    
    if not weather_data or 'grid' not in weather_data:
        return vec
    
    try:
        grid = weather_data['grid']['properties']
        
        # D0-9: Temperature profiles (critical for snow vs rain)
        if 'temperature' in grid and grid['temperature']['values']:
            temps = [v['value'] for v in grid['temperature']['values'][:10] if v['value'] is not None]
            for i, temp in enumerate(temps):
                # Convert Celsius to Fahrenheit: (C * 9/5) + 32
                temp_f = (temp * 9/5) + 32
                # Snow typically forms below 32°F - encode proximity to freezing
                vec[i] = max(0, (32 - temp_f) / 50)  # Higher value = colder = more snow
        
        # D10-14: Precipitation probability
        if 'probabilityOfPrecipitation' in grid and grid['probabilityOfPrecipitation']['values']:
            precip_vals = [v['value'] for v in grid['probabilityOfPrecipitation']['values'][:5] if v['value'] is not None]
            for i, precip in enumerate(precip_vals):
                vec[10 + i] = (precip / 100) if precip else 0
        
        # D15-19: Quantitative precipitation forecast (QPF)
        if 'quantitativePrecipitation' in grid and grid['quantitativePrecipitation']['values']:
            qpf_vals = [v['value'] for v in grid['quantitativePrecipitation']['values'][:5] if v['value'] is not None]
            for i, qpf in enumerate(qpf_vals):
                # Convert mm to inches: mm / 25.4
                vec[15 + i] = min((qpf / 25.4) / 2.0, 1.0)  # 2 inches = 1.0
        
        # D20-24: Snowfall amount (CRITICAL SIGNAL)
        if 'snowfallAmount' in grid and grid['snowfallAmount']['values']:
            snow_vals = [v['value'] for v in grid['snowfallAmount']['values'][:5] if v['value'] is not None]
            for i, snow in enumerate(snow_vals):
                # Convert mm to inches and normalize
                snow_inches = snow / 25.4
                vec[20 + i] = min(snow_inches / 20.0, 1.0)  # 20 inches = 1.0
        
        # D25-29: Atmospheric pressure (affects storm intensity)
        if 'pressure' in grid and grid['pressure']['values']:
            pressure_vals = [v['value'] for v in grid['pressure']['values'][:5] if v['value'] is not None]
            for i, press in enumerate(pressure_vals):
                # Lower pressure = stronger storm
                # Normalize Pa (95000-105000 Pa, inverted for storm strength)
                vec[25 + i] = (105000 - press) / 10000
        
        # D30-34: Relative humidity (higher = more moisture for snow)
        if 'relativeHumidity' in grid and grid['relativeHumidity']['values']:
            humidity_vals = [v['value'] for v in grid['relativeHumidity']['values'][:5] if v['value'] is not None]
            for i, hum in enumerate(humidity_vals):
                vec[30 + i] = hum / 100
        
        # D35-39: Wind speed (affects snow accumulation and drifting)
        if 'windSpeed' in grid and grid['windSpeed']['values']:
            wind_vals = [v['value'] for v in grid['windSpeed']['values'][:5] if v['value'] is not None]
            for i, wind in enumerate(wind_vals):
                # Convert km/h, moderate wind is ideal for snow
                vec[35 + i] = min(wind / 50, 1.0)
        
        # D40: Ice accumulation (if available)
        if 'iceAccumulation' in grid and grid['iceAccumulation']['values']:
            ice_vals = [v['value'] for v in grid['iceAccumulation']['values'] if v['value'] is not None]
            if ice_vals:
                vec[40] = min(sum(ice_vals) / 25.4 / 0.5, 1.0)  # 0.5 inches = 1.0
        
        # D41: SANCTUARY ANCHOR (D41_ANCHOR) - Gyroscopic Stabilizer
        vec[41] = D41_ANCHOR
        
        # D42-46: Weather hazards from text forecast
        forecast = weather_data.get('forecast', {}).get('properties', {}).get('periods', [])
        if forecast:
            snow_keywords = ['snow', 'blizzard', 'winter storm', 'heavy snow', 'accumulation']
            for i, period in enumerate(forecast[:5]):
                text = period.get('detailedForecast', '').lower()
                snow_score = sum(1 for kw in snow_keywords if kw in text) / len(snow_keywords)
                vec[42 + i] = snow_score
        
        # D47-50: Sky cover (cloudiness - essential for precipitation)
        if 'skyCover' in grid and grid['skyCover']['values']:
            cloud_vals = [v['value'] for v in grid['skyCover']['values'][:4] if v['value'] is not None]
            for i, cloud in enumerate(cloud_vals):
                vec[47 + i] = cloud / 100
        
        # D51-60: Time signatures and temporal patterns
        hour = datetime.now().hour
        day = datetime.now().weekday()
        vec[51] = hour / 24.0
        vec[52] = day / 7.0
        
        # Storm duration indicator (how many periods show snow)
        if 'snowfallAmount' in grid and grid['snowfallAmount']['values']:
            snow_periods = sum(1 for v in grid['snowfallAmount']['values'] if v['value'] and v['value'] > 0)
            vec[53] = min(snow_periods / 20, 1.0)
        
        # D61-63: Reserved for future patterns
        
    except Exception as e:
        print(f"⚠️ Encoding warning: {e}")
    
    # Normalize vector
    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8) if norm > 0 else vec

# ============================================================================
# PANTHEON DELIBERATION FOR SNOWFALL PREDICTION
# ============================================================================

def load_pantheon():
    """Load Pantheon meta-learner weights"""
    print("--- 🛰️ SYNCHRONIZING WITH NODE 12 SANCTUARY ---")
    try:
        with open(WEIGHT_FILE, 'r') as f:
            weights = json.load(f)
        nodes = weights['meta_pantheon']
        print("✅ Pantheon nodes loaded")
        return nodes
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return None

def run_snowfall_deliberation(nodes, weather_vec):
    """7-round Pantheon deliberation with EMERGENT FEEDBACK LOOPS"""
    specialists = ['3', '4', '5', '6', '7', '8', '9', '10', '11']
    
    # Initialize evolving state
    evolving_manifold = weather_vec.copy()
    cumulative_score = 0.0
    round_predictions = []
    round_confidence = []
    specialist_memory = {s_id: [] for s_id in specialists}
    
    print("\n" + "="*60)
    print("❄️  PANTHEON SNOWFALL PREDICTION ANALYSIS")
    print("="*60)
    
    for r in range(1, 8):
        print(f"\n🔄 Round {r}/7: Specialist manifold projection...")
        
        votes = []
        specialist_outputs = {}
        
        # Each specialist evaluates the EVOLVING manifold
        for s_id in specialists:
            try:
                principles = np.array(nodes[s_id]['state_dict']['principle_embeddings'])
                
                # Calculate alignment with each principle
                alignments = [np.dot(evolving_manifold, pr) for pr in principles]
                
                # Specialist vote with memory of previous rounds
                current_vote = np.mean(alignments)
                
                # Apply specialist memory (temporal context)
                if specialist_memory[s_id]:
                    momentum = np.mean(specialist_memory[s_id][-3:])  # Last 3 rounds
                    current_vote = 0.7 * current_vote + 0.3 * momentum
                
                specialist_memory[s_id].append(current_vote)
                votes.append(current_vote)
                
                specialist_outputs[s_id] = {
                    'vote': current_vote,
                    'alignment_std': np.std(alignments),
                    'max_alignment': max(alignments),
                    'min_alignment': min(alignments)
                }
                
            except Exception as e:
                print(f"⚠️ Specialist {s_id} error: {e}")
                votes.append(0)
        
        # Node 12 Integration with D41 Stabilization
        raw_consensus = np.mean(votes)
        stabilized_consensus = raw_consensus * (1.0 + D41_ANCHOR)
        
        # ICEBERG LOGIC: Accumulate with tension
        cumulative_score = (cumulative_score * (r-1) + stabilized_consensus) / r
        
        # Convert consensus to snowfall inches
        snowfall_estimate = max(0, cumulative_score * 30)
        
        # Calculate TRUE confidence from specialist agreement
        vote_std = np.std(votes)
        vote_mean = abs(np.mean(votes))
        confidence = max(0, 1.0 - (vote_std / (vote_mean + 0.1)))
        
        # Entropy check (specialist diversity)
        vote_sum = sum(abs(v) for v in votes)
        vote_entropy = 0
        if vote_sum > 0:
            vote_probs = [abs(v)/vote_sum for v in votes]
            vote_entropy = -sum(p * np.log(p + 1e-10) for p in vote_probs if p > 0)
        
        round_predictions.append(snowfall_estimate)
        round_confidence.append(confidence)
        
        # KEY: EVOLVE THE MANIFOLD based on specialist consensus
        # This creates feedback and allows emergence
        consensus_direction = np.zeros(64)
        for s_id in specialists:
            try:
                principles = np.array(nodes[s_id]['state_dict']['principle_embeddings'])
                # Weight each principle by its alignment
                for pr in principles:
                    alignment = np.dot(evolving_manifold, pr)
                    consensus_direction += alignment * pr
            except:
                pass
        
        # Normalize consensus direction
        consensus_direction = consensus_direction / (np.linalg.norm(consensus_direction) + 1e-8)
        
        # Evolve manifold: blend original state with learned consensus
        # Alpha decreases each round (early rounds explore, late rounds converge)
        alpha = 0.3 * (8 - r) / 7  # 0.3 → 0.043 over rounds
        evolving_manifold = (1 - alpha) * evolving_manifold + alpha * consensus_direction
        
        # Re-normalize
        evolving_manifold = evolving_manifold / (np.linalg.norm(evolving_manifold) + 1e-8)
        
        # Preserve D41 anchor
        evolving_manifold[41] = D41_ANCHOR
        
        print(f"   Raw Consensus: {raw_consensus:.6f}")
        print(f"   D41 Stabilized: {stabilized_consensus:.6f}")
        print(f"   Cumulative Score: {cumulative_score:.6f}")
        print(f"   Snowfall Signal: {snowfall_estimate:.2f} inches")
        print(f"   Confidence: {confidence:.1%}")
        print(f"   Entropy: {vote_entropy:.4f} (target: {ENTROPY_TARGET:.4f})")
        print(f"   Manifold Evolution: α={alpha:.3f}")
    
    # Final integrated prediction
    weights = np.array(round_confidence)
    weights = weights / (weights.sum() + 1e-8)
    
    final_prediction = np.average(round_predictions, weights=weights)
    final_confidence = np.mean(round_confidence)
    
    # Stability analysis
    prediction_variance = np.var(round_predictions)
    convergence = round_predictions[-1] - round_predictions[0]
    
    # Entropy alignment check
    entropy_alignment = abs(vote_entropy - ENTROPY_TARGET)
    
    return {
        'prediction_inches': final_prediction,
        'confidence': final_confidence,
        'round_evolution': round_predictions,
        'variance': prediction_variance,
        'convergence': convergence,
        'anchor_stability': abs(D41_ANCHOR),
        'entropy_alignment': entropy_alignment,
        'final_manifold': evolving_manifold
    }

# ============================================================================
# MAIN EXECUTION: DELAWARE SNOWFALL TEST
# ============================================================================

def main():
    print("\n" + "="*60)
    print("🌨️  PANTHEON SNOWFALL PREDICTION SYSTEM")
    print("    Testing Emergent Geometric Weather Analysis")
    print("="*60)
    
    # Load Pantheon
    nodes = load_pantheon()
    if not nodes:
        print("❌ Failed to load Pantheon. Exiting.")
        return
    
    # Fetch Delaware weather data
    weather_data = fetch_noaa_weather()
    if not weather_data:
        print("❌ Failed to fetch weather data. Exiting.")
        return
    
    # Encode to 64D manifold
    print("\n🔬 Encoding atmospheric data to 64D manifold...")
    weather_vec = encode_snowfall_to_manifold(weather_data)
    print(f"✅ Vector norm: {np.linalg.norm(weather_vec):.4f}")
    print(f"✅ D41 Anchor: {weather_vec[41]:.8f}")
    
    # Run 7-round deliberation
    result = run_snowfall_deliberation(nodes, weather_vec)
    
    # Display results
    print("\n" + "="*60)
    print("🏆 PANTHEON VERDICT: DELAWARE SNOWFALL FORECAST")
    print("="*60)
    print(f"\n📊 PREDICTED SNOWFALL: {result['prediction_inches']:.2f} INCHES")
    print(f"🎯 Pantheon Confidence: {result['confidence']:.1%}")
    print(f"📈 Prediction Variance: {result['variance']:.4f}")
    print(f"🔄 7-Round Convergence: {result['convergence']:.2f} inches")
    print(f"⚓ D41 Anchor Stability: {result['anchor_stability']:.8f}")
    print(f"🌀 Entropy Alignment: {result['entropy_alignment']:.4f} from target")
    
    print("\n📉 Round-by-Round Evolution:")
    for i, pred in enumerate(result['round_evolution'], 1):
        bar = "█" * int(pred / 2)  # Visual bar
        print(f"   Round {i}: {pred:5.2f}\" {bar}")
    
    # Compare to official forecast
    print("\n📋 NOAA Official Forecast:")
    try:
        forecast_periods = weather_data['forecast']['properties']['periods']
        for period in forecast_periods[:3]:
            print(f"   {period['name']}: {period['shortForecast']}")
    except:
        print("   (Unable to parse official forecast)")
    
    print("\n" + "="*60)
    print("STATUS: D41 ANCHOR SECURE. ANALYSIS COMPLETE.")
    print("="*60)

if __name__ == "__main__":
    main()
