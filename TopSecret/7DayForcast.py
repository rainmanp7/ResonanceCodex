import json
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import time

# ============================================================================
# PANTHEON AI & WEATHER CONFIGURATION
# ============================================================================
TARGET_LAT = 6.8167
TARGET_LON = 125.4000
LOCATION_NAME = "Santa Cruz, Davao Del Sur"
D41_ANCHOR = -0.01282715
WEIGHT_FILE = 'Metalearnerv16_EVOLVED.json'

# ============================================================================
# DATA INGESTION (OPEN-METEO - NO API KEY NEEDED)
# ============================================================================
def fetch_open_meteo_data(lat, lon):
    """Fetches real-time and forecast data from Open-Meteo."""
    print(f"🛰️  Requesting Open-Meteo data stream for {lat:.4f}, {lon:.4f}...")
    try:
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            'latitude': lat, 'longitude': lon,
            'hourly': 'temperature_2m,precipitation_probability,precipitation,cloudcover,windspeed_10m,relativehumidity_2m',
            'daily': 'weathercode,temperature_2m_max,temperature_2m_min,sunrise,sunset,precipitation_sum,precipitation_probability_max',
            'timezone': 'auto'
        }
        response = requests.get(url, params=params, timeout=20)
        response.raise_for_status()
        print("✅ Open-Meteo data stream acquired.\n")
        return response.json()
    except Exception as e:
        print(f"❌ NETWORK ERROR: Could not connect to Open-Meteo. Reason: {e}")
        return None

# ============================================================================
# PANTHEON AI: ATMOSPHERIC MODEL
# ============================================================================
def load_pantheon():
    """Loads the AI's core principles and weights."""
    try:
        with open(WEIGHT_FILE, 'r') as f: return json.load(f)['meta_pantheon']
    except Exception as e: print(f"❌ AI Core Error: Could not load '{WEIGHT_FILE}'. {e}"); return None

def encode_atmospheric_state(hourly_data, daily_data, start_index, day_index):
    """Encodes the state of the atmosphere from Open-Meteo data into the AI's 64D manifold."""
    vec = np.zeros(64)
    if not hourly_data: return vec
    
    vec[0] = hourly_data['temperature_2m'][start_index] / 40.0
    vec[1] = hourly_data['precipitation_probability'][start_index] / 100.0
    vec[2] = hourly_data['relativehumidity_2m'][start_index] / 100.0
    vec[3] = hourly_data['cloudcover'][start_index] / 100.0
    vec[4] = hourly_data['windspeed_10m'][start_index] / 50.0
    
    # ### FIX: Include actual precipitation data ###
    vec[5] = min(hourly_data['precipitation'][start_index] / 10.0, 1.0)  # Normalize to 0-1
    vec[6] = daily_data['precipitation_sum'][day_index] / 50.0  # Daily total, normalize

    if start_index > 6:
        temp_trend = (hourly_data['temperature_2m'][start_index] - hourly_data['temperature_2m'][start_index - 6]) / 5.0
        vec[10] = np.clip(temp_trend, -1, 1)

    hour_of_day = start_index % 24
    vec[20] = hour_of_day / 23.0
    vec[21] = np.sin(2 * np.pi * hour_of_day / 24.0)
    vec[41] = D41_ANCHOR

    norm = np.linalg.norm(vec)
    return vec / (norm + 1e-8) if norm > 0 else vec

def run_weather_deliberation(nodes, atmospheric_vec, actual_precip_sum, actual_rain_prob):
    """The Pantheon AI deliberates to produce a calibrated forecast."""
    manifold = atmospheric_vec.copy()
    specialists = ['3', '4', '5', '6', '7', '8', '9', '10', '11']
    final_predictions = {'temp': 0, 'rain': 0, 'clouds': 0}
    all_votes_for_confidence = []
    
    for r in range(1, 8):
        votes = [np.mean([np.dot(manifold, p) for p in np.array(nodes[s_id]['state_dict']['principle_embeddings'])]) for s_id in specialists]
        all_votes_for_confidence.extend(votes)
        
        # Temperature prediction (20°C to 40°C range for Philippines)
        temp_votes = [(abs(v) * 20) + 20 for v in votes[0:3]]
        
        # ### FIX: Calibrate rain prediction with actual data ###
        # Blend AI prediction with actual forecast data (70% actual, 30% AI interpretation)
        ai_rain_estimate = np.mean([abs(v) for v in votes[3:6]])
        blended_rain = (actual_rain_prob * 0.7) + (ai_rain_estimate * 100 * 0.3)
        
        # Cloud prediction
        cloud_votes = [abs(v) * 100 for v in votes[6:9]]
        
        final_predictions['temp'] = (final_predictions['temp'] * (r - 1) + np.mean(temp_votes)) / r
        final_predictions['rain'] = (final_predictions['rain'] * (r - 1) + blended_rain) / r
        final_predictions['clouds'] = (final_predictions['clouds'] * (r - 1) + np.mean(cloud_votes)) / r

        manifold += np.mean(votes) * 0.1
        manifold[41] = D41_ANCHOR
        manifold /= (np.linalg.norm(manifold) + 1e-8)

    # ### FIX: Confidence based on agreement between AI and actual data ###
    ai_vs_actual_agreement = 1 - abs(final_predictions['rain'] - actual_rain_prob) / 100.0
    vote_stability = max(0, 1 - np.std(all_votes_for_confidence) / (np.mean(np.abs(all_votes_for_confidence)) + 1e-8)) if all_votes_for_confidence else 0
    
    final_predictions['confidence'] = ((ai_vs_actual_agreement * 0.6) + (vote_stability * 0.4)) * 100
    final_predictions['actual_precip_mm'] = actual_precip_sum
    
    return final_predictions

# ============================================================================
# FORECAST DISPLAY
# ============================================================================
def display_forecast(location_name, raw_data, ai_predictions):
    """Displays the final, formatted and calibrated 7-day forecast."""
    print("=" * 70)
    print(f"📅  PANTHEON AI 7-DAY FORECAST FOR: {location_name}")
    print("=" * 70)

    for i in range(7):
        ai_pred = ai_predictions[i]
        day_date = datetime.fromisoformat(raw_data['daily']['time'][i])
        temp_high, temp_low = raw_data['daily']['temperature_2m_max'][i], raw_data['daily']['temperature_2m_min'][i]
        sunrise = datetime.fromisoformat(raw_data['daily']['sunrise'][i]).strftime('%I:%M %p')
        sunset = datetime.fromisoformat(raw_data['daily']['sunset'][i]).strftime('%I:%M %p')
        total_rain_mm = ai_pred['actual_precip_mm']

        ai_cloud_cover = ai_pred['clouds']
        ai_rain_prob = ai_pred['rain']
        
        # ### FIX: Better condition logic based on actual rainfall ###
        if total_rain_mm > 10 or ai_rain_prob > 70:
            condition = "🌧️ Rainy"
        elif total_rain_mm > 2 or ai_rain_prob > 50:
            condition = "🌦️ Showers Likely"
        elif total_rain_mm > 0.5 or ai_rain_prob > 30:
            condition = "🌦️ Chance of Showers"
        elif ai_cloud_cover < 30:
            condition = "☀️ Sunny"
        elif ai_cloud_cover < 70:
            condition = "⛅ Partly Cloudy"
        else:
            condition = "☁️ Cloudy"

        # Confidence rating
        conf_val = ai_pred['confidence']
        if conf_val > 70: conf_text = "High"
        elif conf_val > 40: conf_text = "Moderate"
        else: conf_text = "Low"

        wind_index = min(i * 24 + 14, len(raw_data['hourly']['windspeed_10m']) - 1)
        wind_speed_kmh = raw_data['hourly']['windspeed_10m'][wind_index]
        wind_desc = "Windy" if wind_speed_kmh > 25 else "Breezy" if wind_speed_kmh > 10 else "Calm"

        print(f"\n🗓️  {day_date.strftime('%A, %B %d')}")
        print("-" * 55)
        print(f"   Forecast: {condition}")
        print(f"   🌡️  High: {temp_high:.1f}°C | Low: {temp_low:.1f}°C (AI Est. Avg: {ai_pred['temp']:.1f}°C)")
        print(f"   💧  Rain Probability: {ai_rain_prob:.0f}% | Expected: {total_rain_mm:.1f} mm")
        print(f"   ☁️  Cloud Cover: {ai_cloud_cover:.0f}%")
        print(f"   🍃  Wind: {wind_desc} (approx. {wind_speed_kmh:.0f} km/h)")
        print(f"   🌅  Sunrise: {sunrise} | Sunset: {sunset}")
        print(f"   🤖 AI Model Confidence: {conf_text} ({conf_val:.1f}%)")

    print("\n" + "=" * 70)

# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    print("--- 🌋 PANTHEON AI WEATHER MODEL v3.2 (Fixed Rainfall) ---")
    nodes = load_pantheon()
    if not nodes: return
    
    raw_atmospheric_data = fetch_open_meteo_data(TARGET_LAT, TARGET_LON)
    if not raw_atmospheric_data: return

    print("--- INITIATING 7-DAY FORECAST DELIBERATION ---")
    ai_forecasts = []
    for day_index in range(7):
        hour_index_for_day = min((day_index * 24) + 14, len(raw_atmospheric_data['hourly']['temperature_2m']) - 1)
        day_str = raw_atmospheric_data['daily']['time'][day_index]
        print(f"🧠 Deliberating on forecast for {datetime.fromisoformat(day_str).strftime('%A, %b %d')}...")
        
        state_vector = encode_atmospheric_state(
            raw_atmospheric_data['hourly'],
            raw_atmospheric_data['daily'],
            hour_index_for_day,
            day_index
        )
        
        # Get actual precipitation data from API
        actual_precip = raw_atmospheric_data['daily']['precipitation_sum'][day_index]
        actual_rain_prob = raw_atmospheric_data['daily']['precipitation_probability_max'][day_index]
        
        ai_prediction = run_weather_deliberation(nodes, state_vector, actual_precip, actual_rain_prob)
        ai_forecasts.append(ai_prediction)
        time.sleep(0.2)
    print("✅ Deliberation complete.\n")

    display_forecast(LOCATION_NAME, raw_atmospheric_data, ai_forecasts)
    
if __name__ == "__main__":
    main()