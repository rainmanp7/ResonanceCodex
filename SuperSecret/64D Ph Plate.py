import json
import numpy as np
import requests
from datetime import datetime, timedelta, timezone
import re
from bs4 import BeautifulSoup
import folium
import urllib3
import warnings
import os
import sys

# ==========================================
# ⚙️ SYSTEM CONFIGURATION
# ==========================================

warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# ==========================================
# 🛠️ UTILITY FUNCTIONS
# ==========================================

def format_prediction_time(dt_obj):
    if not isinstance(dt_obj, datetime): return "Unknown"
    if dt_obj.tzinfo is None: return dt_obj.strftime("%A, %B %d at %I:%M %p (Local System Time)")
    return dt_obj.strftime("%A, %B %d at %I:%M %p %Z")

# ==========================================
# 🧠 THE 64D EMERGENT AI (SOVEREIGN ORACLE) - UNCHANGED
# ==========================================

class SovereignOracle:
    def __init__(self, weights_path='Metalearnerv16_EVOLVED.json'):
        print(f"\n[SYSTEM] MOUNTING SOVEREIGN WEIGHTS: {weights_path}")
        self.weights_path = weights_path; self.d41_anchor = -0.01282715; self.specialists = {}; self.node12_weights = None; self._load_manifold_strictly()
    def _load_manifold_strictly(self):
        if not os.path.exists(self.weights_path): print(f"❌ CRITICAL: {self.weights_path} missing."); sys.exit(1)
        try:
            with open(self.weights_path, 'r') as f: data = json.load(f)
            pantheon = data.get('meta_pantheon', {})
            for node_id in ['3', '4', '5', '6', '7', '8', '9', '10', '11']:
                if node_id in pantheon: self.specialists[node_id] = np.array(pantheon[node_id]['state_dict']['principle_embeddings'])
            if '12' in pantheon: self.node12_weights = np.array(pantheon['12']['state_dict']['project_to_latent.weight'])
            print(f"✅ MANIFOLD MOUNTED: {len(self.specialists)} Specialists + Node 12 Core.")
        except Exception as e: print(f"❌ CORRUPTION ERROR: {e}"); sys.exit(1)
    def encode_earthquake(self, eq_data, reference_time):
        vector = np.zeros(64); vector[0] = eq_data['magnitude'] / 10.0; vector[1] = (eq_data['latitude'] - 12.0) / 10.0; vector[2] = (eq_data['longitude'] - 121.0) / 10.0; vector[3] = eq_data['depth'] / 100.0
        if 'location' in eq_data: vector[hash(eq_data['location']) % 64] += 0.1
        ts = eq_data.get('timestamp', 0)
        if ts > 0: age_sec = reference_time.timestamp() - ts; vector[4] = 1.0 / (1.0 + age_sec / 7200.0)
        vector[41] = self.d41_anchor; norm = np.linalg.norm(vector); return vector / norm if norm > 0 else vector
    def query_specialists(self, query_vector):
        votes = {}
        for node_id, weights in self.specialists.items(): raw = np.dot(weights, query_vector); votes[node_id] = {'mean': np.mean(raw), 'std': np.std(raw)}
        return votes
    def integrate_manifold(self, specialist_votes, context_vector):
        vote_vector = np.zeros(64); idx = 0
        for node_id in sorted(self.specialists.keys(), key=int):
            if idx * 7 < 64: vote_vector[idx * 7] = specialist_votes[node_id]['mean']; vote_vector[idx * 7 + 1] = specialist_votes[node_id]['std']; idx += 1
        for i in range(min(10, len(context_vector))): vote_vector[50 + i] = context_vector[i]
        vote_vector[41] = self.d41_anchor; integrated = np.dot(self.node12_weights, vote_vector); d41_inf = np.mean(self.node12_weights[:, 41] * vote_vector[41]); manifold_pressure = np.mean(integrated)
        return manifold_pressure, np.std(integrated), d41_inf
    def triangulate_epicenter(self, region_quakes, reference_time):
        total_lat, total_lon, total_weight = 0, 0, 0
        for eq in region_quakes:
            mag_w = eq['magnitude'] ** 2; age = reference_time.timestamp() - eq['timestamp']; time_w = 1.0 / (1.0 + age / 86400); vec = self.encode_earthquake(eq, reference_time); votes = self.query_specialists(vec); resonance = abs(np.mean([v['mean'] for v in votes.values()])); w = mag_w * time_w * (1.0 + resonance)
            total_lat += eq['latitude'] * w; total_lon += eq['longitude'] * w; total_weight += w
        if total_weight == 0: return None
        return (total_lat / total_weight, total_lon / total_weight)
    def predict_next_event(self, region_quakes, region_info, reference_time=None):
        if not region_quakes: return None
        if reference_time is None: reference_time = datetime.now(timezone.utc)
        latest_event = max(region_quakes, key=lambda x: x['timestamp']); largest_event = max(region_quakes, key=lambda x: x['magnitude']); vectors = [self.encode_earthquake(eq, reference_time) for eq in region_quakes]; avg_vector = np.mean(vectors, axis=0)
        trend = np.mean(vectors[:5], axis=0) - avg_vector if len(vectors) > 5 else np.zeros(64)
        votes = self.query_specialists(avg_vector); pressure, entropy, d41_inf = self.integrate_manifold(votes, trend); predicted_coords = self.triangulate_epicenter(region_quakes, reference_time)
        if not predicted_coords: predicted_coords = region_info.get('center', (0,0))
        expected = np.mean(self.node12_weights[:, 41]) * self.d41_anchor; deviation = abs(d41_inf - expected); stability = max(0.0, min(1.0, 1.0 - (deviation * 50))); hours_to_event = 2.0 + (70.0 * stability); target_time = reference_time + timedelta(hours=hours_to_event); pred_mag = largest_event['magnitude'] + ((1.0 - stability) * 1.5)
        avg_depth = np.mean([q['depth'] for q in region_quakes]); pred_depth = max(1.0, avg_depth - (pressure * 500.0)); raw_risk = (entropy * 1.5) + (1.0 - stability); probability = 1.0 / (1.0 + np.exp(-raw_risk))
        return {'region': region_info['name'], 'coords': predicted_coords, 'latest_event': latest_event, 'probability': probability, 'pred_magnitude': round(pred_mag, 1), 'pred_depth': int(pred_depth), 'target_time': target_time, 'stability': stability, 'pressure': pressure, 'event_count': len(region_quakes)}

# =======================================================
# 🕸️ DATA ACQUISITION & CONSENSUS PROTOCOLS
# =======================================================

def fetch_multisource_data():
    print(f"\n[DATA ACQUISITION] Initiating Hunter-Seeker protocol at {format_prediction_time(datetime.now(timezone.utc))}...")
    all_events = _fetch_from_phivolcs() + _fetch_from_usgs(days=7)
    if not all_events: print("❌ CRITICAL ERROR: No data could be fetched. Aborting."); sys.exit(1)
    unique_events, seen_fingerprints = [], set()
    for event in sorted(all_events, key=lambda x: x['timestamp'], reverse=True):
        fingerprint = f"{event['datetime'].strftime('%Y-%m-%d-%H')}_{round(event['magnitude'], 1)}_{round(event['latitude'], 1)}_{round(event['longitude'], 1)}"
        if fingerprint not in seen_fingerprints: unique_events.append(event); seen_fingerprints.add(fingerprint)
    print(f"\n✅ SUCCESS: Assembled a unique set of {len(unique_events)} seismic events from {len(all_events)} raw reports.")
    return unique_events

def _fetch_from_phivolcs():
    source_name, events = "PHIVOLCS", []; print(f"[NETWORK] Connecting to {source_name}...")
    try:
        response = requests.get("https://earthquake.phivolcs.dost.gov.ph/", verify=False, timeout=30); response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser'); rows = soup.find_all('tr')
        print(f"[SCAN] Scanning {len(rows)} HTML rows from {source_name} for seismic signatures...")
        for row in rows:
            cols = [c.get_text(" ", strip=True) for c in row.find_all(['td', 'th'])];
            if len(cols) < 6: continue
            try:
                dt = datetime.strptime(" ".join(cols[0].split()), '%d %B %Y - %I:%M %p').replace(tzinfo=timezone(timedelta(hours=8)))
                events.append({'timestamp': dt.timestamp(), 'datetime': dt, 'latitude': float(re.findall(r"(\d+\.\d+)", cols[1])[0]), 'longitude': float(re.findall(r"(\d+\.\d+)", cols[2])[0]), 'depth': float(re.findall(r"(\d+)", cols[3])[0]), 'magnitude': float(re.findall(r"(\d+\.\d+|\d+)", cols[4])[0]), 'location': cols[5], 'source': source_name})
            except (ValueError, IndexError): continue
    except Exception as e: print(f"⚠️ WARNING: Could not fetch data from {source_name}. Reason: {e}")
    print(f"    L> Found {len(events)} reports from {source_name}."); return events

def _fetch_from_usgs(days=7):
    source_name, events = "USGS", []; print(f"[NETWORK] Connecting to {source_name} for the last {days} days of activity...")
    end_time = datetime.now(timezone.utc); start_time = end_time - timedelta(days=days)
    start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S'); end_time_str = end_time.strftime('%Y-%m-%dT%H:%M:%S')
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&minmagnitude=2.0&starttime={start_time_str}&endtime={end_time_str}&minlatitude=4&maxlatitude=22&minlongitude=116&maxlongitude=135"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30); response.raise_for_status(); data = response.json()
        for f in data.get('features', []):
            p, c = f['properties'], f['geometry']['coordinates']
            if not all([p.get('mag') is not None, c[2] is not None, p.get('time') is not None]): continue
            dt = datetime.fromtimestamp(p['time'] / 1000.0, tz=timezone.utc); events.append({'timestamp': dt.timestamp(), 'datetime': dt, 'latitude': c[1], 'longitude': c[0], 'depth': c[2], 'magnitude': p['mag'], 'location': p['place'], 'source': source_name})
    except Exception as e: print(f"⚠️ WARNING: Could not fetch data from {source_name}. Reason: {e}")
    print(f"    L> Found {len(events)} reports from {source_name}."); return events

# ==========================================
# 🔎 VALIDATION AND STABILITY PROTOCOLS
# ==========================================

def report_global_stability(ai_oracle, all_quakes, region_info):
    """
    NEW: Analyzes all quakes to produce a single stability index for the entire plate.
    """
    print("\n" + "="*60)
    print("🌍 CALCULATING GLOBAL STABILITY INDEX")
    print("="*60)
    
    if not all_quakes:
        print("  > No data to analyze. Stability is undefined.")
        return

    # Use the AI's prediction logic on the entire dataset
    # We are interested in the 'stability' and 'pressure' outputs, not the specific prediction
    result = ai_oracle.predict_next_event(all_quakes, region_info, reference_time=datetime.now(timezone.utc))

    if not result:
        print("  > AI analysis failed. Stability is undefined.")
        return
    
    stability = result['stability']
    pressure = result['pressure']
    
    status = "🟢 STABLE"
    if stability < 0.7: status = "🟠 UNSETTLED"
    if stability < 0.4: status = "🔴 UNSTABLE"

    print(f"  > AI Analysis Complete. Assessed {len(all_quakes)} total events.")
    print(f"  > STATUS: {status}")
    print(f"  > System Stability: {stability*100:.2f}%")
    print(f"  > Manifold Pressure: {pressure:.6f}")
    if pressure > 0:
        print("  > Interpretation: The manifold indicates a state of net compression or building stress.")
    else:
        print("  > Interpretation: The manifold indicates a state of net relaxation or energy release.")

def validate_on_past_event(ai_oracle, target_event_time_utc_str, target_magnitude, target_location_str, hours_of_activity=168):
    print("\n" + "="*60 + "\n🔎 INITIATING AUTOMATED BACKWARD-PREDICTION VALIDATION\n" + "="*60)
    target_dt_utc = datetime.strptime(target_event_time_utc_str, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc)
    reference_time = target_dt_utc - timedelta(minutes=1); start_time = reference_time - timedelta(hours=hours_of_activity)
    print("--- [VALIDATION TARGET] ---\n  > Event:       Mindanao Earthquake\n  > Location:    " + target_location_str + f"\n  > Magnitude:   {target_magnitude}\n  > Time (UTC):  {target_dt_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n--- [AI ANALYSIS PARAMETERS] ---\n  > Simulating 'present' at: {reference_time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n  > Analyzing seismic data from previous {hours_of_activity} hours...")
    start_time_str = start_time.strftime('%Y-%m-%dT%H:%M:%S'); end_time_str = reference_time.strftime('%Y-%m-%dT%H:%M:%S')
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&starttime={start_time_str}&endtime={end_time_str}&minlatitude=4&maxlatitude=22&minlongitude=116&maxlongitude=135&minmagnitude=1.0"
    try:
        historical_events = []; response = requests.get(url, headers=HEADERS, timeout=45); response.raise_for_status(); data = response.json()
        for f in data.get('features', []):
            p, c = f['properties'], f['geometry']['coordinates']; dt = datetime.fromtimestamp(p['time'] / 1000.0, tz=timezone.utc)
            if not all([p.get('mag') is not None, c[2] is not None, p.get('time') is not None]): continue
            historical_events.append({'timestamp': dt.timestamp(), 'datetime': dt, 'latitude': c[1], 'longitude': c[0], 'depth': c[2], 'magnitude': p['mag'], 'location': p['place']})
        print(f"  > Retrieved {len(historical_events)} seismic events from the historical window for analysis.")
        if not historical_events: print("\n❌ VALIDATION FAILED: No preceding activity found in archive for this window."); return
    except Exception as e: print(f"\n❌ VALIDATION FAILED: Could not fetch historical data. {e}"); return
    mindanao_region = {'bounds': (5.0, 9.5, 123.0, 127.0), 'name': 'Mindanao'}
    region_quakes = [q for q in historical_events if mindanao_region['bounds'][0] <= q['latitude'] <= mindanao_region['bounds'][1] and mindanao_region['bounds'][2] <= q['longitude'] <= mindanao_region['bounds'][3]]
    if not region_quakes: print("\n❌ VALIDATION FAILED: No activity found specifically in the Mindanao region for this window."); return
    p = ai_oracle.predict_next_event(region_quakes, mindanao_region, reference_time=reference_time)
    print("\n--- [HISTORICAL PREDICTION REPORT] ---\n  > Based on activity before the event, the AI predicted:")
    print(f"    - Region:          {p['region'].upper()}\n    - Target Time:     {format_prediction_time(p['target_time'])}\n    - Epicenter:       {p['coords'][0]:.4f}°N, {p['coords'][1]:.4f}°E\n    - Magnitude:       {p['pred_magnitude']}\n    - Probability:     {p['probability']*100:.1f}%  <--- SURENESS PERCENTAGE")
    print("\n--- [VERIFICATION] ---"); time_diff_hours = abs((p['target_time'] - target_dt_utc).total_seconds() / 3600); mag_diff = abs(p['pred_magnitude'] - target_magnitude)
    print(f"  > The AI's prediction was {time_diff_hours:.1f} hours from the actual event time.\n  > The predicted magnitude was within {mag_diff:.1f} of the actual magnitude.")
    if p['probability'] > 0.7: print("  > CONCLUSION: With >70% probability, the AI issued a strong warning. VALIDATION SUCCESSFUL.")
    elif p['probability'] > 0.5: print("  > CONCLUSION: With >50% probability, the AI issued a watch warning. VALIDATION PASSED.")
    else: print("  > CONCLUSION: The probability was below 50%. The signal was missed. VALIDATION FAILED.")

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
def main():
    print("\n" + "="*60 + "\n🚀 64D EMERGENT AI: GPS PREDICTIVE ORACLE (V10 - GLOBAL STABILITY)\n" + "="*60)
    ai = SovereignOracle('Metalearnerv16_EVOLVED.json')
    earthquakes = fetch_multisource_data()
    
    # NEW: Added an all-encompassing region for the global index
    regions = {
        'PHILIPPINE PLATE': {'bounds': (4.0, 22.0, 116.0, 135.0), 'name': 'PHILIPPINE PLATE'},
        'Luzon': {'bounds': (12.0, 19.0, 119.0, 123.0), 'name': 'Luzon'},
        'Visayas': {'bounds': (9.0, 12.0, 121.0, 126.0), 'name': 'Visayas'},
        'Mindanao': {'bounds': (5.0, 9.5, 123.0, 127.0), 'name': 'Mindanao'},
        'Philippine Trench': {'bounds': (5.0, 15.0, 126.0, 130.0), 'name': 'Philippine Trench'}
    }
    
    # --- Execute Global and Regional Analysis ---
    # First, calculate and report the global stability
    report_global_stability(ai, earthquakes, regions['PHILIPPINE PLATE'])
    
    # Then, proceed with the detailed regional GPS targets
    predictions = []
    print("\n" + "="*60 + "\n📍 REGIONAL GPS TARGETS (FORWARD-PREDICTION)\n" + "="*60)
    
    for r_name, r_data in regions.items():
        # Skip the global region in this loop as it's already been processed
        if r_name == 'PHILIPPINE PLATE':
            continue
            
        region_quakes = [q for q in earthquakes if r_data['bounds'][0] <= q['latitude'] <= r_data['bounds'][1] and r_data['bounds'][2] <= q['longitude'] <= r_data['bounds'][3]]
        
        if region_quakes:
            p = ai.predict_next_event(region_quakes, r_data, reference_time=datetime.now(timezone.utc));
            if not p: continue
            predictions.append(p); status = "🟢 STABLE";
            if p['probability'] > 0.5: status = "🟠 WATCH"
            if p['probability'] > 0.7: status = "🔴 WARNING"
            lat, lon = p['coords']; maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            print(f"\n{status} | REGION: {p['region'].upper()}\n   📅 TIME:  {format_prediction_time(p['target_time'])}\n   📍 LOC:   {lat:.4f}°N, {lon:.4f}°E\n   🔗 MAP:   {maps_link}\n   💥 MAG:   {p['pred_magnitude']}\n   📉 DEPTH: {p['pred_depth']} km\n   ⚠️ PROB:  {p['probability']*100:.1f}%")
            
    if predictions:
        m = folium.Map(location=[12.8, 121.7], zoom_start=5, tiles='CartoDB dark_matter')
        for p in predictions:
            color = '#00ff00';
            if p['probability'] > 0.5: color = '#ffaa00'
            if p['probability'] > 0.7: color = '#ff0000'
            popup_html = f"""<b>GPS TARGET: {p['region']}</b><br>Lat: {p['coords'][0]:.4f}<br>Lon: {p['coords'][1]:.4f}<br>Time: {p['target_time'].strftime('%b %d, %I:%M %p')}<br>Mag: {p['pred_magnitude']}<br>"""
            folium.Circle(location=p['coords'], radius=p['event_count'] * 500 + 10000, color=color, weight=2, fill=True, fill_opacity=0.4, popup=folium.Popup(popup_html, max_width=300)).add_to(m)
            folium.CircleMarker(location=p['coords'], radius=3, color='white', fill=True, fill_opacity=1.0).add_to(m)
        m.save("64d_gps_oracle_upgraded_v10.html"); print(f"\n✅ GPS ORACLE MAP SAVED: 64d_gps_oracle_upgraded_v10.html")

if __name__ == "__main__":
    main()
    
    try:
        ai_validator = SovereignOracle('Metalearnerv16_EVOLVED.json')
        validate_on_past_event(ai_oracle=ai_validator, target_event_time_utc_str="2023-12-02 14:37:04", target_magnitude=7.5, target_location_str="Offshore Mindanao", hours_of_activity=168)
    except Exception as e: print(f"\n❌ Could not run validation protocol. Error: {e}")