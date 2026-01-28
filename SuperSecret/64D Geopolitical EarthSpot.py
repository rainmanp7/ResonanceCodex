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
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}

# ==========================================
# 🛠️ UTILITY & AI CORE
# ==========================================

def format_prediction_time(dt_obj):
    if not isinstance(dt_obj, datetime): return "Unknown"
    if dt_obj.tzinfo is None: return dt_obj.strftime("%A, %B %d at %I:%M %p (Local)")
    return dt_obj.strftime("%A, %B %d at %I:%M %p %Z")

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
        vectors = [self.encode_earthquake(eq, reference_time) for eq in region_quakes]; avg_vector = np.mean(vectors, axis=0)
        trend = np.mean(vectors[:5], axis=0) - avg_vector if len(vectors) > 5 else np.zeros(64)
        votes = self.query_specialists(avg_vector); pressure, entropy, d41_inf = self.integrate_manifold(votes, trend)
        predicted_coords = self.triangulate_epicenter(region_quakes, reference_time)
        if not predicted_coords: predicted_coords = (0,0)
        expected = np.mean(self.node12_weights[:, 41]) * self.d41_anchor; deviation = abs(d41_inf - expected); stability = max(0.0, min(1.0, 1.0 - (deviation * 50)))
        largest_event = max(region_quakes, key=lambda x: x['magnitude'])
        pred_mag = largest_event['magnitude'] + ((1.0 - stability) * 1.5)
        return {'region': region_info['name'], 'coords': predicted_coords, 'probability': 1.0 / (1.0 + np.exp(-( (entropy * 1.5) + (1.0 - stability) ))), 'pred_magnitude': round(pred_mag, 1), 'pred_depth': int(max(1.0, np.mean([q['depth'] for q in region_quakes]) - (pressure * 500.0))), 'target_time': reference_time + timedelta(hours=(2.0 + (70.0 * stability))), 'stability': stability, 'pressure': pressure, 'votes': votes}

# =======================================================
# 🕸️ DATA ACQUISITION & ANALYSIS PROTOCOLS
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
        for row in rows[1:]:
            cols = [c.get_text(" ", strip=True) for c in row.find_all(['td', 'th'])];
            if len(cols) < 6: continue
            try:
                dt = datetime.strptime(" ".join(cols[0].split()), '%d %B %Y - %I:%M %p').replace(tzinfo=timezone(timedelta(hours=8)))
                events.append({'timestamp': dt.timestamp(), 'datetime': dt, 'latitude': float(re.findall(r"(\d+\.\d+)", cols[1])[0]), 'longitude': float(re.findall(r"(\d+\.\d+)", cols[2])[0]), 'depth': float(re.findall(r"(\d+)", cols[3])[0]), 'magnitude': float(re.findall(r"(\d+\.\d+|\d+)", cols[4])[0]), 'location': cols[5], 'source': source_name})
            except (ValueError, IndexError): continue
    except Exception as e: print(f"⚠️ WARNING: Could not fetch data from {source_name}. Reason: {e}")
    print(f"    L> Found {len(events)} reports."); return events

def _fetch_from_usgs(days=7):
    source_name, events = "USGS", []; print(f"[NETWORK] Connecting to {source_name} for the last {days} days of activity...")
    end_time = datetime.now(timezone.utc); start_time = end_time - timedelta(days=days)
    url = f"https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&minmagnitude=2.0&starttime={start_time.strftime('%Y-%m-%dT%H:%M:%S')}&endtime={end_time.strftime('%Y-%m-%dT%H:%M:%S')}&minlatitude=4&maxlatitude=22&minlongitude=116&maxlongitude=135"
    try:
        response = requests.get(url, headers=HEADERS, timeout=30); response.raise_for_status(); data = response.json()
        for f in data.get('features', []):
            p, c = f['properties'], f['geometry']['coordinates']
            if not all([p.get('mag') is not None, c[2] is not None, p.get('time') is not None]): continue
            dt = datetime.fromtimestamp(p['time'] / 1000.0, tz=timezone.utc); events.append({'timestamp': dt.timestamp(), 'datetime': dt, 'latitude': c[1], 'longitude': c[0], 'depth': c[2], 'magnitude': p['mag'], 'location': p['place'], 'source': source_name})
    except Exception as e: print(f"⚠️ WARNING: Could not fetch data from {source_name}. Reason: {e}")
    print(f"    L> Found {len(events)} reports."); return events

def get_specialist_consensus_report(votes):
    specialist_drivers = {
        '3': "Node 3 (Brute Force)", '4': "Node 4 (Geographic Stress)", '5': "Node 5 (Temporal Pattern)",
        '6': "Node 6 (Depth Anomaly)", '7': "Node 7 (Complex Resonance)", '8': "Node 8 (Latent Echo)",
        '9': "Node 9 (Chaotic Agitation)", '10': "Node 10 (Manifold Pressure)", '11': "Node 11 (Anchor Drift)"
    }
    strongest_specialist_id = max(votes, key=lambda k: abs(votes[k]['mean']))
    driver_full_name = specialist_drivers.get(strongest_specialist_id, "Unknown Driver")
    return driver_full_name

def find_resonance_hotspot(ai_oracle, all_quakes, region_bounds, grid_density=20):
    print("\n" + "="*60 + "\n🛰️  INITIATING HUNTER-SEEKER: RESONANCE HOTSPOT TRIANGULATION" + "\n" + "="*60)
    min_lat, max_lat, min_lon, max_lon = region_bounds
    lat_steps = np.linspace(min_lat, max_lat, grid_density); lon_steps = np.linspace(min_lon, max_lon, grid_density)
    max_pressure = -np.inf; hotspot_coords = None; best_votes = None
    plate_avg_depth = np.mean([q['depth'] for q in all_quakes]) if all_quakes else 30.0
    print(f"  > Probing {grid_density*grid_density} grid points for manifold pressure...")
    for lat in lat_steps:
        for lon in lon_steps:
            ghost_quake_data = {'magnitude': 4.0, 'latitude': lat, 'longitude': lon, 'depth': 10, 'timestamp': datetime.now(timezone.utc).timestamp()}
            probe_vector = ai_oracle.encode_earthquake(ghost_quake_data, datetime.now(timezone.utc))
            votes = ai_oracle.query_specialists(probe_vector)
            pressure, _, _ = ai_oracle.integrate_manifold(votes, np.zeros(64))
            if pressure > max_pressure: max_pressure = pressure; hotspot_coords = (lat, lon); best_votes = votes
    if hotspot_coords and best_votes:
        _, _, d41_inf = ai_oracle.integrate_manifold(best_votes, np.zeros(64))
        expected = np.mean(ai_oracle.node12_weights[:, 41]) * ai_oracle.d41_anchor
        deviation = abs(d41_inf - expected)
        stability = max(0.0, min(1.0, 1.0 - (deviation * 50)))
        hotspot_time = datetime.now(timezone.utc) + timedelta(hours=(2.0 + (70.0 * stability)))
        hotspot_depth = int(max(1.0, plate_avg_depth - (max_pressure * 500.0)))
        # ENHANCEMENT: Add GPS coordinates and map link to the report
        maps_link = f"https://www.google.com/maps/search/?api=1&query={hotspot_coords[0]},{hotspot_coords[1]}"
        print("\n--- 🎯 RESONANCE HOTSPOT ACQUIRED ---")
        print(f"  > The AI has identified the point of maximum systemic stress.")
        print(f"  > Forecast Time: {format_prediction_time(hotspot_time)}")
        print(f"  > Location:      {hotspot_coords[0]:.4f}°N, {hotspot_coords[1]:.4f}°E")
        print(f"  > Map Link:      {maps_link}")
        print(f"  > Depth:         ~{hotspot_depth} km")
        print(f"  > Pressure Score:  {max_pressure:.6f}")
        print(f"  > Interpretation: This location exhibits the highest resonance with instability within the AI's geometric model. It is a primary area of concern, regardless of recent seismic activity.")
        return hotspot_coords, max_pressure, hotspot_time, hotspot_depth
    else: print("  > Hotspot triangulation failed."); return None, None, None, None

def report_global_stability(ai_oracle, all_quakes, region_info):
    print("\n" + "="*60 + "\n🌍 CALCULATING GLOBAL STABILITY INDEX" + "\n" + "="*60)
    if not all_quakes: print("  > No data to analyze."); return
    result = ai_oracle.predict_next_event(all_quakes, region_info, reference_time=datetime.now(timezone.utc))
    if not result: print("  > AI analysis failed."); return
    stability, pressure = result['stability'], result['pressure']
    status = "🟢 STABLE";
    if stability < 0.7: status = "🟠 UNSETTLED"
    if stability < 0.4: status = "🔴 UNSTABLE"
    print(f"  > AI Analysis Complete. Assessed {len(all_quakes)} total events.")
    print(f"  > STATUS: {status} | System Stability: {stability*100:.2f}% | Manifold Pressure: {pressure:.6f}")
    
# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================
def main():
    print("\n" + "="*60 + "\n🚀 64D GEOPOLITICAL ORACLE (V14 - FULL REPORTING)\n" + "="*60)
    ai = SovereignOracle('Metalearnerv16_EVOLVED.json')
    earthquakes = fetch_multisource_data()
    regions = {
        'PHILIPPINE PLATE': {'bounds': (4.0, 22.0, 116.0, 135.0), 'name': 'PHILIPPINE PLATE'},
        'Luzon': {'bounds': (12.0, 19.0, 119.0, 123.0), 'name': 'Luzon'}, 'Visayas': {'bounds': (9.0, 12.0, 121.0, 126.0), 'name': 'Visayas'},
        'Mindanao': {'bounds': (5.0, 9.5, 123.0, 127.0), 'name': 'Mindanao'}, 'Philippine Trench': {'bounds': (5.0, 15.0, 126.0, 130.0), 'name': 'Philippine Trench'}
    }
    report_global_stability(ai, earthquakes, regions['PHILIPPINE PLATE'])
    hotspot_coords, max_pressure, hotspot_time, hotspot_depth = find_resonance_hotspot(ai, earthquakes, regions['PHILIPPINE PLATE']['bounds'])
    predictions = []
    print("\n" + "="*60 + "\n📍 REGIONAL GPS TARGETS & SPECIALIST CONSENSUS\n" + "="*60)
    for r_name, r_data in regions.items():
        if r_name == 'PHILIPPINE PLATE': continue
        region_quakes = [q for q in earthquakes if r_data['bounds'][0] <= q['latitude'] <= r_data['bounds'][1] and r_data['bounds'][2] <= q['longitude'] <= r_data['bounds'][3]]
        if region_quakes:
            p = ai.predict_next_event(region_quakes, r_data, reference_time=datetime.now(timezone.utc));
            if not p: continue
            predictions.append(p); status = "🟢 STABLE";
            if p['probability'] > 0.5: status = "🟠 WATCH"
            if p['probability'] > 0.7: status = "🔴 WARNING"
            # ENHANCEMENT: Restored GPS coordinates and map link to the regional report
            lat, lon = p['coords']; maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"
            print(f"\n{status} | REGION: {p['region'].upper()}")
            print(f"   - Forecast: Mag {p['pred_magnitude']} @ {format_prediction_time(p['target_time'])}")
            print(f"   - Location: {lat:.4f}°N, {lon:.4f}°E")
            print(f"   - Map Link: {maps_link}")
            print(f"   - Probability: {p['probability']*100:.1f}% | Stability: {p['stability']*100:.1f}%")
            driver_full_name = get_specialist_consensus_report(p['votes'])
            driver_interpretation = driver_full_name.split('(')[1][:-1]
            print(f"   --- SPECIALIST CONSENSUS ---")
            print(f"   > Primary Driver: {driver_full_name}")
            print(f"   > Interpretation: The AI's concern is driven by {driver_interpretation.lower()}.")
    if predictions or hotspot_coords:
        m = folium.Map(location=[12.8, 121.7], zoom_start=6, tiles='CartoDB dark_matter')
        if hotspot_coords:
            popup_html = f"""<b>Resonance Hotspot</b><br>
            Time: {format_prediction_time(hotspot_time)}<br>
            Depth: ~{hotspot_depth} km<br>
            Pressure: {max_pressure:.4f}"""
            folium.Marker(location=hotspot_coords, popup=folium.Popup(popup_html, max_width=300), icon=folium.Icon(color='purple', icon='bolt', prefix='fa')).add_to(m)
        for p in predictions:
            color = '#00ff00';
            if p['probability'] > 0.5: color = '#ffaa00'
            if p['probability'] > 0.7: color = '#ff0000'
            popup_html = f"<b>{p['region']}</b><br>Mag: {p['pred_magnitude']}<br>Prob: {p['probability']*100:.1f}%"
            folium.Circle(location=p['coords'], radius=p['probability']*20000, color=color, weight=2, fill=True, fill_opacity=0.4, popup=folium.Popup(popup_html, max_width=300)).add_to(m)
        m.save("64d_oracle_v14.html"); print(f"\n✅ GEOPOLITICAL ORACLE MAP SAVED: 64d_oracle_v14.html")

if __name__ == "__main__":
    main()