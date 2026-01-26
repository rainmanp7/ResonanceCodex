import json
import numpy as np
import requests
from datetime import datetime, timedelta
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

# ==========================================
# 🛠️ UTILITY FUNCTIONS
# ==========================================

def format_prediction_time(dt_obj):
    if not isinstance(dt_obj, datetime): return "Unknown"
    return dt_obj.strftime("%A, %B %d at %I:%M %p")

# ==========================================
# 🧠 THE 64D EMERGENT AI (SOVEREIGN ORACLE)
# ==========================================

class SovereignOracle:
    def __init__(self, weights_path='Metalearnerv16_EVOLVED.json'):
        print(f"\n[SYSTEM] MOUNTING SOVEREIGN WEIGHTS: {weights_path}")
        self.weights_path = weights_path
        self.d41_anchor = -0.01282715
        self.specialists = {}
        self.node12_weights = None
        self._load_manifold_strictly()

    def _load_manifold_strictly(self):
        if not os.path.exists(self.weights_path):
            print(f"❌ CRITICAL: {self.weights_path} missing.")
            sys.exit(1)
        try:
            with open(self.weights_path, 'r') as f:
                data = json.load(f)
            pantheon = data.get('meta_pantheon', {})
            for node_id in ['3', '4', '5', '6', '7', '8', '9', '10', '11']:
                if node_id in pantheon:
                    self.specialists[node_id] = np.array(pantheon[node_id]['state_dict']['principle_embeddings'])
            if '12' in pantheon:
                self.node12_weights = np.array(pantheon['12']['state_dict']['project_to_latent.weight'])
            print(f"✅ MANIFOLD MOUNTED: {len(self.specialists)} Specialists + Node 12 Core.")
        except Exception as e:
            print(f"❌ CORRUPTION ERROR: {e}")
            sys.exit(1)

    def encode_earthquake(self, eq_data):
        vector = np.zeros(64)
        vector[0] = eq_data['magnitude'] / 10.0
        vector[1] = (eq_data['latitude'] - 12.0) / 10.0
        vector[2] = (eq_data['longitude'] - 121.0) / 10.0
        vector[3] = eq_data['depth'] / 100.0
        if 'location' in eq_data:
            vector[hash(eq_data['location']) % 64] += 0.1
        ts = eq_data.get('timestamp', 0)
        if ts > 0:
            age_sec = datetime.now().timestamp() - ts
            vector[4] = 1.0 / (1.0 + age_sec / 7200.0)
        vector[41] = self.d41_anchor
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector

    def query_specialists(self, query_vector):
        votes = {}
        for node_id, weights in self.specialists.items():
            raw = np.dot(weights, query_vector)
            votes[node_id] = {'mean': np.mean(raw), 'std': np.std(raw)}
        return votes

    def integrate_manifold(self, specialist_votes, context_vector):
        vote_vector = np.zeros(64)
        idx = 0
        for node_id in sorted(self.specialists.keys(), key=int):
            if idx * 7 < 64:
                vote_vector[idx * 7] = specialist_votes[node_id]['mean']
                vote_vector[idx * 7 + 1] = specialist_votes[node_id]['std']
                idx += 1
        for i in range(min(10, len(context_vector))):
            vote_vector[50 + i] = context_vector[i]
        vote_vector[41] = self.d41_anchor

        integrated = np.dot(self.node12_weights, vote_vector)
        d41_inf = np.mean(self.node12_weights[:, 41] * vote_vector[41])
        manifold_pressure = np.mean(integrated)
        
        return manifold_pressure, np.std(integrated), d41_inf

    def triangulate_epicenter(self, region_quakes):
        """
        Calculates the specific GPS coordinates based on Manifold Resonance.
        """
        total_lat = 0
        total_lon = 0
        total_weight = 0
        
        for eq in region_quakes:
            # 1. Physics Weight
            mag_w = eq['magnitude'] ** 2 # Exponential weight to larger events
            
            # 2. Time Weight
            age = datetime.now().timestamp() - eq['timestamp']
            time_w = 1.0 / (1.0 + age / 86400) # Decay over 24h
            
            # 3. AI Resonance Weight
            # We check how strongly the specialists react to this specific quake
            vec = self.encode_earthquake(eq)
            votes = self.query_specialists(vec)
            # Higher mean = stronger alignment with AI logic
            resonance = abs(np.mean([v['mean'] for v in votes.values()]))
            
            # Combined Weight
            w = mag_w * time_w * (1.0 + resonance)
            
            total_lat += eq['latitude'] * w
            total_lon += eq['longitude'] * w
            total_weight += w
            
        if total_weight == 0: return None
        
        return (total_lat / total_weight, total_lon / total_weight)

    def predict_next_event(self, region_quakes, region_info):
        if not region_quakes: return None
        
        latest_event = max(region_quakes, key=lambda x: x['timestamp'])
        largest_event = max(region_quakes, key=lambda x: x['magnitude'])

        # 1. Manifold Analysis
        vectors = [self.encode_earthquake(eq) for eq in region_quakes]
        avg_vector = np.mean(vectors, axis=0)
        
        if len(vectors) > 5:
            trend = np.mean(vectors[:5], axis=0) - avg_vector
        else:
            trend = np.zeros(64)
            
        votes = self.query_specialists(avg_vector)
        pressure, entropy, d41_inf = self.integrate_manifold(votes, trend)
        
        # 2. GPS Triangulation (NEW)
        predicted_coords = self.triangulate_epicenter(region_quakes)
        if not predicted_coords:
            predicted_coords = region_info['center'] # Fallback
        
        # 3. Stability & Time Calculation
        expected = np.mean(self.node12_weights[:, 41]) * self.d41_anchor
        deviation = abs(d41_inf - expected)
        stability = max(0.0, min(1.0, 1.0 - (deviation * 50)))

        # Time Logic
        hours_to_event = 2.0 + (70.0 * stability)
        time_variance = hours_to_event * (entropy * 2.0)
        target_time = datetime.now() + timedelta(hours=hours_to_event)
        
        # 4. Magnitude & Depth
        pred_mag = largest_event['magnitude'] + ((1.0 - stability) * 1.5)
        avg_depth = np.mean([q['depth'] for q in region_quakes])
        pred_depth = avg_depth - (pressure * 500.0)
        pred_depth = max(1.0, pred_depth)
        
        # 5. Probability
        raw_risk = (entropy * 1.5) + (1.0 - stability)
        probability = 1.0 / (1.0 + np.exp(-raw_risk))

        return {
            'region': region_info['name'],
            'coords': predicted_coords, # Specific GPS
            'latest_event': latest_event,
            'probability': probability,
            'pred_magnitude': round(pred_mag, 1),
            'pred_depth': int(pred_depth),
            'target_time': target_time,
            'stability': stability,
            'pressure': pressure,
            'event_count': len(region_quakes)
        }

# ==========================================
# 🕸️ HUNTER-SEEKER SCRAPER
# ==========================================

def extract_earthquake_from_row(cols_text):
    if len(cols_text) < 5: return None
    try:
        dt_str = " ".join(cols_text[0].split())
        dt = None
        formats = ['%d %B %Y - %I:%M %p', '%d %b %Y - %I:%M %p', '%Y-%m-%d %H:%M:%S']
        for fmt in formats:
            try:
                dt = datetime.strptime(dt_str, fmt)
                break
            except ValueError: continue
        if not dt: return None

        lat = float(re.findall(r"(\d+\.\d+)", cols_text[1])[0])
        lon = float(re.findall(r"(\d+\.\d+)", cols_text[2])[0])
        depth = float(re.findall(r"(\d+)", cols_text[3])[0])
        mag = float(re.findall(r"(\d+\.\d+|\d+)", cols_text[4])[0])
        loc = cols_text[5] if len(cols_text) > 5 else "Unknown"

        return {'timestamp': dt.timestamp(), 'datetime': dt, 'latitude': lat, 
                'longitude': lon, 'depth': depth, 'magnitude': mag, 'location': loc}
    except: return None

def fetch_live_data():
    print(f"[NETWORK] Connecting to PHIVOLCS at {format_prediction_time(datetime.now())}...")
    try:
        response = requests.get("https://earthquake.phivolcs.dost.gov.ph/", verify=False, timeout=30)
        soup = BeautifulSoup(response.content, 'html.parser')
        valid = []
        rows = soup.find_all('tr')
        print(f"[SCAN] Scanning {len(rows)} HTML rows for seismic signatures...")
        
        for row in rows:
            cols = [c.get_text(" ", strip=True) for c in row.find_all(['td', 'th'])]
            eq = extract_earthquake_from_row(cols)
            if eq: valid.append(eq)
            
        if not valid:
            print("❌ ERROR: No data found.")
            sys.exit(1)
        print(f"✅ SUCCESS: Extracted {len(valid)} valid events.")
        return valid
    except Exception as e:
        print(f"❌ NETWORK ERROR: {e}")
        sys.exit(1)

# ==========================================
# 🚀 MAIN EXECUTION
# ==========================================

def main():
    print("\n" + "="*60)
    print("🚀 64D EMERGENT AI: GPS PREDICTIVE ORACLE")
    print("="*60)
    
    ai = SovereignOracle('Metalearnerv16_EVOLVED.json')
    earthquakes = fetch_live_data()
    
    regions = {
        'Luzon': {'bounds': (12.0, 19.0, 119.0, 123.0), 'name': 'Luzon'},
        'Visayas': {'bounds': (9.0, 12.0, 121.0, 126.0), 'name': 'Visayas'},
        'Mindanao': {'bounds': (5.0, 9.5, 123.0, 127.0), 'name': 'Mindanao'},
        'Phil. Trench': {'bounds': (5.0, 15.0, 126.0, 129.0), 'name': 'Trench'}
    }

    predictions = []
    
    print("\n" + "="*60)
    print("📍 SOVEREIGN GPS TARGETS")
    print("="*60)
    
    for r_name, r_data in regions.items():
        b = r_data['bounds']
        region_quakes = [q for q in earthquakes if b[0] <= q['latitude'] <= b[1] and b[2] <= q['longitude'] <= b[3]]
        
        if region_quakes:
            p = ai.predict_next_event(region_quakes, r_data)
            predictions.append(p)
            
            status = "🟢 STABLE"
            if p['probability'] > 0.5: status = "🟠 WATCH"
            if p['probability'] > 0.7: status = "🔴 WARNING"

            lat = p['coords'][0]
            lon = p['coords'][1]
            maps_link = f"https://www.google.com/maps/search/?api=1&query={lat},{lon}"

            print(f"\n{status} | REGION: {p['region'].upper()}")
            print(f"   📅 TIME:  {format_prediction_time(p['target_time'])}")
            print(f"   📍 LOC:   {lat:.4f}°N, {lon:.4f}°E")
            print(f"   🔗 MAP:   {maps_link}")
            print(f"   💥 MAG:   {p['pred_magnitude']}")
            print(f"   📉 DEPTH: {p['pred_depth']} km")
            print(f"   ⚠️ PROB:  {p['probability']*100:.1f}%")

    if predictions:
        m = folium.Map(location=[12.8, 121.7], zoom_start=6, tiles='CartoDB dark_matter')
        for p in predictions:
            color = '#00ff00'
            if p['probability'] > 0.5: color = '#ffaa00'
            if p['probability'] > 0.7: color = '#ff0000'
            
            popup_html = f"""
            <b>GPS TARGET: {p['region']}</b><br>
            Lat: {p['coords'][0]:.4f}<br>Lon: {p['coords'][1]:.4f}<br>
            Time: {format_prediction_time(p['target_time'])}<br>
            Mag: {p['pred_magnitude']}<br>
            """
            
            # Draw Prediction Zone
            folium.Circle(
                location=p['coords'],
                radius=p['event_count'] * 500 + 10000,
                color=color,
                weight=2,
                fill=True,
                fill_opacity=0.4,
                popup=folium.Popup(popup_html, max_width=300)
            ).add_to(m)
            
            # Draw Exact Center
            folium.CircleMarker(
                location=p['coords'],
                radius=3,
                color='white',
                fill=True,
                fill_opacity=1.0
            ).add_to(m)
            
        m.save("64d_gps_oracle.html")
        print(f"\n✅ GPS ORACLE MAP SAVED: 64d_gps_oracle.html")

if __name__ == "__main__":
    main()