import json
import numpy as np
import requests
from datetime import datetime, timedelta
import re
from bs4 import BeautifulSoup
import ssl
import urllib3
import time

# Disable SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ============================================================================
# NEW HELPER FUNCTION FOR SPECIFIC DATE/TIME FORMATTING
# ============================================================================

def format_specific_datetime(dt_obj):
    """
    Formats a datetime object into a specific, user-friendly string.
    Example: "Monday, October 28, 2024 at 09:15 AM"
    """
    if not isinstance(dt_obj, datetime):
        return "Invalid Time"
    # Format: Day, Month Day, Year at HH:MM AM/PM
    return dt_obj.strftime("%A, %B %d, %Y at %I:%M %p")

# ============================================================================
# REAL PHIVOLCS DATA EXTRACTION (with modified sample output)
# ============================================================================

def extract_phivolcs_earthquake_data():
    """Extract actual earthquake data from PHIVOLCS website"""
    
    print("="*80)
    print("🌋 EXTRACTING REAL PHIVOLCS EARTHQUAKE DATA")
    print("="*80)
    
    url = "https://earthquake.phivolcs.dost.gov.ph/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    }
    
    try:
        print("🔄 Connecting to PHIVOLCS...")
        response = requests.get(url, headers=headers, verify=False, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Failed to fetch: {response.status_code}")
            return []
        
        soup = BeautifulSoup(response.content, 'html.parser')
        print(f"✅ Page loaded: {len(response.content)} bytes")
        
        tables = soup.find_all('table')
        print(f"📊 Found {len(tables)} tables on the page")
        
        earthquakes = []
        
        for table_idx, table in enumerate(tables):
            rows = table.find_all('tr')
            if len(rows) < 3:
                continue
            
            headers_text = [th.get_text(strip=True).lower() for th in rows[0].find_all(['th', 'td'])]
            
            earthquake_keywords = ['date', 'time', 'magnitude', 'depth', 'lat', 'lon', 'location']
            keyword_matches = sum(1 for h in headers_text if any(kw in h for kw in earthquake_keywords))
            
            if keyword_matches >= 3:
                print(f"\n🎯 TABLE {table_idx + 1}: Likely earthquake data ({len(rows)} rows)")
                print(f"   Headers: {headers_text}")
                
                for row_idx, row in enumerate(rows[1:]):
                    cells = row.find_all(['td', 'th'])
                    cell_data = [cell.get_text(strip=True) for cell in cells]
                    
                    if len(cell_data) >= 5:
                        eq = parse_earthquake_row(headers_text, cell_data)
                        if eq:
                            earthquakes.append(eq)
        
        print(f"\n📈 Successfully extracted {len(earthquakes)} earthquake events from main page")
        
        print("\n📅 Checking EQLatest-Monthly page...")
        monthly_data = extract_monthly_data()
        if monthly_data:
            earthquakes.extend(monthly_data)
            print(f"📈 Added {len(monthly_data)} events from monthly archive")
        
        unique_earthquakes = remove_duplicates(earthquakes)
        print(f"📊 Final unique earthquake count: {len(unique_earthquakes)}")
        
        if unique_earthquakes:
            print("\n📋 SAMPLE EARTHQUAKES (Formatted with specific date/time):")
            for i, eq in enumerate(unique_earthquakes[:5]):
                # --- MODIFIED: Use the new specific datetime formatter ---
                formatted_time = format_specific_datetime(eq.get('date_time'))
                print(f"\n{i+1}. {formatted_time}")
                print(f"   Magnitude: {eq.get('magnitude', 'Unknown')}")
                print(f"   Depth: {eq.get('depth', 'Unknown')} km")
                print(f"   Location: {eq.get('location', 'Unknown')}")
                print(f"   Coordinates: {eq.get('latitude', '?')}°N, {eq.get('longitude', '?')}°E")
        
        return unique_earthquakes
        
    except Exception as e:
        print(f"❌ Error extracting data: {e}")
        import traceback
        traceback.print_exc()
        return []

# --- (The parsing functions like parse_earthquake_row, parse_datetime, etc., remain unchanged) ---
def parse_earthquake_row(headers, cell_data):
    """Parse a row of earthquake data"""
    eq = {}
    
    try:
        # Map headers to data
        for i, header in enumerate(headers):
            if i < len(cell_data):
                value = cell_data[i]
                
                # Parse based on header content
                if 'date' in header or 'time' in header:
                    # Try to parse date/time
                    dt = parse_datetime(value)
                    if dt:
                        eq['date_time'] = dt
                        eq['timestamp'] = dt.timestamp()
                
                elif 'magnitude' in header or 'mag' in header:
                    # Extract magnitude
                    mag = extract_magnitude(value)
                    if mag:
                        eq['magnitude'] = mag
                
                elif 'depth' in header:
                    # Extract depth in km
                    depth = extract_depth(value)
                    if depth:
                        eq['depth'] = depth
                
                elif 'lat' in header or 'latitude' in header:
                    # Extract latitude
                    lat = extract_coordinate(value, is_lat=True)
                    if lat:
                        eq['latitude'] = lat
                
                elif 'lon' in header or 'longitude' in header:
                    # Extract longitude
                    lon = extract_coordinate(value, is_lat=False)
                    if lon:
                        eq['longitude'] = lon
                
                elif 'location' in header or 'place' in header or 'area' in header:
                    eq['location'] = value
        
        # If we have at least magnitude and some location info, it's valid
        if 'magnitude' in eq:
            # Ensure we have coordinates (try to estimate from location if missing)
            if 'latitude' not in eq or 'longitude' not in eq:
                if 'location' in eq:
                    # Try to get coordinates from location name
                    coords = estimate_coordinates(eq['location'])
                    if coords:
                        eq['latitude'], eq['longitude'] = coords
            
            return eq
    
    except Exception as e:
        # print(f"⚠️ Error parsing row: {e}")
        return None

def parse_datetime(value):
    """Parse various datetime formats used by PHIVOLCS"""
    try:
        # Common PHIVOLCS formats
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%d-%b-%Y %H:%M:%S',
            '%b %d, %Y %H:%M:%S',
            '%Y/%m/%d %H:%M:%S',
            '%m/%d/%Y %H:%M:%S',
            '%d/%m/%Y %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except:
                continue
        
        # Try to extract just date
        date_match = re.search(r'\d{4}-\d{2}-\d{2}', value)
        if date_match:
            date_str = date_match.group()
            time_match = re.search(r'\d{2}:\d{2}:\d{2}', value)
            if time_match:
                time_str = time_match.group()
                return datetime.strptime(f"{date_str} {time_str}", '%Y-%m-%d %H:%M:%S')
            else:
                return datetime.strptime(date_str, '%Y-%m-%d')
    
    except:
        pass
    
    return None

def extract_magnitude(text):
    """Extract magnitude from text"""
    try:
        patterns = [
            r'[Mm]\s*(\d+\.\d+|\d+)', r'Magnitude\s*:\s*(\d+\.\d+|\d+)',
            r'Mag\s*:\s*(\d+\.\d+|\d+)', r'(\d+\.\d+|\d+)\s*[Mm]',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try: return float(match.group(1))
                except: pass
        try: return float(text.strip())
        except: pass
    except: pass
    return None

def extract_depth(text):
    """Extract depth in km from text"""
    try:
        patterns = [
            r'(\d+\.\d+|\d+)\s*km', r'Depth\s*:\s*(\d+\.\d+|\d+)',
            r'(\d+\.\d+)\s*kilometers',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try: return float(match.group(1))
                except: pass
        try: return float(text.strip())
        except: pass
    except: pass
    return None

def extract_coordinate(text, is_lat=True):
    """Extract coordinate from text"""
    try:
        patterns = [
            r'(\d+\.\d+)°?\s*[NS]?' if is_lat else r'(\d+\.\d+)°?\s*[EW]?',
            r'(\d+\.\d+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    coord = float(match.group(1))
                    if is_lat and 'S' in text.upper(): coord = -coord
                    elif not is_lat and 'W' in text.upper(): coord = -coord
                    return coord
                except: pass
        try: return float(text.strip())
        except: pass
    except: pass
    return None

def estimate_coordinates(location):
    """Estimate coordinates from location name (Philippines-specific)"""
    location_coords = {
        'manila': (14.5995, 120.9842), 'quezon city': (14.6760, 121.0437),
        'cebu': (10.3157, 123.8854), 'davao': (7.1907, 125.4553),
        'marikina valley fault': (14.6000, 121.1000), 'philippine trench': (12.0000, 126.5000),
        'taal': (14.0021, 120.9937), 'mayon': (13.2572, 123.6856), 'pinatubo': (15.1300, 120.3500)
    }
    location_lower = location.lower()
    for key, coords in location_coords.items():
        if key in location_lower: return coords
    return (12.8797, 121.7740)

def extract_monthly_data():
    """Extract data from EQLatest-Monthly page"""
    url = "https://earthquake.phivolcs.dost.gov.ph/EQLatest-Monthly/"
    try:
        response = requests.get(url, verify=False, timeout=30)
        soup = BeautifulSoup(response.content, 'html.parser')
        earthquakes = []
        for table in soup.find_all('table'):
            rows = table.find_all('tr')
            if len(rows) > 10:
                headers = [th.get_text(strip=True).lower() for th in rows[0].find_all(['th', 'td'])]
                if any(kw in ' '.join(headers) for kw in ['date', 'time', 'magnitude', 'depth']):
                    for row in rows[1:]:
                        cells = [cell.get_text(strip=True) for cell in row.find_all(['td', 'th'])]
                        if len(cells) >= 4:
                            eq = parse_monthly_row(cells)
                            if eq: earthquakes.append(eq)
        return earthquakes
    except Exception as e:
        print(f"⚠️ Error extracting monthly data: {e}")
        return []

def parse_monthly_row(cells):
    """Parse row from monthly data"""
    try:
        eq = {}
        if len(cells) > 1:
            try:
                dt_str = f"{cells[0]} {cells[1]}"
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
                eq['date_time'], eq['timestamp'] = dt, dt.timestamp()
            except: pass
        if len(cells) > 3:
            try: eq['latitude'] = float(cells[2])
            except: pass
            try: eq['longitude'] = float(cells[3])
            except: pass
        if len(cells) > 5:
            try: eq['depth'] = float(cells[4])
            except: pass
            mag = extract_magnitude(cells[5])
            if mag: eq['magnitude'] = mag
        if len(cells) > 6: eq['location'] = cells[6]
        eq['source'] = 'PHIVOLCS Monthly'
        return eq if 'magnitude' in eq else None
    except Exception: return None

def remove_duplicates(earthquakes):
    """Remove duplicate earthquake entries"""
    unique, seen = [], set()
    for eq in earthquakes:
        time_key = int(eq.get('timestamp', 0) / 60)
        loc_key = f"{eq.get('latitude', 0):.2f},{eq.get('longitude', 0):.2f}"
        mag_key = f"{eq.get('magnitude', 0):.1f}"
        key = f"{time_key}_{loc_key}_{mag_key}"
        if key not in seen:
            seen.add(key)
            unique.append(eq)
    return unique
    
# ============================================================================
# ENHANCED PANTHEON WITH REAL DATA (with modified prediction generation)
# ============================================================================

def run_pantheon_with_real_data():
    """Run Pantheon prediction system with real PHIVOLCS data"""
    print("\n" + "="*80)
    print("🌋 PANTHEON PREDICTION WITH REAL PHIVOLCS DATA")
    print("="*80)
    
    try:
        with open('Metalearnerv16_EVOLVED.json', 'r') as f:
            pantheon = json.load(f).get('meta_pantheon', {})
        if not pantheon: raise FileNotFoundError("No meta_pantheon found")
        print(f"✅ Loaded Pantheon with {len(pantheon)} nodes")
    except Exception as e:
        print(f"❌ Error loading Pantheon: {e}")
        return
    
    print("\n📡 FETCHING LIVE PHIVOLCS DATA...")
    earthquakes = extract_phivolcs_earthquake_data()
    
    if not earthquakes:
        print("⚠️ No PHIVOLCS data available, using fallback...")
        earthquakes = get_fallback_usgs_data()
    
    if not earthquakes:
        print("❌ CRITICAL: No earthquake data could be fetched. Aborting analysis.")
        return

    print(f"\n📊 ANALYZING {len(earthquakes)} EARTHQUAKE EVENTS")
    
    analysis_time = datetime.now()
    analysis = analyze_seismic_patterns(earthquakes, analysis_time)
    
    predictions = generate_predictions(analysis, pantheon, analysis_time)
    
    display_predictions(predictions, earthquakes, analysis_time)
    
    save_results(predictions, earthquakes, analysis_time)

def analyze_seismic_patterns(earthquakes, analysis_time):
    """Analyze seismic patterns from real data"""
    print("\n" + "="*80)
    print("🔬 ANALYZING SEISMIC PATTERNS")
    print("="*80)
    
    if not earthquakes: return {"error": "No data"}
    
    regions = {
        'Luzon': {'events': [], 'coordinates': (15.5, 121.0)},
        'Visayas': {'events': [], 'coordinates': (10.5, 123.5)},
        'Mindanao': {'events': [], 'coordinates': (7.5, 125.0)},
        'Manila Metro Area': {'events': [], 'coordinates': (14.6, 121.0)},
        'Philippine Trench': {'events': [], 'coordinates': (10.0, 127.0)},
    }
    
    for eq in earthquakes:
        lat, lon = eq.get('latitude', 0), eq.get('longitude', 0)
        if 12.0 <= lat <= 19.0 and 118.0 <= lon <= 124.0: regions['Luzon']['events'].append(eq)
        if 9.0 <= lat <= 12.0 and 122.0 <= lon <= 126.0: regions['Visayas']['events'].append(eq)
        if 5.0 <= lat <= 9.5 and 123.0 <= lon <= 127.0: regions['Mindanao']['events'].append(eq)
        if 14.2 <= lat <= 14.9 and 120.8 <= lon <= 121.2: regions['Manila Metro Area']['events'].append(eq)
        if 5.0 <= lat <= 14.0 and 126.0 <= lon <= 128.0: regions['Philippine Trench']['events'].append(eq)
    
    analysis = {'total_events': len(earthquakes), 'timestamp': analysis_time.isoformat(), 'regions': {}}
    
    for name, data in regions.items():
        if data['events']:
            magnitudes = [e.get('magnitude', 0) for e in data['events']]
            recent = [e for e in data['events'] if 'timestamp' in e and (analysis_time.timestamp() - e['timestamp']) < 86400]
            analysis['regions'][name] = {
                'event_count': len(data['events']),
                'recent_24h': len(recent),
                'max_magnitude': max(magnitudes) if magnitudes else 0,
                'avg_magnitude': sum(magnitudes) / len(magnitudes) if magnitudes else 0,
                'coordinates': data['coordinates']
            }
            print(f"\n📍 {name}:")
            print(f"   Events: {len(data['events'])}, Recent (24h): {len(recent)}, Max Mag: {max(magnitudes) if magnitudes else 'N/A'}")
            
    return analysis

def generate_predictions(analysis, pantheon, analysis_time):
    """Generate predictions with specific date/time windows"""
    print("\n" + "="*80)
    print("🎯 GENERATING PREDICTIONS WITH SPECIFIC TIMEFRAMES")
    print("="*80)
    
    predictions = []
    specialist_nodes = [node for node in pantheon.values() if isinstance(node, dict) and 'state_dict' in node]
    print(f"🧠 Using {len(specialist_nodes)} specialist nodes for analysis")
    
    for region, data in analysis.get('regions', {}).items():
        if data['event_count'] > 0:
            risk = calculate_risk_score(data)
            pred = {'region': region, 'coordinates': data['coordinates'], 'risk_score': risk}
            
            # --- MODIFIED: Calculate specific start and end dates for the time window ---
            if risk > 0.7:
                mag_pred, start_delta, end_delta, conf = min(8.0, data['max_magnitude'] + 1.5), timedelta(hours=24), timedelta(hours=72), 0.7
            elif risk > 0.5:
                mag_pred, start_delta, end_delta, conf = min(7.0, data['max_magnitude'] + 1.0), timedelta(days=3), timedelta(days=7), 0.5
            elif risk > 0.3:
                mag_pred, start_delta, end_delta, conf = min(6.0, data['max_magnitude'] + 0.5), timedelta(days=7), timedelta(days=14), 0.3
            else:
                mag_pred, start_delta, end_delta, conf = max(4.0, data['avg_magnitude']), timedelta(days=14), timedelta(days=30), 0.2

            start_date = analysis_time + start_delta
            end_date = analysis_time + end_delta
            
            pred.update({
                'predicted_magnitude': round(mag_pred, 1),
                'confidence': conf,
                'probability': min(0.5, risk * 0.7),
                'time_window_start_iso': start_date.isoformat(),
                'time_window_end_iso': end_date.isoformat(),
                'time_window_display': f"Between {format_specific_datetime(start_date)} and {format_specific_datetime(end_date)}"
            })
            
            predictions.append(pred)
            
            print(f"\n📍 {region}:")
            print(f"   Predicted Magnitude: M{pred['predicted_magnitude']:.1f}")
            print(f"   Time Window: {pred['time_window_display']}")
            print(f"   Probability: {pred['probability']:.1%}, Risk Score: {risk:.2f}")
    
    return sorted(predictions, key=lambda x: x['risk_score'], reverse=True)

def calculate_risk_score(region_data):
    """Calculate risk score based on seismic activity"""
    score = 0.0
    score += 0.3 if region_data['event_count'] > 20 else 0.2 if region_data['event_count'] > 10 else 0.1 if region_data['event_count'] > 5 else 0
    score += 0.3 if region_data['recent_24h'] > 5 else 0.2 if region_data['recent_24h'] > 2 else 0.1 if region_data['recent_24h'] > 0 else 0
    score += 0.4 if region_data['max_magnitude'] > 6.0 else 0.3 if region_data['max_magnitude'] > 5.0 else 0.2 if region_data['max_magnitude'] > 4.0 else 0.1
    return min(1.0, score)

def get_fallback_usgs_data():
    """Get fallback data from USGS if PHIVOLCS fails"""
    try:
        now = datetime.now()
        params = {
            'format': 'geojson', 'starttime': (now - timedelta(days=7)).strftime('%Y-%m-%d'),
            'endtime': now.strftime('%Y-%m-%d'), 'minlatitude': 4.0, 'maxlatitude': 21.0,
            'minlongitude': 116.0, 'maxlongitude': 127.0, 'minmagnitude': 2.5,
            'orderby': 'time-asc', 'limit': 100
        }
        response = requests.get("https://earthquake.usgs.gov/fdsnws/event/1/query", params=params, timeout=30)
        data = response.json()
        earthquakes = []
        for feature in data.get('features', []):
            props, coords = feature['properties'], feature['geometry']['coordinates']
            earthquakes.append({
                'date_time': datetime.fromtimestamp(props['time']/1000), 'timestamp': props['time']/1000,
                'latitude': coords[1], 'longitude': coords[0], 'depth': coords[2],
                'magnitude': props['mag'], 'location': props.get('place', 'Unknown'), 'source': 'USGS'
            })
        print(f"📊 Using {len(earthquakes)} USGS events as fallback")
        return earthquakes
    except Exception as e:
        print(f"⚠️ USGS fallback failed: {e}")
        return []

def display_predictions(predictions, earthquakes, analysis_time):
    """Display prediction results with specific date/time details"""
    print("\n" + "="*80)
    print("🏆 PANTHEON PREDICTION RESULTS")
    print("="*80)
    
    # --- MODIFIED: Display the analysis time with specific formatting ---
    print(f"\nAnalysis conducted on: {format_specific_datetime(analysis_time)}")
    print(f"Based on analysis of {len(earthquakes)} recent earthquake events.")
    
    if not predictions:
        print("\n📭 No significant predictions at this time. Seismic activity appears to be within normal ranges.")
        return
        
    print(f"🎯 Generated {len(predictions)} regional predictions.")
    
    print("\n🔴 HIGHEST RISK PREDICTIONS:")
    
    high_risk_found = False
    for i, pred in enumerate(predictions[:3]):
        if pred['risk_score'] > 0.3:
            high_risk_found = True
            print(f"\n{i+1}. {pred['region']}")
            print(f"   📍 Coordinates: {pred['coordinates'][0]:.4f}°N, {pred['coordinates'][1]:.4f}°E")
            print(f"   📊 Predicted Magnitude: M{pred['predicted_magnitude']}")
            # --- MODIFIED: Use the new specific 'time_window_display' field ---
            print(f"   🎯 Predicted Timeframe: {pred['time_window_display']}")
            print(f"   📈 Probability: {pred['probability']:.1%}")
            print(f"   ⚠️  Risk Score: {pred['risk_score']:.2f} (Confidence: {pred['confidence']:.0%})")
            
            if pred['predicted_magnitude'] >= 6.0:
                print(f"   💡 RECOMMENDATION: High alert. Potential for significant seismic event. Prepare accordingly.")
            elif pred['predicted_magnitude'] >= 5.0:
                print(f"   💡 RECOMMENDATION: Monitor closely. Increased activity detected in the region.")

    if not high_risk_found:
        print("\n🟢 CURRENT STATUS: All regions show low to moderate seismic activity.")
        print("   No immediate high-risk (Risk Score > 0.3) predictions generated at this time.")

def save_results(predictions, earthquakes, analysis_time):
    """Save results to file with enhanced, readable timestamps"""
    timestamp_str = analysis_time.strftime("%Y%m%d_%H%M%S")
    
    results = {
        # --- MODIFIED: Added a human-readable timestamp to the saved file ---
        'analysis_timestamp_iso': analysis_time.isoformat(),
        'analysis_timestamp_readable': format_specific_datetime(analysis_time),
        'data_source': 'PHIVOLCS + Pantheon Analysis',
        'total_events_analyzed': len(earthquakes),
        'predictions': predictions,
        'earthquake_data_sample': [
            {**eq, 'date_time': eq['date_time'].isoformat()} 
            for eq in earthquakes[:10] if 'date_time' in eq
        ]
    }
    
    filename = f"PANTHEON_PHILIPPINES_{timestamp_str}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {filename}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 LAUNCHING PANTHEON WITH REAL PHIVOLCS DATA")
    # --- MODIFIED: Use the new specific datetime formatter for the launch time ---
    print(f"Launch Time: {format_specific_datetime(datetime.now())}")
    
    run_pantheon_with_real_data()
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)