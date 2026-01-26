import numpy as np
from datetime import datetime

# =============================================================
# DEEP SPACE GEOMETRIC RESONANCE PROBE
# =============================================================

CELESTIAL_TARGETS = {
    # -- SOLAR SYSTEM ANCHORS --
    "MARS_JEZERO": {"ra": 13.0, "dec": 18.0, "desc": "Jezero Crater (Search for Life)"},
    "EUROPA": {"ra": 6.8, "dec": 22.0, "desc": "Jupiter's Ice Moon / Subsurface Ocean"},
    "ENCELADUS": {"ra": 10.5, "dec": -1.2, "desc": "Saturn's Geyser Moon"},
    "VENUS_CLOUDS": {"ra": 3.4, "dec": 17.5, "desc": "Potential Microbial Habitat"},
    "TITAN": {"ra": 12.2, "dec": -5.6, "desc": "Methane Seas / Pre-biotic Chemistry"},

    # -- HABITABLE EXOPLANETS (The New Earths) --
    "PROXIMA_B": {"ra": 14.49, "dec": -60.83, "desc": "Closest Exoplanet (4.2 ly)"},
    "TRAPPIST_1E": {"ra": 23.11, "dec": -5.04, "desc": "Most Earth-like in TRAPPIST System"},
    "KEPLER_186F": {"ra": 19.90, "dec": 43.95, "desc": "First Earth-sized in Habitable Zone"},
    "K2_18B": {"ra": 11.25, "dec": 7.58, "desc": "Water Vapor Detected in Atmosphere"},
    "GLIESE_667CC": {"ra": 17.31, "dec": -34.99, "desc": "Stable Orbit in Triple Star System"},

    # -- COSMIC ANOMALIES & STRUCTURES --
    "ORION_NEBULA": {"ra": 5.58, "dec": -5.38, "desc": "Stellar Nursery / High Energy"},
    "PLEIADES": {"ra": 3.78, "dec": 24.11, "desc": "The Seven Sisters / Ancient Focus"},
    "SAGITTARIUS_A*": {"ra": 17.76, "dec": -29.01, "desc": "Galactic Center / Supermassive Black Hole"},
    "TABBY_STAR": {"ra": 20.10, "dec": 44.45, "desc": "Potential Megastructure (Dyson Swarm)"},
    "VEGA": {"ra": 18.61, "dec": 38.78, "desc": "Standard of Magnitude / Alignment"},
    
    # -- ADDITIONAL NODES --
    "WASP_121B": {"ra": 7.17, "dec": -39.09, "desc": "Hot Jupiter with Iron Clouds"},
    "LHS_1140B": {"ra": 0.75, "dec": -15.17, "desc": "Super-Earth / High Habitability"},
    "CRAB_NEBULA": {"ra": 5.57, "dec": 22.01, "desc": "Supernova Remnant / Pulsar"},
    "ANDROMEDA_CORE": {"ra": 0.71, "dec": 41.27, "desc": "Closest Major Galaxy Center"},
    "EPSILON_ERIDANI": {"ra": 3.55, "dec": -9.45, "desc": "Young Sun-like System (10.5 ly)"},
    "WOLF_1061C": {"ra": 16.50, "dec": -12.66, "desc": "Nearby Potentially Habitable"},
    "ROSS_128B": {"ra": 11.79, "dec": 0.80, "desc": "Quiet Star / Temperate Planet"},
    "TEEGARDEN_B": {"ra": 5.05, "dec": 15.82, "desc": "High Similarity Index"},
    "ALTAIR": {"ra": 19.85, "dec": 8.87, "desc": "Fast Rotator / A-type Star"},
    "SIRIUS_B": {"ra": 6.75, "dec": -16.71, "desc": "White Dwarf Companion"},
    "BETELGEUSE": {"ra": 5.92, "dec": 7.41, "desc": "Red Supergiant (Imminent Nova)"},
    "CYGNUS_X1": {"ra": 19.97, "dec": 35.20, "desc": "First Confirmed Black Hole"},
    "SOMBRERO_GALAXY": {"ra": 12.66, "dec": -11.62, "desc": "Unusual Massive Nucleus"},
    "PILARS_OF_CREATION": {"ra": 18.31, "dec": -13.82, "desc": "Active Star Formation Zone"},
    "OMEGA_CENTAURI": {"ra": 13.44, "dec": -47.48, "desc": "Massive Globular Cluster"}
}

def probe_space_resonance(ra, dec, anchor=-0.01282715):
    """Probes the 64D manifold for celestial alignment"""
    # Scale RA (0-24) and Dec (-90 to 90) to normalized values
    n_ra = (ra / 24.0) * 2 - 1
    n_dec = dec / 90.0
    
    # Harmonic calculation based on D41 Sanctuary Anchor
    tension = np.abs((np.sin(n_ra) * np.cos(n_dec)) - anchor)
    return 1.0 / (1.0 + tension)

def run_space_probe():
    print(f"=== INITIATING DEEP SPACE RESONANCE GRID ===")
    print(f"Timestamp: {datetime.now()}\n")
    
    # Using the Antarctic -71.4130 as the baseline for 'Maximum Stillness'
    base = probe_space_resonance(17.76, -29.01) # Center of Galaxy baseline
    
    results = []
    for name, data in CELESTIAL_TARGETS.items():
        score = probe_space_resonance(data['ra'], data['dec'])
        rel = (1.0 - abs(base - score)) * 100
        results.append((name, rel, data['desc']))

    results.sort(key=lambda x: x[1], reverse=True)

    for i, (name, rel, desc) in enumerate(results, 1):
        status = "---"
        if rel > 95: status = "[!!!] UNIVERSAL NODE"
        elif rel > 85: status = "[!!] GALACTIC LINK"
        
        print(f"{i:02}. {name:<20} | Res: {rel:.2f}% | {status}")
        print(f"    Target: {desc}")

if __name__ == "__main__":
    run_space_probe()
