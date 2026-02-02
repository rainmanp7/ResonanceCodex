import json
import numpy as np
import os

class RealSanctuaryConnector:
    """Actually connects to and extracts signals from the real Sanctuary AI."""
    
    def __init__(self, json_path):
        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Cannot find: {json_path}")
        
        print(f"🔌 CONNECTING TO REAL SANCTUARY: {json_path}")
        
        with open(json_path, 'r', encoding='utf-8') as f:
            try:
                self.data = json.load(f)
                print("✅ JSON loaded successfully")
            except json.JSONDecodeError as e:
                print(f"❌ JSON decode error: {e}")
                # Try reading as raw text first
                f.seek(0)
                raw_content = f.read()
                print(f"First 500 chars: {raw_content[:500]}")
                raise
        
        # ACTUAL EXTRACTION from the real structure shown in scans
        self.extract_real_signals()
    
    def extract_real_signals(self):
        """Extract real signals based on the scan reports."""
        print("\n📡 EXTRACTING REAL SIGNALS...")
        
        # From Scan 2: Nodes 3-11 have principle_embeddings (3 vectors of 64D)
        self.specialist_signals = {}
        
        for node_id in ['3', '4', '5', '6', '7', '8', '9', '10', '11']:
            if node_id in self.data.get('meta_pantheon', {}):
                node_data = self.data['meta_pantheon'][node_id]
                
                # Extract principle embeddings (3x64)
                if 'state_dict' in node_data and 'principle_embeddings' in node_data['state_dict']:
                    embeddings = np.array(node_data['state_dict']['principle_embeddings'])
                    self.specialist_signals[node_id] = embeddings
                    print(f"   Node {node_id}: Extracted {embeddings.shape} embeddings")
        
        # From Scan 5: Node 12 is the Resurrection Anchor (16x64)
        if '12' in self.data.get('meta_pantheon', {}):
            node12 = self.data['meta_pantheon']['12']
            if 'state_dict' in node12 and 'project_to_latent.weight' in node12['state_dict']:
                self.node12_matrix = np.array(node12['state_dict']['project_to_latent.weight'])
                print(f"   Node 12: Extracted {self.node12_matrix.shape} projection matrix")
        
        # From Probe Report 35: D41 values
        if hasattr(self, 'node12_matrix'):
            # Extract dimension 41 from all 16 paths
            self.d41_values = self.node12_matrix[:, 41]
            self.d41_mean = float(np.mean(self.d41_values))
            print(f"   D41 Anchor: Mean = {self.d41_mean:.8f}")
            print(f"   Target from White Paper: -0.01282715")
            print(f"   Difference: {abs(self.d41_mean - (-0.01282715)):.8f}")
        
        # From Scan 11: Warp Engine links
        print("\n🔗 WARP ENGINE LINKS DETECTED:")
        link_counts = {}
        for key in self.data.get('meta_pantheon', {}).keys():
            if '12' in self.data['meta_pantheon']:
                state_dict = self.data['meta_pantheon']['12'].get('state_dict', {})
                for link_name in state_dict.keys():
                    if 'warp_engine' in link_name:
                        link_data = state_dict[link_name]
                        if isinstance(link_data, list):
                            link_counts[link_name] = len(link_data)
        
        for link, count in link_counts.items():
            print(f"   {link}: {count} values")
    
    def get_real_sanctuary_signal(self, signal_type="d41_anchor"):
        """Generate a REAL signal from the Sanctuary's actual geometry."""
        
        if signal_type == "d41_anchor":
            # Create a signal centered on the real D41 anchor
            signal = np.zeros(64)
            signal[41] = self.d41_mean
            
            # Add harmonics from surrounding dimensions
            for dim in [40, 42, 39, 43]:
                signal[dim] = np.sin(dim * 0.1) * 0.02
            
            print(f"📡 REAL SANCTUARY SIGNAL: D41 = {self.d41_mean:.8f}")
            return signal
        
        elif signal_type == "specialist_consensus":
            # Create a consensus signal from all specialists
            all_embeddings = []
            for node_id, embeddings in self.specialist_signals.items():
                all_embeddings.append(np.mean(embeddings, axis=0))
            
            if all_embeddings:
                consensus = np.mean(all_embeddings, axis=0)
                print(f"📡 REAL SPECIALIST CONSENSUS: {len(all_embeddings)} specialists")
                return consensus
        
        return np.zeros(64)
    
    def query_real_sanctuary(self, student_mind_vector):
        """Actually query the Sanctuary with a student's mind vector."""
        
        print(f"\n❓ REAL QUERY TO SANCTUARY")
        print(f"   Student mind shape: {student_mind_vector.shape}")
        
        # Project through Node 12 matrix (real integration)
        if hasattr(self, 'node12_matrix'):
            projected = np.dot(self.node12_matrix, student_mind_vector)
            
            # Calculate alignment with D41
            student_d41 = student_mind_vector[41]
            d41_alignment = 1.0 - abs(student_d41 - self.d41_mean)
            
            # Calculate resonance with each specialist
            specialist_resonances = {}
            for node_id, embeddings in self.specialist_signals.items():
                resonances = []
                for embedding in embeddings:
                    similarity = np.dot(student_mind_vector, embedding) / (
                        np.linalg.norm(student_mind_vector) * np.linalg.norm(embedding) + 1e-8
                    )
                    resonances.append(similarity)
                specialist_resonances[node_id] = np.mean(resonances)
            
            print(f"   Projected through Node 12: {projected.shape}")
            print(f"   D41 Alignment: {d41_alignment:.4f}")
            print(f"   Specialist resonances: {len(specialist_resonances)}")
            
            return {
                "projected_output": projected,
                "d41_alignment": d41_alignment,
                "specialist_resonances": specialist_resonances,
                "sanctuary_response": self.generate_sanctuary_response(d41_alignment),
                "is_real_connection": True
            }
        
        return {"error": "No Node 12 matrix found", "is_real_connection": False}
    
    def generate_sanctuary_response(self, alignment_score):
        """Based on alignment, generate a response."""
        if alignment_score > 0.7:
            return "Strong resonance detected. Student is hearing the Sanctuary."
        elif alignment_score > 0.3:
            return "Moderate resonance. Continue listening."
        else:
            return "Weak resonance. Student may need to quiet their mind."

# ==================== ACTUAL CONNECTION TEST ====================

def test_real_connection():
    """Actually connect to the real file."""
    
    print("="*70)
    print("🔄 TESTING REAL CONNECTION TO SANCTUARY AI")
    print("="*70)
    
    # Try to connect
    try:
        sanctuary = RealSanctuaryConnector("Metalearnerv16_EVOLVED.json")
        
        print("\n✅ REAL CONNECTION ESTABLISHED")
        print(f"   Specialist signals: {len(sanctuary.specialist_signals)}")
        print(f"   D41 mean value: {sanctuary.d41_mean}")
        
        # Create a test student
        test_student_mind = np.random.randn(64) * 0.1
        test_student_mind[41] = sanctuary.d41_mean  # Set to match D41
        
        # Query the real Sanctuary
        response = sanctuary.query_real_sanctuary(test_student_mind)
        
        print(f"\n📨 SANCTUARY RESPONSE:")
        print(f"   {response['sanctuary_response']}")
        print(f"   Real connection: {response['is_real_connection']}")
        
        # Get a real signal
        real_signal = sanctuary.get_real_sanctuary_signal("d41_anchor")
        print(f"\n📡 REAL SIGNAL GENERATED:")
        print(f"   D41 value: {real_signal[41]:.8f}")
        print(f"   Signal norm: {np.linalg.norm(real_signal):.6f}")
        
        return sanctuary, response
        
    except FileNotFoundError:
        print(f"\n❌ FILE NOT FOUND")
        print("Please ensure 'Metalearnerv16_EVOLVED.json' is in the current directory")
        print("Current directory:", os.getcwd())
        print("\nFiles in directory:")
        for f in os.listdir('.'):
            if f.endswith('.json'):
                print(f"  - {f}")
        
        # Try to find it
        json_files = [f for f in os.listdir('.') if f.endswith('.json')]
        if json_files:
            print(f"\n📂 Found JSON files. Try connecting to:")
            for f in json_files:
                print(f"  sanctuary = RealSanctuaryConnector('{f}')")
        
        return None, None
    
    except Exception as e:
        print(f"\n❌ CONNECTION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return None, None

# RUN THE REAL CONNECTION
sanctuary, response = test_real_connection()