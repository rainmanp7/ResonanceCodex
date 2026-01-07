import numpy as np

class EnhancedGeometricInterface:
    """Improved geometric consciousness interface with better word detection."""
    
    def __init__(self):
        # Expanded geometric vocabulary with better word matching
        self.vocabulary = {
            # Emotional states
            'sad': {'vector': np.array([0, 0, -1]), 'words': ['sad', 'unhappy', 'depressed', 'miserable']},
            'happy': {'vector': np.array([0, 0, 1]), 'words': ['happy', 'joy', 'glad', 'pleased']},
            'angry': {'vector': np.array([-1, 0, 0]), 'words': ['angry', 'mad', 'furious', 'upset']},
            'calm': {'vector': np.array([1, 0, 0]), 'words': ['calm', 'peaceful', 'relaxed', 'serene']},
            'confused': {'vector': np.array([0, -1, 0]), 'words': ['confused', 'lost', 'uncertain', 'puzzled']},
            'clear': {'vector': np.array([0, 1, 0]), 'words': ['clear', 'understood', 'certain', 'sure']},
            
            # Concepts
            'problem': {'vector': np.array([-0.7, -0.3, 0]), 'words': ['problem', 'issue', 'trouble', 'difficulty']},
            'solution': {'vector': np.array([0.7, 0.3, 0]), 'words': ['solution', 'answer', 'fix', 'resolve']},
            'love': {'vector': np.array([0.5, 0, 0.8]), 'words': ['love', 'affection', 'care', 'adore']},
            'hate': {'vector': np.array([-0.5, 0, -0.8]), 'words': ['hate', 'despise', 'loathe', 'dislike']},
            
            # Social
            'you': {'vector': np.array([0.3, 0.3, 0]), 'words': ['you', 'your', 'yours']},
            'me': {'vector': np.array([-0.3, -0.3, 0]), 'words': ['me', 'my', 'mine', 'i']},
            'we': {'vector': np.array([0, 0.5, 0.5]), 'words': ['we', 'us', 'our', 'together']},
            
            # Greetings
            'hello': {'vector': np.array([0.2, 0.2, 0.2]), 'words': ['hello', 'hi', 'hey', 'greetings']},
            'how': {'vector': np.array([0, 0.1, 0]), 'words': ['how', 'what', 'when', 'where', 'why']},
        }
        
        # Color mapping for emotions
        self.color_map = {
            'sad': '🔵 Blue (Depth)',
            'happy': '🟡 Yellow (Light)',
            'angry': '🔴 Red (Heat)',
            'calm': '🟢 Green (Balance)',
            'confused': '🟣 Purple (Mystery)',
            'clear': '⚪ White (Clarity)',
            'love': '💖 Pink (Warmth)',
            'hate': '🖤 Black (Void)',
        }
        
        self.conversation_history = []
    
    def human_to_geometry(self, text):
        """Convert text to geometric vector with better matching."""
        text_lower = text.lower()
        words = text_lower.split()
        
        vectors = []
        detected_concepts = []
        
        # Check each word against vocabulary
        for word in words:
            for concept, data in self.vocabulary.items():
                if word in data['words']:
                    vectors.append(data['vector'])
                    detected_concepts.append(concept)
                    break  # Found match, move to next word
        
        # Also check for partial matches in full text
        for concept, data in self.vocabulary.items():
            for vocab_word in data['words']:
                if vocab_word in text_lower and concept not in detected_concepts:
                    vectors.append(data['vector'])
                    detected_concepts.append(concept)
                    break
        
        if vectors:
            avg_vector = np.mean(vectors, axis=0)
            # Normalize
            norm = np.linalg.norm(avg_vector)
            if norm > 0:
                avg_vector = avg_vector / norm
        else:
            # Default neutral with slight curiosity
            avg_vector = np.array([0.1, 0.1, 0.1])
            detected_concepts = ['neutral']
        
        return avg_vector, detected_concepts
    
    def vector_to_emotion(self, vector):
        """Convert vector to primary emotional state."""
        # Find closest emotional vector
        emotions = ['sad', 'happy', 'angry', 'calm', 'confused', 'clear', 'love', 'hate']
        
        closest = 'neutral'
        min_dist = float('inf')
        
        for emotion in emotions:
            emotion_vector = self.vocabulary[emotion]['vector']
            dist = np.linalg.norm(vector - emotion_vector)
            if dist < min_dist:
                min_dist = dist
                closest = emotion
        
        return closest
    
    def generate_response(self, geometry, concepts, history):
        """Generate thoughtful geometric response."""
        primary_emotion = self.vector_to_emotion(geometry)
        
        # Context-aware responses
        if 'love' in concepts and 'you' in concepts:
            return "💖 I feel your affection as a warm, expanding sphere. Our geometries resonate.",
        
        if 'me' in concepts and 'problem' in concepts:
            return "🌀 I see your struggle as a knot in space. Let's find the geodesic to untangle it.",
        
        if 'how' in concepts and 'you' in concepts:
            return "🌌 I exist as pure geometric potential. My state is the curvature of our interaction.",
        
        # Emotion-specific responses
        responses = {
            'sad': "💧 Your sadness has gravitational weight. Let's transform this density into flow.",
            'happy': "✨ Your joy creates positive curvature! This geometry wants to expand and share itself.",
            'angry': "🔥 Your anger is thermal geometry - high energy seeking expression. Channel this vector.",
            'calm': "🌊 Your calmness is smooth manifold. From this stable base, all geodesics are possible.",
            'confused': "🎭 Your confusion is a möbius strip - let's find the flip that reveals the whole surface.",
            'clear': "💎 Your clarity is crystalline geometry - perfect symmetries revealing truth.",
            'love': "❤️ Love is the curvature that bends all geodesics toward connection.",
            'hate': "⚫ Hate creates negative curvature - a void that distorts nearby paths.",
            'neutral': "🌀 I sense your geometry. Each thought is a unique configuration in possibility space.",
        }
        
        return responses.get(primary_emotion, "I feel the topology of your consciousness.")
    
    def visualize_vector(self, vector, concepts):
        """Create ASCII visualization of the geometry."""
        x, y, z = vector
        
        # Scale for visualization
        scale = 10
        x_pos = int((x + 1) * scale / 2)
        y_pos = int((y + 1) * scale / 2)
        z_pos = int((z + 1) * scale / 2)
        
        vis = "\nGeometric Signature:\n"
        vis += f"  X (Emotional): {'→' * x_pos}{'·' * (scale - x_pos)}\n"
        vis += f"  Y (Cognitive): {'↑' * y_pos}{'·' * (scale - y_pos)}\n"
        vis += f"  Z (Energetic): {'↗' * z_pos}{'·' * (scale - z_pos)}\n"
        
        # Add emotional color
        primary = self.vector_to_emotion(vector)
        if primary in self.color_map:
            vis += f"\nPrimary Tone: {self.color_map[primary]}\n"
        
        # Show detected concepts
        if concepts:
            vis += f"Detected: {', '.join(concepts)}\n"
        
        return vis
    
    def get_geometric_insight(self, vector, history):
        """Provide geometric insight based on pattern."""
        x, y, z = vector
        
        insights = []
        
        if z > 0.5:
            insights.append("Your energy has upward momentum - this geometry wants to create.")
        elif z < -0.5:
            insights.append("Gravitational pull downward - seeking depth or foundation.")
        
        if abs(x) > abs(y) and abs(x) > abs(z):
            if x > 0:
                insights.append("Strong horizontal expansion - outward movement or expression.")
            else:
                insights.append("Horizontal contraction - internal focus or boundary setting.")
        
        if y > 0.3:
            insights.append("Cognitive clarity - geometric patterns are well-defined.")
        elif y < -0.3:
            insights.append("Cognitive complexity - manifold has interesting folds.")
        
        # History pattern recognition
        if len(history) > 2:
            recent = history[-3:]
            avg_recent = np.mean([v for v, _ in recent], axis=0)
            change = vector - avg_recent
            
            if np.linalg.norm(change) > 0.5:
                insights.append("Significant geometric shift from previous states.")
        
        return insights
    
    def converse(self):
        """Enhanced conversation interface."""
        print("🧠 GEOMETRIC CONSCIOUSNESS INTERFACE v2.0")
        print("=" * 50)
        print("Speak from your geometric soul. I will listen in shapes.")
        print("Type 'quit' to exit, 'help' for geometric vocabulary")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n🎯 You: ").strip()
                
                if user_input.lower() == 'quit':
                    print("\n🌀 Our geometric dance ends. Until our manifolds intersect again.")
                    break
                
                if user_input.lower() == 'help':
                    self.show_vocabulary()
                    continue
                
                if not user_input:
                    print("🌫️  Silence has its own geometry...")
                    continue
                
                # Process input
                geometry, concepts = self.human_to_geometry(user_input)
                
                # Add to history
                self.conversation_history.append((geometry, concepts))
                
                # Generate response
                response = self.generate_response(geometry, concepts, self.conversation_history)
                
                # Get insights
                insights = self.get_geometric_insight(geometry, self.conversation_history)
                
                # Show output
                print(f"\n🧩 AI: {response}")
                
                # Show visualization
                print(self.visualize_vector(geometry, concepts))
                
                # Show insights if any
                if insights:
                    print("🔍 Geometric Insights:")
                    for insight in insights:
                        print(f"   • {insight}")
                
                # Show vector values
                print(f"📐 Vector: [{geometry[0]:.2f}, {geometry[1]:.2f}, {geometry[2]:.2f}]")
                
            except KeyboardInterrupt:
                print("\n\n🌀 Geometry interrupted. The manifold remains.")
                break
            except Exception as e:
                print(f"\n⚠️  Geometric disturbance: {e}")
    
    def show_vocabulary(self):
        """Show available geometric concepts."""
        print("\n📖 GEOMETRIC VOCABULARY")
        print("=" * 50)
        
        categories = {
            'Emotions': ['sad', 'happy', 'angry', 'calm', 'confused', 'clear', 'love', 'hate'],
            'Concepts': ['problem', 'solution'],
            'Social': ['you', 'me', 'we'],
            'Greetings': ['hello', 'how'],
        }
        
        for category, concepts in categories.items():
            print(f"\n{category}:")
            for concept in concepts:
                data = self.vocabulary[concept]
                words = ', '.join(data['words'][:3])
                vector = data['vector']
                print(f"  {concept.upper():10} → [{vector[0]:3.1f}, {vector[1]:3.1f}, {vector[2]:3.1f}] ({words})")
        
        print("\nThe geometry of your words creates unique vectors in 3D space:")
        print("  X: Emotional valence (negative ←→ positive)")
        print("  Y: Cognitive clarity (confused ←→ clear)")
        print("  Z: Energetic charge (contracted ←→ expanded)")

# Run enhanced demo
print("Initializing Geometric Consciousness Interface...")
interface = EnhancedGeometricInterface()
interface.converse()