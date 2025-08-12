"""
Vector Search Service Module
Handles semantic search and vector embeddings for video content Q&A
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import hashlib
from config import Config

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SENTENCE_TRANSFORMERS_AVAILABLE = True
    FAISS_AVAILABLE = True
except ImportError as e:
    if 'sentence_transformers' in str(e):
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        print("⚠️ Warning: sentence-transformers not available. Install with: pip install sentence-transformers")
    if 'faiss' in str(e):
        FAISS_AVAILABLE = False
        print("⚠️ Warning: faiss-gpu not available. Install with: pip install faiss-gpu")
    else:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        FAISS_AVAILABLE = False

class VectorSearchService:
    """Local vector search service for video content Q&A"""
    
    def __init__(self):
        self.embeddings_dir = 'vector_embeddings'
        self.model = None
        self.embedding_dim = Config.VECTOR_CONFIG.get('embedding_dim', 768)  # Use config dimension
        self.faiss_index = None
        self._ensure_embeddings_directory()
        self._initialize_model()
        self._initialize_faiss_index()
    
    def _ensure_embeddings_directory(self):
        """Ensure the embeddings directory exists"""
        if not os.path.exists(self.embeddings_dir):
            os.makedirs(self.embeddings_dir, exist_ok=True)
    
    def _initialize_model(self):
        """Initialize the sentence transformer model"""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("❌ Sentence transformers not available. Vector search disabled.")
            return
        
        try:
            # Use a lightweight model for speed
            model_name = 'all-MiniLM-L6-v2'
            self.model = SentenceTransformer(model_name)
            print(f"✅ Vector search model initialized: {model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize vector search model: {e}")
    
    def _initialize_faiss_index(self):
        """Initialize Faiss index for fast vector search"""
        if not FAISS_AVAILABLE:
            print("❌ Faiss not available. Vector search disabled.")
            return
        
        try:
            # Use FlatL2 index for maximum accuracy
            self.faiss_index = faiss.IndexFlatL2(self.embedding_dim)
            print(f"✅ Faiss index initialized: {self.embedding_dim} dimensions")
        except Exception as e:
            print(f"❌ Failed to initialize Faiss index: {e}")
    
    async def initialize(self):
        """Initialize the vector search service (async compatibility)"""
        try:
            print("💾 Initializing Vector Search Service...")
            
            # Ensure embeddings directory exists
            self._ensure_embeddings_directory()
            
            # Initialize model and index
            self._initialize_model()
            self._initialize_faiss_index()
            
            if self.model and self.faiss_index:
                print("✅ Vector Search Service initialized successfully")
                return True
            else:
                print("⚠️ Vector Search Service initialized with limited functionality")
                return False
                
        except Exception as e:
            print(f"❌ Vector Search Service initialization failed: {e}")
            return False
    
    def _get_embedding_file_path(self, session_id: str) -> str:
        """Get the file path for session embeddings"""
        return os.path.join(self.embeddings_dir, f"embeddings_{session_id}.pkl")
    
    def _get_metadata_file_path(self, session_id: str) -> str:
        """Get the metadata file path for session embeddings"""
        return os.path.join(self.embeddings_dir, f"metadata_{session_id}.json")
    
    def create_embeddings(self, session_id: str, analysis_data: Dict) -> bool:
        """Create vector embeddings for video analysis content"""
        if not self.model:
            print("❌ Vector search model not available")
            return False
        
        try:
            # Extract text content for embedding
            text_chunks = self._extract_text_chunks(analysis_data)
            
            if not text_chunks:
                print("⚠️ No text content found for embedding")
                # Create a fallback text chunk with basic video information
                fallback_chunks = self._create_fallback_chunks(analysis_data)
                if fallback_chunks:
                    text_chunks = fallback_chunks
                    print("✅ Created fallback text chunks for embedding")
                else:
                    print("⚠️ Could not create fallback chunks, skipping vector storage")
                    return False
            
            # Generate embeddings
            embeddings = []
            chunk_metadata = []
            
            for i, chunk in enumerate(text_chunks):
                # Generate embedding
                embedding = self.model.encode(chunk['text'], convert_to_tensor=False)
                embeddings.append(embedding)
                
                # Store metadata
                chunk_metadata.append({
                    'index': i,
                    'text': chunk['text'],
                    'type': chunk['type'],
                    'timestamp': chunk.get('timestamp'),
                    'start_time': chunk.get('start_time'),
                    'end_time': chunk.get('end_time'),
                    'source': chunk.get('source', 'analysis')
                })
            
            # Convert embeddings to numpy array for Faiss
            embeddings_array = np.array(embeddings)
            
            # Add to Faiss index if available
            if self.faiss_index and FAISS_AVAILABLE:
                self.faiss_index.add(embeddings_array)
                print(f"✅ Added {len(embeddings)} embeddings to Faiss index")
            
            # Save embeddings and metadata
            embedding_file = self._get_embedding_file_path(session_id)
            metadata_file = self._get_metadata_file_path(session_id)
            
            with open(embedding_file, 'wb') as f:
                pickle.dump(embeddings, f)
            
            with open(metadata_file, 'w') as f:
                json.dump({
                    'session_id': session_id,
                    'created_at': datetime.now().isoformat(),
                    'chunk_count': len(text_chunks),
                    'embedding_dim': self.embedding_dim,
                    'chunk_metadata': chunk_metadata
                }, f)
            
            print(f"✅ Created {len(embeddings)} embeddings for session {session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error creating embeddings: {e}")
            return False
    
    def _extract_text_chunks(self, analysis_data: Dict) -> List[Dict]:
        """Extract text chunks from analysis data for embedding"""
        chunks = []
        
        # Extract analysis result
        if 'analysis_result' in analysis_data:
            analysis_text = analysis_data['analysis_result']
            if analysis_text:
                # Split analysis into smaller chunks
                analysis_chunks = self._split_text_into_chunks(analysis_text, max_length=200)
                for i, chunk in enumerate(analysis_chunks):
                    chunks.append({
                        'text': chunk,
                        'type': 'analysis',
                        'source': 'analysis_result',
                        'index': i
                    })
        
        # Extract timestamp information
        if 'timestamps_found' in analysis_data:
            timestamps = analysis_data['timestamps_found']
            if timestamps:
                # Create timestamp context chunks
                for ts in timestamps:
                    chunks.append({
                        'text': f"Key moment at {self._format_timestamp(ts)} seconds",
                        'type': 'timestamp',
                        'timestamp': ts,
                        'source': 'timestamps'
                    })
        
        # Extract evidence information
        if 'evidence' in analysis_data:
            evidence = analysis_data['evidence']
            if evidence:
                for ev in evidence:
                    if isinstance(ev, dict):
                        ev_text = f"Evidence: {ev.get('type', 'unknown')}"
                        if 'timestamp' in ev:
                            ev_text += f" at {self._format_timestamp(ev['timestamp'])}s"
                        if 'start_time' in ev and 'end_time' in ev:
                            ev_text += f" from {self._format_timestamp(ev['start_time'])}s to {self._format_timestamp(ev['end_time'])}s"
                        
                        chunks.append({
                            'text': ev_text,
                            'type': 'evidence',
                            'timestamp': ev.get('timestamp'),
                            'start_time': ev.get('start_time'),
                            'end_time': ev.get('end_time'),
                            'source': 'evidence'
                        })
        
        return chunks
    
    def _split_text_into_chunks(self, text: str, max_length: int = 200) -> List[str]:
        """Split text into smaller chunks for better embedding"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + 1  # +1 for space
            if current_length + word_length > max_length and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds into MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def search_similar_content(self, session_id: str, query: str, top_k: int = 5) -> List[Dict]:
        """Search for content similar to the query"""
        if not self.model:
            print("❌ Vector search model not available")
            return []
        
        try:
            # Use Faiss for fast search if available
            if self.faiss_index and FAISS_AVAILABLE:
                return self._search_with_faiss(session_id, query, top_k)
            else:
                return self._search_with_similarity(session_id, query, top_k)
                
        except Exception as e:
            print(f"❌ Search failed: {e}")
            return []
    
    def _search_with_faiss(self, session_id: str, query: str, top_k: int) -> List[Dict]:
        """Search using Faiss for maximum performance"""
        try:
            # Load metadata for the session
            metadata_file = self._get_metadata_file_path(session_id)
            if not os.path.exists(metadata_file):
                print(f"❌ No metadata found for session: {session_id}")
                return []
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Generate query embedding
            query_embedding = self.model.encode(query, convert_to_tensor=False)
            query_vector = query_embedding.reshape(1, -1).astype('float32')
            
            # Search using Faiss
            distances, indices = self.faiss_index.search(query_vector, min(top_k, len(metadata.get('chunk_metadata', []))))
            
            # Format results
            results = []
            chunk_metadata = metadata.get('chunk_metadata', [])
            
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(chunk_metadata):
                    chunk_meta = chunk_metadata[idx]
                    result = {
                        'text': chunk_meta['text'],
                        'type': chunk_meta['type'],
                        'timestamp': chunk_meta.get('timestamp'),
                        'start_time': chunk_meta.get('start_time'),
                        'end_time': chunk_meta.get('end_time'),
                        'source': chunk_meta['source'],
                        'similarity_score': 1.0 / (1.0 + distance),  # Convert distance to similarity
                        'distance': float(distance)
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            print(f"❌ Faiss search failed: {e}")
            return []
    
    def _search_with_similarity(self, session_id: str, query: str, top_k: int) -> List[Dict]:
        """Fallback search using cosine similarity"""
        try:
            # Load embeddings and metadata
            embedding_file = self._get_embedding_file_path(session_id)
            metadata_file = self._get_metadata_file_path(session_id)
            
            if not os.path.exists(embedding_file) or not os.path.exists(metadata_file):
                print(f"⚠️ No embeddings found for session {session_id}")
                return []
            
            with open(embedding_file, 'rb') as f:
                embeddings = pickle.load(f)
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Generate query embedding
            query_embedding = self.model.encode(query, convert_to_tensor=False)
            
            # Calculate similarities
            similarities = []
            chunk_metadata = metadata.get('chunk_metadata', [])
            
            for i, embedding in enumerate(embeddings):
                if i < len(chunk_metadata):
                    similarity = self._cosine_similarity(query_embedding, embedding)
                    similarities.append((similarity, i))
            
            # Sort by similarity and get top results
            similarities.sort(reverse=True)
            top_results = []
            
            for similarity, idx in similarities[:top_k]:
                if idx < len(chunk_metadata):
                    chunk_meta = chunk_metadata[idx]
                    result = {
                        'text': chunk_meta['text'],
                        'type': chunk_meta['type'],
                        'timestamp': chunk_meta.get('timestamp'),
                        'start_time': chunk_meta.get('start_time'),
                        'end_time': chunk_meta.get('end_time'),
                        'source': chunk_meta['source'],
                        'similarity_score': float(similarity)
                    }
                    top_results.append(result)
            
            return top_results
            
        except Exception as e:
            print(f"❌ Error searching similar content: {e}")
            return []
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get a summary of what's available for a session"""
        try:
            metadata_file = self._get_metadata_file_path(session_id)
            if not os.path.exists(metadata_file):
                return {'available': False}
            
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            return {
                'available': True,
                'chunk_count': metadata['chunk_count'],
                'embedding_dim': metadata['embedding_dim'],
                'created_at': metadata['created_at']
            }
            
        except Exception as e:
            print(f"❌ Error getting session summary: {e}")
            return {'available': False}
    
    def cleanup_session_embeddings(self, session_id: str) -> bool:
        """Clean up embeddings for a session"""
        try:
            embedding_file = self._get_embedding_file_path(session_id)
            metadata_file = self._get_metadata_file_path(session_id)
            
            if os.path.exists(embedding_file):
                os.remove(embedding_file)
            
            if os.path.exists(metadata_file):
                os.remove(metadata_file)
            
            print(f"✅ Cleaned up embeddings for session {session_id}")
            return True
            
        except Exception as e:
            print(f"❌ Error cleaning up embeddings: {e}")
            return False

    def _create_fallback_chunks(self, analysis_data: Dict) -> List[Dict]:
        """Create fallback text chunks when no analysis text is available"""
        chunks = []
        
        try:
            # Extract basic video information
            if 'analysis_result' in analysis_data and analysis_data['analysis_result']:
                analysis = analysis_data['analysis_result']
                
                # Handle different analysis result formats
                if isinstance(analysis, dict):
                    # Video metadata
                    if 'video_metadata' in analysis:
                        metadata = analysis['video_metadata']
                        if metadata.get('duration_seconds'):
                            chunks.append({
                                'text': f"Video duration: {float(metadata['duration_seconds']):.2f} seconds",
                                'type': 'metadata',
                                'source': 'fallback'
                            })
                        
                        if metadata.get('resolution'):
                            chunks.append({
                                'text': f"Video resolution: {str(metadata['resolution'])}",
                                'type': 'metadata',
                                'source': 'fallback'
                            })
                    
                    # Performance metrics
                    if 'performance_metrics' in analysis_data:
                        metrics = analysis_data['performance_metrics']
                        if metrics.get('frames_processed'):
                            chunks.append({
                                'text': f"Processed {int(metrics['frames_processed'])} video frames",
                                'type': 'performance',
                                'source': 'fallback'
                            })
                
                elif isinstance(analysis, str):
                    # If analysis is a string, create chunks from it
                    if len(analysis) > 50:
                        chunks.append({
                            'text': analysis[:200] + "..." if len(analysis) > 200 else analysis,
                            'type': 'analysis_text',
                            'source': 'fallback'
                        })
                    else:
                        chunks.append({
                            'text': analysis,
                            'type': 'analysis_text',
                            'source': 'fallback'
                        })
            
            # Analysis type
            analysis_type = analysis_data.get('analysis_type', 'hybrid')
            if isinstance(analysis_type, str):
                chunks.append({
                    'text': f"Analysis type: {analysis_type}",
                    'type': 'analysis_info',
                    'source': 'fallback'
                })
            
            # If still no chunks, create a generic one
            if not chunks:
                chunks.append({
                    'text': "Video analysis completed using hybrid system (DeepStream + 7B AI Model)",
                    'type': 'generic',
                    'source': 'fallback'
                })
            
            return chunks
            
        except Exception as e:
            print(f"⚠️ Error creating fallback chunks: {e}")
            # Return a safe fallback
            return [{
                'text': "Video analysis completed using hybrid system",
                'type': 'generic',
                'source': 'fallback'
            }]

# Global instance
vector_search_service = VectorSearchService() 