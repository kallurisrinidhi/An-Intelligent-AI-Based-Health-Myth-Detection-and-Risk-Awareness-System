from sentence_transformers import SentenceTransformer
import pickle

print("Loading Transformer model...")

# Industry Semantic AI Model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Save model
pickle.dump(model, open("sbert_model.pkl", "wb"))

print("✅ SBERT Model Ready")