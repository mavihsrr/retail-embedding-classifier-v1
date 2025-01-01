# 🛍️ Retail Embedding Classifier v1

This repository contains the script used to fine-tune the `retail-embedding-classifier-v1`, a specialized model optimized for generating embeddings for retail-specific tasks. 
The fine-tuned model is available on [Hugging Face](https://huggingface.co/mavihsrr/retail-embedding-classifier-v1).

- **Product Similarity Matching**
- **Taxonomy Conversion**
- **Inventory Migration**
- **Grouping Similar Products**
- **Recommendation Systems**

---
- **Fine-Tuned for Retail**: Trained on 217,894 samples of retail-specific product descriptions and related metadata.
- **Embeddings for Retail-Specific Tasks**: Generates high-quality, 768-dimensional embeddings.
- **Supports Multi-Tier Retail Use Cases**: From similarity searches to classification and clustering.
- **Efficient Inference**: Built on `BAAI/bge-base-en`, leveraging the `CosineSimilarityLoss` function for robust semantic textual similarity.
---
## 🚀 Quick Start

Install the required library:
```pip install -U sentence-transformers```

Load the model and get started:
```
from sentence_transformers import SentenceTransformer
# Load the model
model = SentenceTransformer("mavihsrr/retail-embedding-classifier-v1")
# Example inputs
sentences = [
    "Organic Almond Butter - Creamy and Unsalted. High in Protein!",
    "Peanut Butter - All Natural, Unsalted, Smooth and Creamy.",
]
# Generate embeddings
embeddings = model.encode(sentences)
# Print shape of embeddings
print(embeddings.shape)  # Output: (2, 768)
```

## 📊 Training Details
- Base Model: BAAI/bge-base-en
- Loss Function: CosineSimilarityLoss
- Dataset: 174,064 training samples, focusing on retail-specific texts.
- Evaluation: 21,759 evaluation samples, achieving consistent high similarity scores.

## 💡 Use Cases
1. Product Similarity Matching
Find products similar to a given item based on their embeddings:
```
from sklearn.metrics.pairwise import cosine_similarity
similarity_matrix = cosine_similarity(embeddings)
print(similarity_matrix)
```
2. Taxonomy Conversion
Easily map products from one taxonomy to another using vector similarity.

3. Recommendations
Build recommendation systems by clustering or ranking products based on embedding similarity.

## 🤝 Contributing
Contributions are welcome! If you find issues or want to improve the model, feel free to create an issue or submit a pull request.

## 📄 License
This project is licensed under the MIT License.

✨ Connect with Us
For queries or suggestions, reach out via GitHub Issues or contact:
Email: shivam.m@ionio.io

