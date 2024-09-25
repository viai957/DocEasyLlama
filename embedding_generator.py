# embedding_generator.py

from fastembed import TextEmbedding, SparseTextEmbedding
from typing import List

class EmbeddingGenerator:
    def __init__(self):
        self.dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5", batch_size=32)
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1", batch_size=32)

    def generate_dense_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate dense embeddings for the list of texts."""
        return list(self.dense_model.embed(texts))

    def generate_sparse_embeddings(self, texts: List[str]) -> List:
        """Generate sparse embeddings for the list of texts."""
        return list(self.sparse_model.embed(texts))
