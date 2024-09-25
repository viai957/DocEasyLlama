# vector_db_handler.py

from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, SparseVector, NamedSparseVector, SearchRequest
from qdrant_client.http import models
from typing import List, Dict, Any, Tuple
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import (
    Distance,
    NamedSparseVector,
    NamedVector,
    SparseVector,
    PointStruct,
    SearchRequest,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    ScoredPoint,
)

class VectorDBHandler:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    def create_collection(self, vector_size: int, distance: models.Distance = models.Distance.COSINE):
        """
        Create a new collection in Qdrant.
        
        Args:
            vector_size (int): The size of the vector embeddings.
            distance (models.Distance): The distance metric to use (default: COSINE).
        
        Raises:
            UnexpectedResponse: If there's an error creating the collection.
        """
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    "text-dense": models.VectorParams(
                        size=models.dense_vector_size,
                        distance=distance
                    )
                },
                sparse_vectors_config={
                    "text-sparse": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,  # Move `on_disk` here into SparseIndexParams
                        )
                    )
                },
            )
            self.logger.info(f"Collection '{self.collection_name}' created successfully with dense and sparse vectors.")

        except models.UnexpectedResponse as e:
            self.logger.error(f"Failed to create collection '{self.collection_name}': {e}")
            raise    

    def add_documents(self, documents: List[str], sparse_embeddings: List, dense_embeddings: List[List[float]], metadata: List[Dict[str, Any]], ids: List[str]):
        """
        Add documents with embeddings and metadata to the collection.
        
        Args:
            documents (List[str]): List of document texts.
            embeddings (List[List[float]]): List of embedding vectors.
            metadata (List[Dict[str, Any]]): List of metadata dictionaries.
            ids (List[str]): List of unique identifiers for the documents.
        
        Raises:
            ValueError: If the input lists have mismatched lengths.
            UnexpectedResponse: If there's an error adding the documents.
        """
        if not (len(documents) == len(sparse_embeddings) == len(dense_embeddings) == len(metadata) == len(ids)):
            raise ValueError("All input lists must have the same length.")
        
        try:
            points = []
            for idx in range(len(documents)):
                sparse_vectors = SparseVector(
                    indices=sparse_embeddings[idx].indices.tolist(),
                    values=sparse_embeddings[idx].values.tolist(),
                )
                dense_vector = dense_embeddings[idx]
                point = PointStruct(
                    id=ids[idx],
                    payload=metadata[idx],
                    vector = {
                        "text-dense": dense_vector,
                        "text-sparse": sparse_vectors
                    },
                )
                points.append(point)

            self.client.upsert(
                collection_name=self.collection_name,
                points=points
                )
            self.logger.info(f"Successfully added {len(documents)} documents to collection '{self.collection_name}'.")
        
        except UnexpectedResponse as e:
            self.logger.error(f"Failed to add documents to collection '{self.collection_name}': {str(e)}")
            raise

    def hybrid_search(self, query_sparse_vector: SparseVector, query_dense_vector: List[float], limit: int = 10) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Search for similar documents in the collection using a hybrid query.
        
        Args:
            query_vector (SparseEmbedding): The sparse query vector to search for.
            query_dense_vector (List[float]): The dense query vector to search for.
            limit (int): The maximum number of results to return (default: 10).
            query_filter (Optional[Dict]): Optional filter to apply to the search (default: None).
        
        Returns:
            List[Dict[str, Any]]: List of search results.
        
        Raises:
            UnexpectedResponse: If there's an error during the search operation.
        """
        try:
            results = self.client.search_batch(
                collection_name=self.collection_name,
                requests=[
                        SearchRequest(
                            vector=NamedVector(
                                name="text-dense",
                                vector=query_dense_vector,
                            ),
                            limit=limit,
                            with_payload=True,
                        ),
                        SearchRequest(
                            vector=NamedSparseVector(
                                name="text-sparse",
                                vector=SparseVector(
                                    indices=query_sparse_vector.indices.tolist(),
                                    values=query_sparse_vector.values.tolist(),
                                ),
                            ),
                            limit=limit,
                            with_payload=True,
                        ),
                    ],
                )

            return results[0], results[1]
        
        except UnexpectedResponse as e:
            self.logger.error(f"Search operation failed in collection '{self.collection_name}': {str(e)}")
            raise
    