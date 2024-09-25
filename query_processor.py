from vector_db_handler import VectorDBHandler
from embedding_generator import EmbeddingGenerator
from utils import log_error
from typing import List, Dict, Any, Tuple
import numpy as np
import logging
logger = logging.getLogger(__name__)
from typing import Tuple, List
from typing import Tuple, List, Dict, Any, Optional
from qdrant_client import QdrantClient
from qdrant_client.http.models import ScoredPoint
from qdrant_client.http.exceptions import UnexpectedResponse
from fastembed import TextEmbedding, SparseTextEmbedding, SparseEmbedding
import numpy as np

class QueryProcessorError(Exception):
    """Custom exception class for QueryProcessor errors."""
    pass

class QueryProcessor:
    def __init__(self, qdrant_client: QdrantClient, collection_name: str):
        self.client = qdrant_client
        self.collection_name = collection_name
        self.dense_model = TextEmbedding(model_name="BAAI/bge-large-en-v1.5")
        self.sparse_model = SparseTextEmbedding(model_name="prithivida/Splade_PP_en_v1")

    @staticmethod
    def initialize(host: str, port: int, collection_name: str) -> 'QueryProcessor':
        """Initialize the Qdrant client and QueryProcessor."""
        from qdrant_client import QdrantClient
        
        try:
            client = QdrantClient(host=host, port=port)
            vector_db_handler = VectorDBHandler(client, collection_name)
            embedding_generator = EmbeddingGenerator()
            return QueryProcessor(vector_db_handler, embedding_generator)
        except Exception as e:
            log_error(f"Failed to initialize QueryProcessor: {str(e)}")
            raise QueryProcessorError("Failed to initialize QueryProcessor") from e

    def process_query(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Process a query and return relevant chunks with metadata."""
        try:
            sparse_results, dense_results = self.hybrid_search(query, limit)
            ranked_results = self.rank_results(sparse_results, dense_results)

            final_results = []
            for result_id, score in ranked_results:
                point = self.retrieve_point(result_id)  # Call the new method here
                if point:
                    point["score"] = score
                    final_results.append(point)

            return final_results
        except QueryProcessorError as e:
            logger.error(f"Query processing failed: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during query processing: {str(e)}")
            return []

    def generate_embeddings(self, query: str) -> Tuple[np.ndarray, SparseEmbedding]:
        """Generate both dense and sparse embeddings for the query."""
        try:
            dense_embedding = next(self.dense_model.embed([query]))
            sparse_embedding = next(self.sparse_model.embed([query]))
            return dense_embedding, sparse_embedding
        except StopIteration:
            logger.error("Embedding generation failed: empty result")
            raise QueryProcessorError("Embedding generation failed: empty result")
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            raise QueryProcessorError("Failed to generate embeddings") from e

    def rank_results(self, sparse_results: List[ScoredPoint], dense_results: List[ScoredPoint]) -> List[Tuple[str, float]]:
        """Rank results using RRF after performing hybrid search."""
        try:
            # Update to access attributes properly
            sparse_rank_list = self._rank_list(sparse_results)
            dense_rank_list = self._rank_list(dense_results)
            return self.rrf([sparse_rank_list, dense_rank_list])
        except Exception as e:
            logger.error(f"Result ranking failed: {str(e)}")
            raise QueryProcessorError("Result ranking failed") from e


    def _rank_list(self, search_result: List[ScoredPoint]) -> List[Tuple[str, int]]:
        """Convert search results into a list of (id, rank) tuples."""
        return [(point.id, rank + 1) for rank, point in enumerate(search_result)]

    def rrf(self, rank_lists: List[List[Tuple[str, int]]], alpha: int = 60, default_rank: int = 1000) -> List[Tuple[str, float]]:
        """Reciprocal Rank Fusion (RRF) algorithm to combine ranked lists."""
        try:
            all_items = set(item for rank_list in rank_lists for item, _ in rank_list)
            item_to_index = {item: idx for idx, item in enumerate(all_items)}
            rank_matrix = np.full((len(all_items), len(rank_lists)), default_rank)

            for list_idx, rank_list in enumerate(rank_lists):
                for item, rank in rank_list:
                    rank_matrix[item_to_index[item], list_idx] = rank

            rrf_scores = np.sum(1.0 / (alpha + rank_matrix), axis=1)
            sorted_indices = np.argsort(-rrf_scores)

            return [(list(item_to_index.keys())[idx], rrf_scores[idx]) for idx in sorted_indices]

        except Exception as e:
            log_error(f"RRF ranking failed: {str(e)}")
            raise QueryProcessorError("RRF ranking failed") from e
        
    def hybrid_search(self, query: str, limit: int = 200) -> Tuple[List[ScoredPoint], List[ScoredPoint]]:
        """Perform hybrid search using both dense and sparse embeddings."""
        try:
            dense_embedding, sparse_embedding = self.generate_embeddings(query)

            # Perform batch search with both dense and sparse embeddings
            search_results = self.client.search_batch(
                collection_name=self.collection_name,
                requests=[
                    {
                        "vector": {
                            "name": "text-dense",
                            "vector": dense_embedding.tolist(),
                        },
                        "limit": limit,
                        "with_payload": True,
                    },
                    {
                        "vector": {
                            "name": "text-sparse",
                            "vector": {
                                "indices": sparse_embedding.indices.tolist(),
                                "values": sparse_embedding.values.tolist(),
                            },
                        },
                        "limit": limit,
                        "with_payload": True,
                    },
                ],
            )

            # Returning search results from both dense and sparse searches
            return search_results[0], search_results[1]

        except Exception as e:
            logger.error(f"Hybrid search failed: {str(e)}")
            raise QueryProcessorError("Hybrid search failed") from e
        
    def process_query(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Process a query and return relevant chunks with metadata."""
        try:
            logger.info(f"Processing query: {query} with limit {limit}")
            sparse_results, dense_results = self.hybrid_search(query, limit)
            logger.info(f"Sparse results: {sparse_results}, Dense results: {dense_results}")

            if not sparse_results and not dense_results:
                logger.info("No results from hybrid search.")
                return []

            ranked_results = self.rank_results(sparse_results, dense_results)
            logger.info(f"Ranked results: {ranked_results}")

            final_results = []
            for result_id, score in ranked_results:
                point = self.retrieve_point(result_id)
                if point:
                    point["score"] = score
                    final_results.append(point)

            return final_results
        except QueryProcessorError as e:
            logger.error(f"Query processing failed: {str(e)}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during query processing: {str(e)}")
            return []
    
    def retrieve_point(self, point_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single point from the collection."""
        try:
            point = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[point_id]
            )
            if point:
                return {
                    "id": point_id,
                    "metadata": point[0].payload,
                    "content": point[0].payload.get("chunk_content", "N/A")  # Ensure 'chunk_content' is in payload
                }
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve point {point_id}: {str(e)}")
            return None

