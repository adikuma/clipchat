import os
import uuid
from pathlib import Path
from typing import List, Dict, Optional, Any
import chromadb
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from rich.console import Console
from rich.progress import Progress

console = Console()


class VectorDatabase:
    """manages video embeddings in chromadb"""

    def __init__(self, db_path: str = "./data/chroma_db"):
        self.db_path = Path(db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # initialize chromadb client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )

        self.collection = None
        self.collection_name = "video_segments"

    def get_or_create_collection(
        self, collection_name: Optional[str] = None
    ) -> Collection:
        """get or create a collection in the database"""
        if collection_name:
            self.collection_name = collection_name

        try:
            self.collection = self.client.get_collection(self.collection_name)
            console.print(
                f"[yellow]loaded existing collection: {self.collection_name}[/yellow]"
            )
        except:
            self.collection = self.client.create_collection(self.collection_name)
            console.print(
                f"[green]created new collection: {self.collection_name}[/green]"
            )

        return self.collection

    def delete_collection(self, collection_name: Optional[str] = None):
        """delete a collection from the database"""
        name = collection_name or self.collection_name
        try:
            self.client.delete_collection(name)
            console.print(f"[red]deleted collection: {name}[/red]")
            self.collection = None
        except Exception as e:
            console.print(
                f"[yellow]collection {name} doesn't exist or error: {e}[/yellow]"
            )

    def add_video_segments(
        self,
        metadatas: List[Dict],
        embeddings: List[List[float]],
        video_id: Optional[str] = None,
    ) -> List[str]:
        """
        add video segments to the database

        args:
            metadatas: list of segment metadata
            embeddings: list of embedding vectors
            video_id: optional video identifier

        returns:
            list of segment ids
        """
        if not self.collection:
            self.get_or_create_collection()

        if len(metadatas) != len(embeddings):
            raise ValueError("metadatas and embeddings must have same length")

        console.print(f"[blue]adding {len(metadatas)} segments to database...[/blue]")

        # prepare data for chromadb
        ids = []
        documents = []
        db_metadatas = []

        for i, (metadata, embedding) in enumerate(zip(metadatas, embeddings)):
            # generate unique id
            segment_id = str(uuid.uuid4())
            ids.append(segment_id)

            # document is the transcript text
            documents.append(metadata["transcript"])

            # prepare metadata (chromadb doesn't like complex nested objects)
            db_metadata = {
                "video_segment_id": metadata["video_segment_id"],
                "extracted_frame_path": metadata["extracted_frame_path"],
                "video_path": metadata["video_path"],
                "mid_time_sec": metadata["mid_time_sec"],
                "mid_time_ms": metadata["mid_time_ms"],
            }

            # add optional fields if they exist
            if "chunk_start_time_sec" in metadata:
                db_metadata.update(
                    {
                        "chunk_start_time_sec": metadata["chunk_start_time_sec"],
                        "chunk_end_time_sec": metadata["chunk_end_time_sec"],
                        "num_segments": metadata["num_segments"],
                    }
                )

            if video_id:
                db_metadata["video_id"] = video_id

            db_metadatas.append(db_metadata)

        # add to collection
        self.collection.add(
            ids=ids, embeddings=embeddings, metadatas=db_metadatas, documents=documents
        )

        console.print(f"[green]added {len(ids)} segments to collection[/green]")
        return ids

    def query_segments(
        self,
        query_embedding: List[float],
        n_results: int = 5,
        where_filter: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        query for similar segments

        args:
            query_embedding: embedding vector for query
            n_results: number of results to return
            where_filter: optional metadata filter

        returns:
            query results from chromadb
        """
        if not self.collection:
            raise ValueError("no collection loaded")

        results = self.collection.query(
            query_embeddings=[query_embedding], n_results=n_results, where=where_filter
        )

        return results

    def get_all_segments(self, video_id: Optional[str] = None) -> Dict[str, Any]:
        """
        get all segments, optionally filtered by video_id

        args:
            video_id: optional video identifier to filter by

        returns:
            all segments matching filter
        """
        if not self.collection:
            raise ValueError("no collection loaded")

        where_filter = {"video_id": video_id} if video_id else None

        # get all results (chromadb doesn't have a direct "get all" method)
        # so we query with a large n_results
        results = self.collection.get(
            where=where_filter, include=["metadatas", "documents", "embeddings"]
        )

        return results

    def get_collection_stats(self) -> Dict[str, Any]:
        """get statistics about the current collection"""
        if not self.collection:
            return {"error": "no collection loaded"}

        count = self.collection.count()

        # get sample of data to analyze
        sample = self.collection.peek(limit=10)

        stats = {
            "total_segments": count,
            "collection_name": self.collection_name,
            "sample_metadata_keys": (
                list(sample["metadatas"][0].keys()) if sample["metadatas"] else []
            ),
        }

        return stats

    def delete_video_segments(self, video_id: str):
        """delete all segments for a specific video"""
        if not self.collection:
            raise ValueError("no collection loaded")

        # find segments with this video_id
        results = self.collection.get(
            where={"video_id": video_id}, include=["metadatas"]
        )

        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            console.print(
                f"[red]deleted {len(results['ids'])} segments for video {video_id}[/red]"
            )
        else:
            console.print(f"[yellow]no segments found for video {video_id}[/yellow]")

    def clear_collection(self):
        """clear all data from the current collection"""
        if not self.collection:
            raise ValueError("no collection loaded")

        # delete and recreate collection
        self.delete_collection()
        self.get_or_create_collection()
        console.print("[red]cleared all data from collection[/red]")
