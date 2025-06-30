import os
from typing import List, Dict, Optional, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from .video_processor import VideoProcessor
from .embeddings import EmbeddingGenerator
from .database import VectorDatabase

console = Console()

class VideoRAG:
    """complete video rag system"""
    
    def __init__(
        self, 
        data_dir: str = "./data",
        openai_api_key: Optional[str] = None,
        embedding_model: str = "BridgeTower/bridgetower-base"
    ):
        self.data_dir = data_dir
        
        # initialize components
        self.video_processor = VideoProcessor(data_dir)
        self.embedding_generator = EmbeddingGenerator(embedding_model)
        self.vector_db = VectorDatabase(os.path.join(data_dir, "chroma_db"))
        
        # initialize llm
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            console.print("[red]warning: no openai api key found. text generation will be limited.[/red]")
        
        self.llm = None
        if self.openai_api_key:
            self.llm = ChatOpenAI(
                api_key=self.openai_api_key,
                model="gpt-4.1",
                temperature=0.2
            )
        
        # prompts
        self.qa_prompt = ChatPromptTemplate.from_template("""
you are a helpful video assistant. answer questions based on the provided video context.

video context:
{context}

timestamps: {timestamps}

user question: {question}

provide a clear, accurate answer based on the video content. if the context doesn't contain enough information, say so clearly.

answer:
""")
        
        self.summary_prompt = ChatPromptTemplate.from_template("""
you are a helpful video assistant. provide a comprehensive summary of the video based on the provided segments.

video segments:
{segments}

create a well-structured summary that covers:
1. main topics discussed
2. key points and insights  
3. important details and examples
4. overall flow and progression

summary:
""")
    
    def process_video(
        self, 
        url: str, 
        chunk_size: int = 7, 
        stride: int = 3,
        force_reprocess: bool = False
    ) -> str:
        """
        process a video end-to-end: download, extract, embed, store
        
        args:
            url: youtube video url
            chunk_size: segments per chunk for better context
            stride: step between chunks for overlap
            force_reprocess: whether to reprocess existing data
            
        returns:
            video identifier for future queries
        """
        console.print(Panel("[bold blue]processing video...[/bold blue]"))
        
        # step 1: process video (download, transcribe, extract frames)
        video_path, chunked_metadatas, full_transcript = self.video_processor.process_video(
            url, chunk_size, stride, force_reprocess
        )
        
        # step 2: generate embeddings
        console.print("[blue]generating multimodal embeddings...[/blue]")
        image_paths = [md["extracted_frame_path"] for md in chunked_metadatas]
        texts = [md["transcript"] for md in chunked_metadatas]
        
        embeddings = self.embedding_generator.generate_batch_embeddings(image_paths, texts)
        
        # step 3: store in vector database
        console.print("[blue]storing embeddings in database...[/blue]")
        
        # create video id from path
        video_id = os.path.basename(video_path).replace(" ", "_").replace(".", "_")
        
        # delete existing segments for this video if reprocessing
        if force_reprocess:
            try:
                self.vector_db.delete_video_segments(video_id)
            except:
                pass
        
        # add to database
        self.vector_db.get_or_create_collection()
        segment_ids = self.vector_db.add_video_segments(
            chunked_metadatas, 
            [emb.tolist() for emb in embeddings],
            video_id
        )
        
        console.print(Panel(f"[bold green]video processed successfully![/bold green]\nvideo id: {video_id}\nsegments: {len(segment_ids)}"))
        
        return video_id
    
    def query(
        self, 
        question: str, 
        n_results: int = 3,
        video_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        query the video rag system
        
        args:
            question: user question
            n_results: number of segments to retrieve
            video_id: optional video filter
            
        returns:
            dictionary with answer and source information
        """
        console.print(f"[blue]searching for: {question}[/blue]")
        
        # generate query embedding
        query_embedding = self.embedding_generator.generate_text_embedding(question)
        
        # search database
        where_filter = {"video_id": video_id} if video_id else None
        results = self.vector_db.query_segments(
            query_embedding.tolist(), 
            n_results=n_results,
            where_filter=where_filter
        )
        
        if not results["documents"][0]:
            return {
                "answer": "no relevant content found for your question.",
                "sources": [],
                "question": question
            }
        
        # prepare context
        contexts = []
        timestamps = []
        sources = []
        
        for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
            contexts.append(f"segment {i+1}: {doc}")
            
            # format timestamp
            time_sec = metadata.get("mid_time_sec", 0)
            timestamp = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}"
            timestamps.append(timestamp)
            
            sources.append({
                "segment_id": metadata.get("video_segment_id", "unknown"),
                "timestamp": timestamp,
                "frame_path": metadata.get("extracted_frame_path"),
                "transcript": doc
            })
        
        # generate answer using llm
        answer = "context retrieved successfully."
        if self.llm:
            try:
                context_text = "\n\n".join(contexts)
                timestamps_text = ", ".join(timestamps)
                
                prompt_value = self.qa_prompt.format(
                    context=context_text,
                    timestamps=timestamps_text,
                    question=question
                )
                
                response = self.llm.invoke(prompt_value)
                answer = response.content
                
            except Exception as e:
                console.print(f"[red]llm error: {e}[/red]")
                answer = f"error generating response. context: {context_text[:500]}..."
        
        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "num_sources": len(sources)
        }
    
    def summarize_video(self, video_id: Optional[str] = None, max_segments: int = 20) -> str:
        """
        generate a summary of the entire video
        
        args:
            video_id: optional video filter
            max_segments: maximum segments to include in summary
            
        returns:
            video summary text
        """
        console.print("[blue]generating video summary...[/blue]")
        
        # get all segments for the video
        try:
            all_segments = self.vector_db.get_all_segments(video_id)
        except Exception as e:
            return f"error retrieving video segments: {e}"
        
        if not all_segments["documents"]:
            return "no video content found for summary."
        
        # limit segments if too many
        documents = all_segments["documents"][:max_segments]
        metadatas = all_segments["metadatas"][:max_segments]
        
        # prepare segments text
        segments_text = []
        for i, (doc, metadata) in enumerate(zip(documents, metadatas)):
            time_sec = metadata.get("mid_time_sec", 0)
            timestamp = f"{int(time_sec // 60):02d}:{int(time_sec % 60):02d}"
            segments_text.append(f"[{timestamp}] {doc}")
        
        # generate summary using llm
        if not self.llm:
            return f"summary unavailable (no llm). found {len(documents)} segments covering the video content."
        
        try:
            prompt_value = self.summary_prompt.format(
                segments="\n\n".join(segments_text)
            )
            
            response = self.llm.invoke(prompt_value)
            return response.content
            
        except Exception as e:
            console.print(f"[red]llm error: {e}[/red]")
            return f"error generating summary. found {len(documents)} segments in the video."
    
    def get_video_stats(self) -> Dict[str, Any]:
        """get statistics about processed videos"""
        return self.vector_db.get_collection_stats()
    
    def display_query_results(self, results: Dict[str, Any]):
        """display query results in a nice format"""
        
        # display answer
        console.print(Panel(
            results["answer"], 
            title=f"[bold green]answer to: {results['question']}[/bold green]",
            border_style="green"
        ))
        
        # display sources
        if results["sources"]:
            table = Table(title="sources", show_header=True)
            table.add_column("timestamp", style="cyan")
            table.add_column("content", style="white", max_width=60)
            
            for source in results["sources"]:
                table.add_row(
                    source["timestamp"],
                    source["transcript"][:100] + "..." if len(source["transcript"]) > 100 else source["transcript"]
                )
            
            console.print(table)