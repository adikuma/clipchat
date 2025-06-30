import os
import re
import json
import cv2
import webvtt
import whisper
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from moviepy import VideoFileClip
from yt_dlp import YoutubeDL
from whisper.utils import format_timestamp
from tqdm import tqdm
from rich.console import Console
from rich.progress import Progress

console = Console()

class VideoProcessor:
    """handles video download, transcription, and frame extraction"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # subdirectories
        self.videos_dir = self.data_dir / "videos"
        self.frames_dir = self.data_dir / "frames" 
        self.metadata_dir = self.data_dir / "metadata"
        
        # create subdirs
        for dir_path in [self.videos_dir, self.frames_dir, self.metadata_dir]:
            dir_path.mkdir(exist_ok=True)
    
    def sanitize_filename(self, name: str) -> str:
        """clean filename for safe storage"""
        # remove invalid characters
        cleaned = re.sub(r'[\\/*?:"<>|]', "", name)
        # remove trailing dots and spaces (windows doesn't allow these)
        cleaned = cleaned.rstrip('. ')
        # limit length
        if len(cleaned) > 200:
            cleaned = cleaned[:200]
        # ensure not empty
        if not cleaned:
            cleaned = "video"
        return cleaned
    
    def download_video(self, url: str) -> str:
        """
        download video from youtube url
        
        args:
            url: youtube video url
            
        returns:
            path to downloaded video file
        """
        console.print(f"[blue]downloading video from: {url}[/blue]")
        
        ydl_opts = {
            "format": "bestvideo+bestaudio/best",
            "outtmpl": str(self.videos_dir / "%(title)s.%(ext)s"),
        }
        
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)
            
        console.print(f"[green]video downloaded: {filepath}[/green]")
        return filepath
    
    def write_vtt(self, segments: List[Dict]) -> str:
        """convert whisper segments to vtt format"""
        vtt = "WEBVTT\n\n"
        for i, seg in enumerate(segments, 1):
            start = format_timestamp(seg["start"], always_include_hours=True)
            end = format_timestamp(seg["end"], always_include_hours=True)
            text = seg["text"].strip()
            vtt += f"{i}\n{start} --> {end}\n{text}\n\n"
        return vtt
    
    def generate_transcript(self, video_path: str, model_name: str = "small") -> tuple:
        """
        generate transcript from video using whisper
        
        args:
            video_path: path to video file
            model_name: whisper model size
            
        returns:
            tuple of (transcript_text, audio_path, transcript_path)
        """
        console.print("[blue]generating transcript...[/blue]")
        
        video_dir = Path(video_path).parent
        video_name = Path(video_path).stem
        
        audio_path = video_dir / f"{video_name}_audio.mp3"
        transcript_path = video_dir / f"{video_name}_transcript.vtt"
        
        # extract audio
        console.print("[yellow]extracting audio...[/yellow]")
        clip = VideoFileClip(video_path)
        clip.audio.write_audiofile(str(audio_path), logger=None)
        clip.close()
        
        # whisper transcription
        console.print("[yellow]transcribing with whisper...[/yellow]")
        model = whisper.load_model(model_name)
        opts = dict(task="translate", best_of=1, language="en")
        result = model.transcribe(str(audio_path), **opts)
        
        # write vtt file
        vtt = self.write_vtt(result["segments"])
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(vtt)
        
        console.print(f"[green]transcript saved: {transcript_path}[/green]")
        return result["text"], str(audio_path), str(transcript_path)
    
    def str2seconds(self, time_str: str) -> float:
        """parse hh:mm:ss.mmm to total seconds"""
        t = datetime.strptime(time_str, "%H:%M:%S.%f")
        return (
            t.hour * 3600 +
            t.minute * 60 +
            t.second +
            t.microsecond / 1_000_000
        )
    
    def extract_frames_and_metadata(
        self, 
        video_path: str, 
        transcript_path: str,
        force_reprocess: bool = False
    ) -> List[Dict]:
        """
        extract frames and create metadata from video and transcript
        
        args:
            video_path: path to video file
            transcript_path: path to vtt transcript
            force_reprocess: whether to reprocess existing data
            
        returns:
            list of metadata dictionaries
        """
        video_name = self.sanitize_filename(Path(video_path).stem)
        frames_subdir = self.frames_dir / video_name
        metadata_file = self.metadata_dir / f"{video_name}_metadata.json"
        
        # check if already processed
        if metadata_file.exists() and not force_reprocess:
            console.print("[yellow]loading existing metadata...[/yellow]")
            with open(metadata_file, "r", encoding="utf-8") as f:
                return json.load(f)
        
        # create frame directory
        try:
            frames_subdir.mkdir(exist_ok=True, parents=True)
            console.print(f"[blue]saving frames to: {frames_subdir}[/blue]")
        except Exception as e:
            console.print(f"[red]error creating frame directory {frames_subdir}: {e}[/red]")
            raise
        
        console.print("[blue]extracting frames and metadata...[/blue]")
        
        metadatas = []
        video = cv2.VideoCapture(video_path)
        captions = webvtt.read(transcript_path)
        
        with Progress() as progress:
            task = progress.add_task("extracting frames...", total=len(captions))
            
            for idx, cap in enumerate(captions):
                # parse timestamp
                start_sec = self.str2seconds(cap.start)
                end_sec = self.str2seconds(cap.end)
                mid_sec = (start_sec + end_sec) / 2
                mid_ms = mid_sec * 1000
                
                # extract frame
                video.set(cv2.CAP_PROP_POS_MSEC, mid_ms)
                success, frame = video.read()
                
                if not success:
                    progress.advance(task)
                    continue
                
                # resize frame
                h, w = frame.shape[:2]
                scale = 350 / h
                frame_resized = cv2.resize(frame, (int(w * scale), 350))
                
                # save frame
                img_fname = f"frame_{idx:04d}.jpg"
                img_fpath = frames_subdir / img_fname
                success_write = cv2.imwrite(str(img_fpath), frame_resized)
                
                if not success_write:
                    console.print(f"[red]failed to save frame {img_fpath}[/red]")
                    progress.advance(task)
                    continue
                
                # create metadata
                text = cap.text.replace("\n", " ").strip()
                metadata = {
                    "extracted_frame_path": str(img_fpath),
                    "transcript": text,
                    "video_segment_id": idx,
                    "video_path": video_path,
                    "start_time_sec": start_sec,
                    "end_time_sec": end_sec,
                    "mid_time_sec": mid_sec,
                    "mid_time_ms": mid_ms,
                }
                metadatas.append(metadata)
                
                progress.advance(task)
        
        video.release()
        
        # save metadata
        with open(metadata_file, "w", encoding="utf-8") as f:
            json.dump(metadatas, f, indent=2)
        
        console.print(f"[green]extracted {len(metadatas)} frames[/green]")
        return metadatas
    
    def create_overlapping_chunks(
        self, 
        metadatas: List[Dict], 
        chunk_size: int = 7, 
        stride: int = 3
    ) -> List[Dict]:
        """
        create overlapping chunks from video segments for better context
        
        args:
            metadatas: list of metadata dicts from video segments
            chunk_size: number of segments to include in each chunk
            stride: step size between chunks (smaller = more overlap)
        
        returns:
            list of chunked metadata with combined transcripts
        """
        console.print(f"[blue]creating overlapping chunks (size={chunk_size}, stride={stride})...[/blue]")
        
        chunked_metadatas = []
        
        for i in range(0, len(metadatas), stride):
            # get range for this chunk
            start_idx = max(0, i - chunk_size // 2)
            end_idx = min(len(metadatas), i + chunk_size // 2 + 1)
            
            # ensure minimum chunk size
            if end_idx - start_idx < 3:
                continue
            
            # combine transcripts from multiple segments
            chunk_segments = metadatas[start_idx:end_idx]
            combined_transcript = ' '.join([
                seg['transcript'] for seg in chunk_segments 
                if seg['transcript'].strip()
            ])
            
            # use middle segment as representative
            middle_idx = start_idx + (end_idx - start_idx) // 2
            representative_segment = metadatas[middle_idx]
            
            # verify frame file exists, use first available frame if not
            frame_path = representative_segment['extracted_frame_path']
            if not Path(frame_path).exists():
                # find first existing frame in the chunk
                for seg in chunk_segments:
                    if Path(seg['extracted_frame_path']).exists():
                        frame_path = seg['extracted_frame_path']
                        break
            
            # create chunked metadata
            chunked_metadata = {
                'extracted_frame_path': frame_path,
                'transcript': combined_transcript,
                'video_segment_id': f"chunk_{i:04d}",
                'video_path': representative_segment['video_path'],
                'mid_time_sec': representative_segment['mid_time_sec'],
                'mid_time_ms': representative_segment['mid_time_ms'],
                'chunk_start_time_sec': metadatas[start_idx]['start_time_sec'],
                'chunk_end_time_sec': metadatas[end_idx-1]['end_time_sec'],
                'segments_included': list(range(start_idx, end_idx)),
                'num_segments': end_idx - start_idx
            }
            
            chunked_metadatas.append(chunked_metadata)
        
        console.print(f"[green]created {len(chunked_metadatas)} chunks from {len(metadatas)} segments[/green]")
        return chunked_metadatas
    
    def process_video(
        self, 
        url: str, 
        chunk_size: int = 7, 
        stride: int = 3,
        force_reprocess: bool = False
    ) -> tuple:
        """
        complete video processing pipeline
        
        args:
            url: youtube video url
            chunk_size: segments per chunk
            stride: step between chunks
            force_reprocess: whether to reprocess existing data
            
        returns:
            tuple of (video_path, chunked_metadatas, full_transcript)
        """
        # download video
        video_path = self.download_video(url)
        
        # generate transcript
        full_transcript, audio_path, transcript_path = self.generate_transcript(video_path)
        
        # extract frames and metadata
        metadatas = self.extract_frames_and_metadata(
            video_path, transcript_path, force_reprocess
        )
        
        # create overlapping chunks
        chunked_metadatas = self.create_overlapping_chunks(
            metadatas, chunk_size, stride
        )
        
        return video_path, chunked_metadatas, full_transcript