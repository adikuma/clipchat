import os
import sys
import click
from pathlib import Path
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.table import Table

from .rag import VideoRAG

# load environment variables
load_dotenv()

console = Console()

@click.group()
@click.version_option(version="0.1.0")
def cli():
    """
    clipchat  - video rag cli tool

    process videos, extract knowledge, and chat with your content!
    """
    console.print(
        Panel(
            "[bold blue]clipchat [/bold blue]\n[dim]video rag cli tool[/dim]",
            style="blue",
        )
    )


@cli.command()
@click.option("--url", "-u", required=True, help="youtube video url")
@click.option("--chunk-size", "-c", default=7, help="segments per chunk (default: 7)")
@click.option("--stride", "-s", default=3, help="stride between chunks (default: 3)")
@click.option("--data-dir", "-d", default="./data", help="data directory")
@click.option("--force", "-f", is_flag=True, help="force reprocessing")
def process(url, chunk_size, stride, data_dir, force):
    """process a video: download, extract frames, generate embeddings"""

    try:
        # initialize rag system
        rag = VideoRAG(data_dir=data_dir)

        # process video
        video_id = rag.process_video(
            url=url, chunk_size=chunk_size, stride=stride, force_reprocess=force
        )

        console.print(
            f"[green]video processed successfully! video id: {video_id}[/green]"
        )

    except Exception as e:
        console.print(f"[red]error processing video: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--query", "-q", help="question to ask")
@click.option("--video-id", "-v", help="specific video to search in")
@click.option("--results", "-n", default=3, help="number of results (default: 3)")
@click.option("--data-dir", "-d", default="./data", help="data directory")
def ask(query, video_id, results, data_dir):
    """ask a question about processed videos"""

    try:
        # initialize rag system
        rag = VideoRAG(data_dir=data_dir)

        # get query if not provided
        if not query:
            query = Prompt.ask("[blue]what would you like to know?[/blue]")

        # query the system
        response = rag.query(question=query, n_results=results, video_id=video_id)

        # display results
        rag.display_query_results(response)

    except Exception as e:
        console.print(f"[red]error querying: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--video-id", "-v", help="specific video to summarize")
@click.option("--data-dir", "-d", default="./data", help="data directory")
@click.option("--max-segments", "-m", default=20, help="max segments for summary")
def summarize(video_id, data_dir, max_segments):
    """generate a summary of processed video(s)"""

    try:
        # initialize rag system
        rag = VideoRAG(data_dir=data_dir)

        # generate summary
        summary = rag.summarize_video(video_id=video_id, max_segments=max_segments)

        # display summary
        title = f"summary" + (f" - {video_id}" if video_id else "")
        console.print(
            Panel(summary, title=f"[bold blue]{title}[/bold blue]", border_style="blue")
        )

    except Exception as e:
        console.print(f"[red]error generating summary: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--data-dir", "-d", default="./data", help="data directory")
def stats(data_dir):
    """show statistics about processed videos"""

    try:
        # initialize rag system
        rag = VideoRAG(data_dir=data_dir)

        # get stats
        stats_data = rag.get_video_stats()

        # display stats
        table = Table(title="database statistics", show_header=True)
        table.add_column("metric", style="cyan")
        table.add_column("value", style="white")

        for key, value in stats_data.items():
            table.add_row(str(key), str(value))

        console.print(table)

    except Exception as e:
        console.print(f"[red]error getting stats: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--url", "-u", help="youtube video url to process first")
@click.option("--data-dir", "-d", default="./data", help="data directory")
def interactive(url, data_dir):
    """start interactive mode for processing and querying videos"""

    try:
        # initialize rag system
        rag = VideoRAG(data_dir=data_dir)

        console.print(
            Panel(
                "[bold green]interactive mode[/bold green]\n"
                + "type 'help' for commands, 'quit' to exit",
                style="green",
            )
        )

        current_video_id = None

        # process video if url provided
        if url:
            console.print(f"[blue]processing video: {url}[/blue]")
            current_video_id = rag.process_video(url)
            console.print(f"[green]video ready! video id: {current_video_id}[/green]")

        # interactive loop
        while True:
            try:
                command = Prompt.ask("\n[bold cyan]clipchat>[/bold cyan]").strip()

                if command.lower() in ["quit", "exit", "q"]:
                    console.print("[yellow]goodbye! 👋[/yellow]")
                    break

                elif command.lower() == "help":
                    show_interactive_help()

                elif command.lower() == "stats":
                    stats_data = rag.get_video_stats()
                    for key, value in stats_data.items():
                        console.print(f"[cyan]{key}:[/cyan] {value}")

                elif command.lower().startswith("process "):
                    url = command[8:].strip()
                    if url:
                        console.print(f"[blue]processing: {url}[/blue]")
                        current_video_id = rag.process_video(url)
                        console.print(
                            f"[green]done! video id: {current_video_id}[/green]"
                        )
                    else:
                        console.print("[red]please provide a url after 'process'[/red]")

                elif command.lower() == "summary":
                    summary = rag.summarize_video(video_id=current_video_id)
                    console.print(
                        Panel(summary, title="[bold blue]summary[/bold blue]")
                    )

                elif command.lower().startswith("video "):
                    video_id = command[6:].strip()
                    current_video_id = video_id if video_id else None
                    console.print(
                        f"[green]switched to video: {current_video_id or 'all videos'}[/green]"
                    )

                elif command.strip():
                    # treat as a question
                    response = rag.query(question=command, video_id=current_video_id)
                    rag.display_query_results(response)

            except KeyboardInterrupt:
                console.print("\n[yellow]use 'quit' to exit[/yellow]")
            except Exception as e:
                console.print(f"[red]error: {e}[/red]")

    except Exception as e:
        console.print(f"[red]error in interactive mode: {e}[/red]")
        sys.exit(1)


def show_interactive_help():
    """show help for interactive mode"""
    help_text = """
[bold cyan]interactive commands:[/bold cyan]

[green]process <url>[/green]     - process a new video
[green]video <id>[/green]        - switch to specific video (or 'video' for all)
[green]summary[/green]           - get summary of current video
[green]stats[/green]             - show database statistics
[green]help[/green]              - show this help
[green]quit[/green]              - exit interactive mode

[bold cyan]asking questions:[/bold cyan]
just type your question and press enter!

[bold cyan]examples:[/bold cyan]
• what is the main topic?
• explain the key concepts
• what happens at 5:30?
"""
    console.print(Panel(help_text, title="[bold blue]help[/bold blue]"))


@cli.command()
@click.option("--data-dir", "-d", default="./data", help="data directory")
@click.option("--confirm", is_flag=True, help="skip confirmation prompt")
def clear(data_dir, confirm):
    """clear all processed video data"""

    if not confirm:
        confirmed = Confirm.ask(
            "[red]this will delete all processed video data. are you sure?[/red]"
        )
        if not confirmed:
            console.print("[yellow]operation cancelled[/yellow]")
            return

    try:
        # initialize rag system
        rag = VideoRAG(data_dir=data_dir)

        # clear database
        rag.vector_db.clear_collection()

        # remove data directory contents
        data_path = Path(data_dir)
        if data_path.exists():
            import shutil

            shutil.rmtree(data_path)
            data_path.mkdir(exist_ok=True)

        console.print("[green]all data cleared successfully![/green]")

    except Exception as e:
        console.print(f"[red]error clearing data: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    cli()
