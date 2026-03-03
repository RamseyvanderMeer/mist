"""
MIST CLI - single entry point for all commands.

Usage:
    mist migrate
    mist readiness
    mist build-kg [--incremental] [--output PATH]
    mist train [--config PATH] [--resume PATH] [--device cuda|cpu]
    mist scrape [forum|doc|example] [--limit-items N] [--limit-pages N]
    mist index [--limit N] [--no-resume] [--batch-size N] [--worker-id ID]
    mist extract-titles [--output PATH] [--format csv|json]
    mist feedback stats|export|add|list ...
"""
import sys
import subprocess
from pathlib import Path

import typer

from src.commands import migrate, readiness, knowledge_graph, train, scrape

app = typer.Typer(
    name="mist",
    help="MIST - Multi-modal Intelligent Service Technician. AI-powered automotive diagnostics.",
)


@app.command("migrate")
def migrate_cmd():
    """Run MIST database migrations."""
    raise typer.Exit(migrate.run())


@app.command("readiness")
def readiness_cmd():
    """Check index readiness (ChromaDB, ISTA DB, configs)."""
    raise typer.Exit(readiness.run())


@app.command("build-kg")
def build_kg_cmd(
    incremental: bool = typer.Option(False, "--incremental", "-i", help="Merge with existing graph"),
    output: str | None = typer.Option(None, "--output", "-o", help="Output path for GraphML"),
    db_path: str | None = typer.Option(None, "--db-path", help="Path to BMW diagnostic database"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    """Build knowledge graph from BMW ISTA database."""
    raise typer.Exit(knowledge_graph.run(incremental=incremental, output=output, db_path=db_path, verbose=verbose))


@app.command("train")
def train_cmd(
    config: str | None = typer.Option(None, "--config", "-c", help="Training config path"),
    resume: str | None = typer.Option(None, "--resume", "-r", help="Checkpoint to resume from"),
    embedding_config: str | None = typer.Option(None, "--embedding-config", help="Embedding config path"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
    device: str = typer.Option("auto", "--device", "-d", help="Device: auto, cuda, cpu"),
    data_source: str | None = typer.Option(None, "--data-source", help="db, feedback, or both"),
):
    """Train embeddings from feedback data (contrastive learning)."""
    raise typer.Exit(
        train.run(
            config=config,
            resume=resume,
            embedding_config=embedding_config,
            log_level=log_level,
            device=device,
            data_source=data_source,
        )
    )


@app.command("scrape")
def scrape_cmd(
    spider: str = typer.Argument("forum", help="Spider: forum, doc, example"),
    output_dir: Path = typer.Option(Path("data/training/raw_data"), "--output-dir", "-o"),
    limit_items: int | None = typer.Option(None, "--limit-items", help="Stop after N items"),
    limit_pages: int | None = typer.Option(None, "--limit-pages", help="Stop after N pages"),
    url: str | None = typer.Option(None, "--url", help="Specific URL (forum only)"),
    search: bool = typer.Option(False, "--search", help="Use search-based discovery"),
    targeted: bool = typer.Option(False, "--targeted", help="Targeted subforums"),
    search_codes: bool = typer.Option(False, "--search-codes", help="Search each fault code"),
    re_scrape: bool = typer.Option(False, "--re-scrape", help="Re-process previously scraped URLs"),
):
    """Run web scraper (forum, doc, or example spider)."""
    raise typer.Exit(
        scrape.run(
            spider=spider,
            output_dir=output_dir,
            limit_items=limit_items,
            limit_pages=limit_pages,
            url=url,
            search=search,
            targeted=targeted,
            search_codes=search_codes,
            re_scrape=re_scrape,
        )
    )


def _run_script(script_name: str, extra_args: list[str] | None = None) -> int:
    """Run a script from scripts/ directory. Returns exit code."""
    root = Path(__file__).resolve().parents[2]
    script_path = root / "scripts" / script_name
    if not script_path.exists():
        print(f"Script not found: {script_path}", file=sys.stderr)
        return 1
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd.extend(extra_args)
    result = subprocess.run(cmd, cwd=str(root))
    return result.returncode


@app.command("index")
def index_cmd(
    resume: bool = typer.Option(True, "--resume/--no-resume", help="Resume from checkpoint"),
    batch_size: int = typer.Option(100, "--batch-size", "-b"),
    limit: int | None = typer.Option(None, "--limit", "-n", help="Limit procedures (for testing)"),
    checkpoint_file: str | None = typer.Option(None, "--checkpoint-file"),
    progress_interval: int = typer.Option(100, "--progress-interval"),
    worker_id: str | None = typer.Option(None, "--worker-id", help="Multi-machine worker ID"),
    seed: bool = typer.Option(False, "--seed", help="Seed work queue (multi-machine)"),
    seed_only: bool = typer.Option(False, "--seed-only", help="Only seed and exit"),
    reset_stuck: int = typer.Option(60, "--reset-stuck", help="Reset stuck rows after N minutes"),
):
    """Index repair guides from ISTA DB into vector store."""
    args = []
    if not resume:
        args.append("--no-resume")
    if batch_size != 100:
        args.extend(["--batch-size", str(batch_size)])
    if limit is not None:
        args.extend(["--limit", str(limit)])
    if checkpoint_file:
        args.extend(["--checkpoint-file", checkpoint_file])
    if progress_interval != 100:
        args.extend(["--progress-interval", str(progress_interval)])
    if worker_id:
        args.extend(["--worker-id", worker_id])
    if seed:
        args.append("--seed")
    if seed_only:
        args.append("--seed-only")
    if reset_stuck != 60:
        args.extend(["--reset-stuck", str(reset_stuck)])
    raise typer.Exit(_run_script("index_repair_guides.py", args))


@app.command("extract-titles")
def extract_titles_cmd(
    output: str = typer.Option("data/training/valid_repair_guide_titles.csv", "--output", "-o"),
    format: str = typer.Option("csv", "--format", "-f", help="csv, json, text, agent"),
    include_descriptions: bool = typer.Option(True, "--include-descriptions/--no-descriptions"),
    use_xml_db: bool = typer.Option(True, "--use-xml-db/--no-xml-db"),
    use_llm: bool = typer.Option(False, "--use-llm"),
    llm_provider: str | None = typer.Option(None, "--llm-provider"),
    skip_vector_store: bool = typer.Option(False, "--skip-vector-store"),
    incremental: bool = typer.Option(False, "--incremental"),
):
    """Extract valid repair guide titles for scraping agents."""
    args = ["--output", output, "--format", format]
    if include_descriptions:
        args.append("--include-descriptions")
    if use_xml_db:
        args.append("--use-xml-db")
    else:
        args.append("--no-xml-db")
    if use_llm:
        args.append("--use-llm")
    if llm_provider:
        args.extend(["--llm-provider", llm_provider])
    if skip_vector_store:
        args.append("--skip-vector-store")
    if incremental:
        args.append("--incremental")
    raise typer.Exit(_run_script("extract_repair_guide_titles.py", args))


def main():
    """Entry point when run as python -m src.cli.main."""
    app()


if __name__ == "__main__":
    main()
