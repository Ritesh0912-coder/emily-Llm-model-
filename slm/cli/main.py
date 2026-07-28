"""
CLI entrypoints for Emily SLM.

Commands:
    emily-train    — train a model from a config file
    emily-eval     — evaluate a checkpoint
    emily-chat     — interactive chat or API server
    emily-prepare  — prepare and tokenise a dataset
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    from slm.utils.logging import setup_logger
    setup_logger("slm", level=level)


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------

@click.command("emily-train")
@click.option("--config", "-c", required=True, type=click.Path(exists=True), help="Path to YAML config file")
@click.option("--resume", "-r", type=click.Path(), default=None, help="Checkpoint dir or file to resume from")
@click.option("--data", "-d", type=click.Path(), default=None, help="Override dataset directory")
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable DEBUG logging")
def train(config: str, resume: str | None, data: str | None, verbose: bool) -> None:
    """Train an Emily SLM model from a YAML configuration file."""
    _setup_logging(verbose)
    from slm.config import EmilyConfig
    from slm.model import EmilySLM
    from slm.tokenizer import EmilyTokenizer
    from slm.dataset.loader import DatasetLoader
    from slm.training.trainer import EmilyTrainer

    console.print(Panel.fit("[bold cyan]🧠 Emily SLM Training[/bold cyan]", border_style="cyan"))

    cfg = EmilyConfig.from_yaml(config)
    console.print(f"Config loaded: [green]{config}[/green]")
    console.print(f"Model: [yellow]{cfg.model.name}[/yellow] | "
                  f"layers={cfg.model.n_layers} | d_model={cfg.model.d_model}")

    # Load tokenizer
    tok_path = cfg.tokenizer.model_path
    if not Path(tok_path).exists():
        console.print(f"[red]Tokenizer not found at {tok_path}[/red]")
        console.print("Train a tokenizer first: emily-prepare --config <config>")
        sys.exit(1)
    tokenizer = EmilyTokenizer.load(tok_path)

    # Load datasets
    dataset_dir = Path(data) if data else None
    train_bin = (dataset_dir / "train.bin") if dataset_dir else Path(cfg.dataset.train_path)
    val_bin   = (dataset_dir / "val.bin")   if dataset_dir else Path(cfg.dataset.val_path)

    loader = DatasetLoader(tokenizer, seq_len=cfg.dataset.max_seq_len)
    train_ds = loader.from_binary(train_bin)
    val_ds   = loader.from_binary(val_bin) if val_bin.exists() else None

    if val_ds is None:
        console.print("[yellow]No val.bin found — skipping validation[/yellow]")

    # Build model
    model = EmilySLM(cfg.model)
    console.print(f"[green]{model}[/green]")

    # Train
    trainer = EmilyTrainer(
        model=model,
        config=cfg,
        train_dataset=train_ds,
        val_dataset=val_ds,
        resume_from=resume,
    )
    results = trainer.train()
    console.print(f"\n✅ Training complete. Final step: {results['final_step']:,}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@click.command("emily-eval")
@click.option("--model", "-m", required=True, type=click.Path(), help="Model directory or checkpoint")
@click.option("--tokenizer", "-t", required=True, type=click.Path(exists=True), help="Tokenizer JSON path")
@click.option("--data", "-d", required=True, type=click.Path(exists=True), help="Validation .bin file")
@click.option("--batch-size", "-b", default=16, type=int, help="Evaluation batch size")
@click.option("--seq-len", default=512, type=int, help="Sequence length")
@click.option("--verbose", "-v", is_flag=True, default=False)
def evaluate(model: str, tokenizer: str, data: str, batch_size: int, seq_len: int, verbose: bool) -> None:
    """Evaluate a trained Emily SLM checkpoint."""
    _setup_logging(verbose)
    from slm.model import EmilySLM
    from slm.tokenizer import EmilyTokenizer
    from slm.dataset.loader import DatasetLoader
    from slm.evaluation.evaluator import EmilyEvaluator

    console.print(Panel.fit("[bold green]📊 Emily SLM Evaluation[/bold green]", border_style="green"))

    tok = EmilyTokenizer.load(tokenizer)
    slm_model = EmilySLM.from_pretrained(model)
    slm_model.eval()

    loader = DatasetLoader(tok, seq_len=seq_len)
    val_ds = loader.from_binary(data)

    evaluator = EmilyEvaluator(slm_model, tok)
    metrics = evaluator.evaluate(val_ds, batch_size=batch_size)

    console.print("\n[bold]Results:[/bold]")
    for k, v in metrics.items():
        console.print(f"  {k}: [cyan]{v:.4f}[/cyan]" if isinstance(v, float) else f"  {k}: [cyan]{v}[/cyan]")


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------

@click.command("emily-chat")
@click.option("--model", "-m", required=True, type=click.Path(), help="Model directory")
@click.option("--tokenizer", "-t", required=True, type=click.Path(exists=True), help="Tokenizer JSON path")
@click.option("--temperature", default=0.8, type=float, help="Sampling temperature")
@click.option("--max-tokens", default=256, type=int, help="Max new tokens per response")
@click.option("--serve", is_flag=True, default=False, help="Launch API server instead of REPL")
@click.option("--host", default="0.0.0.0", help="API server host (--serve only)")
@click.option("--port", default=8000, type=int, help="API server port (--serve only)")
@click.option("--verbose", "-v", is_flag=True, default=False)
def chat(
    model: str, tokenizer: str, temperature: float,
    max_tokens: int, serve: bool, host: str, port: int, verbose: bool
) -> None:
    """Interactive chat REPL or launch the REST API server."""
    _setup_logging(verbose)

    if serve:
        # Launch FastAPI server
        os.environ["EMILY_MODEL_PATH"] = model
        os.environ["EMILY_TOKENIZER_PATH"] = tokenizer
        import uvicorn
        console.print(f"[bold cyan]🚀 Emily API starting on http://{host}:{port}[/bold cyan]")
        uvicorn.run("slm.api.app:app", host=host, port=port, reload=False)
        return

    # Interactive REPL
    from slm.inference.engine import EmilyInferenceEngine
    engine = EmilyInferenceEngine(model_path=model, tokenizer_path=tokenizer)

    console.print(Panel.fit(
        "[bold magenta]🧠 Emily SLM Chat[/bold magenta]\n"
        "Type your message and press Enter. Type [bold]exit[/bold] to quit.",
        border_style="magenta",
    ))

    history: list[dict[str, str]] = []
    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if user_input.lower() in ("exit", "quit", "q"):
            console.print("[dim]Goodbye![/dim]")
            break
        if not user_input:
            continue

        history.append({"role": "user", "content": user_input})

        console.print("[bold green]Emily:[/bold green] ", end="")
        response_parts: list[str] = []
        try:
            for chunk in engine.stream(user_input, max_new_tokens=max_tokens, temperature=temperature):
                console.print(chunk, end="", highlight=False)
                response_parts.append(chunk)
        except Exception as exc:
            console.print(f"[red]Error: {exc}[/red]")
            continue
        console.print()
        history.append({"role": "assistant", "content": "".join(response_parts)})


# ---------------------------------------------------------------------------
# prepare_data
# ---------------------------------------------------------------------------

@click.command("emily-prepare")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Input text file or directory of .txt/.jsonl files")
@click.option("--output", "-o", default="datasets/tokenized", type=click.Path(),
              help="Output directory for train.bin / val.bin")
@click.option("--config", "-c", required=True, type=click.Path(exists=True),
              help="YAML config file (for vocab_size and tokenizer settings)")
@click.option("--val-ratio", default=0.05, type=float, help="Fraction held out for validation")
@click.option("--train-tokenizer", is_flag=True, default=False,
              help="Also train the tokenizer from the data")
@click.option("--verbose", "-v", is_flag=True, default=False)
def prepare_data(
    input_path: str, output: str, config: str,
    val_ratio: float, train_tokenizer: bool, verbose: bool
) -> None:
    """Tokenise raw text data and write binary train/val files."""
    _setup_logging(verbose)
    from slm.config import EmilyConfig
    from slm.tokenizer import EmilyTokenizer
    from slm.dataset.preprocessor import TextPreprocessor
    from slm.dataset.loader import DatasetLoader

    console.print(Panel.fit("[bold yellow]📦 Emily Data Preparation[/bold yellow]", border_style="yellow"))

    cfg = EmilyConfig.from_yaml(config)
    pp = TextPreprocessor(min_length=20, dedup=True)

    # Collect texts
    input_p = Path(input_path)
    texts: list[str] = []
    if input_p.is_file():
        raw = input_p.read_text(encoding="utf-8", errors="replace")
        paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
        texts = pp.process(paragraphs)
    elif input_p.is_dir():
        for f in sorted(input_p.rglob("*.txt")) + sorted(input_p.rglob("*.jsonl")):
            raw = f.read_text(encoding="utf-8", errors="replace")
            paragraphs = [p.strip() for p in raw.split("\n\n") if p.strip()]
            texts.extend(pp.process(paragraphs))

    console.print(f"Collected [cyan]{len(texts):,}[/cyan] text segments")

    # Train or load tokenizer
    tok_path = Path(cfg.tokenizer.model_path)
    if train_tokenizer or not tok_path.exists():
        console.print(f"Training BPE tokenizer (vocab_size={cfg.tokenizer.vocab_size})…")
        tokenizer = EmilyTokenizer.train(texts, vocab_size=cfg.tokenizer.vocab_size)
        tok_path.parent.mkdir(parents=True, exist_ok=True)
        tokenizer.save(str(tok_path))
        console.print(f"Tokenizer saved → [green]{tok_path}[/green]")
    else:
        tokenizer = EmilyTokenizer.load(str(tok_path))
        console.print(f"Tokenizer loaded ← [green]{tok_path}[/green]")

    # Tokenise and save
    train_p, val_p = DatasetLoader.tokenise_and_save(
        texts, tokenizer, output_path=output, val_ratio=val_ratio
    )
    console.print(f"✅ Done! train → [green]{train_p}[/green] | val → [green]{val_p}[/green]")


# ---------------------------------------------------------------------------
# Group / alias exports for pyproject.toml scripts
# ---------------------------------------------------------------------------

def main_train() -> None:
    train(standalone_mode=True)

def main_eval() -> None:
    evaluate(standalone_mode=True)

def main_chat() -> None:
    chat(standalone_mode=True)

def main_prepare() -> None:
    prepare_data(standalone_mode=True)
