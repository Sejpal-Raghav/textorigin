from __future__ import annotations

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

from src.services.rewrite import rewrite_document


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="humanizer",
        description="Classify, rewrite, and compare English text with TextOrigin.",
    )
    subcommands = parser.add_subparsers(dest="command")
    rewrite = subcommands.add_parser("rewrite", help="Apply guarded local edits")
    inputs = rewrite.add_mutually_exclusive_group()
    inputs.add_argument("--text", help="Text to rewrite")
    inputs.add_argument("--file", type=Path, help="UTF-8 text file to rewrite")
    rewrite.add_argument("--style", choices=("natural", "formal", "casual"), default="natural")
    rewrite.add_argument("--strength", choices=("light", "medium"), default="light")
    rewrite.add_argument(
        "--provider",
        choices=("local", "deepseek"),
        default="local",
        help="Optional polish provider; DeepSeek sends text externally",
    )
    rewrite.add_argument("--format", choices=("text", "json"), default="text")
    rewrite.add_argument("--output", type=Path, help="Write output to this file")
    rewrite.add_argument("--force", action="store_true", help="Replace an existing output file")

    classify = subcommands.add_parser("classify", help="Classify text with the installed model")
    classify_inputs = classify.add_mutually_exclusive_group()
    classify_inputs.add_argument("--text")
    classify_inputs.add_argument("--file", type=Path)
    classify.add_argument("--format", choices=("text", "json"), default="text")

    compare = subcommands.add_parser("compare", help="Rewrite and compare classifier scores")
    compare_inputs = compare.add_mutually_exclusive_group()
    compare_inputs.add_argument("--text")
    compare_inputs.add_argument("--file", type=Path)
    compare.add_argument("--style", choices=("natural", "formal", "casual"), default="natural")
    compare.add_argument("--strength", choices=("light", "medium"), default="light")
    compare.add_argument("--format", choices=("text", "json"), default="text")
    return parser


def _read_text(args: argparse.Namespace) -> str:
    if getattr(args, "text", None) is not None:
        value = args.text
    elif getattr(args, "file", None) is not None:
        try:
            value = args.file.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise ValueError(f"unable to read {args.file}: {exc}") from exc
    elif not sys.stdin.isatty():
        value = sys.stdin.read()
    else:
        raise ValueError("provide --text, --file, or pipe text on stdin")
    if not value.strip():
        raise ValueError("text cannot be empty")
    return value


def _write_result(value: str, output: Path | None, force: bool) -> None:
    if output is None:
        print(value)
        return
    if output.exists() and not force:
        raise ValueError(f"output file already exists: {output}; pass --force to replace it")
    output.write_text(value, encoding="utf-8")


def _classify(text: str) -> dict:
    from src.api.predictor import predict

    return predict(text)


def _display_scores(scores: dict) -> str:
    return "\n".join(
        (
            f"Human: {scores['human']:.1%}",
            f"AI written: {scores['ai_written']:.1%}",
            f"AI paraphrased: {scores['ai_paraphrased']:.1%}",
        )
    )


def _interactive_args() -> list[str]:
    print("TextOrigin Humanizer")
    operation = input("Operation [rewrite/classify/compare] (rewrite): ").strip() or "rewrite"
    text = input("Text: ")
    args = [operation, "--text", text]
    if operation in {"rewrite", "compare"}:
        style = input("Style [natural/formal/casual] (natural): ").strip() or "natural"
        strength = input("Strength [light/medium] (light): ").strip() or "light"
        args.extend(["--style", style, "--strength", strength])
    if operation == "rewrite":
        provider = input("Provider [local/deepseek] (local): ").strip() or "local"
        args.extend(["--provider", provider])
    return args


def _deepseek_rewrite(text: str, style: str, strength: str) -> dict:
    from src.bootstrap import build_services
    from src.config import get_settings

    services = build_services(get_settings())
    credential = os.getenv("DEEPSEEK_API_KEY")
    if not credential and sys.stdin.isatty():
        credential = getpass.getpass("DeepSeek API key (masked): ").strip()
    if not credential:
        services.close()
        raise ValueError("Configure DEEPSEEK_API_KEY, or use an interactive terminal for masked entry")
    try:
        return services.humanization.humanize(
            text,
            use_llm=True,
            style=style,
            strength=strength,
            api_key=credential,
        )
    finally:
        services.close()


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command is None:
        if argv is None and sys.stdin.isatty():
            return main(_interactive_args())
        parser.print_help()
        return 2
    try:
        text = _read_text(args)
        if args.command == "rewrite":
            if args.provider == "deepseek":
                result = _deepseek_rewrite(text, args.style, args.strength)
                revised = result["humanized_text"]
            else:
                revised = rewrite_document(text, style=args.style, strength=args.strength)
                result = {
                    "status": "completed",
                    "warnings": [],
                    "complete": True,
                }
            rendered = (
                json.dumps(
                    {
                        "schema_version": "1.0",
                        "operation": "rewrite",
                        "status": result["status"],
                        "style": args.style,
                        "strength": args.strength,
                        "provider": args.provider,
                        "original_text": text,
                        "humanized_text": revised,
                        "warnings": result.get("warnings", []),
                        "complete": result.get("complete", True),
                    },
                    ensure_ascii=False,
                )
                if args.format == "json"
                else revised
            )
            _write_result(rendered, args.output, args.force)
            return 0
        if args.command == "classify":
            scores = _classify(text)
            print(json.dumps(scores, ensure_ascii=False) if args.format == "json" else _display_scores(scores))
            return 0

        from src.bootstrap import build_services
        from src.config import get_settings
        services = build_services(get_settings())
        try:
            result = services.comparison.compare(text, style=args.style, strength=args.strength)
        finally:
            services.close()
        revised = result["humanized_text"]
        original_scores = result["original_scores"]
        revised_scores = result["revised_scores"]
        print(json.dumps(result, ensure_ascii=False) if args.format == "json" else f"{revised}\n\nOriginal\n{_display_scores(original_scores)}\n\nRevised\n{_display_scores(revised_scores)}")
        return 0
    except KeyboardInterrupt:
        print("Cancelled.", file=sys.stderr)
        return 130
    except FileExistsError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        from src.errors import ModelUnavailableError, TextOriginError

        print(f"Error: {exc}", file=sys.stderr)
        if isinstance(exc, ModelUnavailableError):
            return 3
        return 1 if isinstance(exc, TextOriginError) else 1


if __name__ == "__main__":
    raise SystemExit(main())
