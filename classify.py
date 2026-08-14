import argparse
import json
import sys


def main():
    parser = argparse.ArgumentParser(
        description="TextOrigin — classify text as human, AI-written, or AI-paraphrased"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text string to classify")
    group.add_argument("--file", type=str, help="Path to a plain-text file to classify")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    if args.file:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except (OSError, UnicodeError) as exc:
            print(f"Error: unable to read {args.file}: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        text = args.text

    from src.api.predictor import predict
    from src.errors import TextOriginError

    try:
        result = predict(text)
    except (TextOriginError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    if args.json:
        print(json.dumps(result, indent=2))
        return

    labels = [("human", "Human"), ("ai_written", "AI Written"), ("ai_paraphrased", "AI Paraphrased")]

    print("\n  TextOrigin — Classification Results")
    print("  " + "─" * 38)
    for key, display in labels:
        pct = result[key]
        bar = "█" * int(pct * 30)
        print(f"  {display:18}  {pct:5.1%}  {bar}")
    print()

    if result.get("observed_signals"):
        print("  Heuristic signals observed (separate from the model score):")
        for feat in result["observed_signals"]:
            print(f"    • {feat}")
    if result.get("truncated"):
        print(f"  Warning: input was truncated to {result['analyzed_tokens']} tokens.")
    print()


if __name__ == "__main__":
    main()
