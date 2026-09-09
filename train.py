"""Run: python train.py --data data/reviews.csv"""
import argparse
import json
from pathlib import Path
from model import train

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(__file__).parent / 'data/reviews.csv')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'reports')
    args = parser.parse_args()
    try:
        _, report, predictions, _, _ = train(args.data.read_bytes())
    except (ValueError, OSError) as exc:
        parser.exit(1, f'Error: {exc}\n')
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    predictions.to_csv(args.output / 'test_predictions.csv', index=False)
    print(f"Training rows: {report['train_rows']}; test rows: {report['test_rows']}")
    print(f"Accuracy: {report['accuracy']:.3f}; macro-F1: {report['macro_f1']:.3f}")
    print(f"Majority baseline accuracy: {report['baseline_accuracy']:.3f}")
    print(report['note'])
