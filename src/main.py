"""Run the insurance training and held-out prediction workflow."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Sequence

from src.clean_data import clean_data
from src.config import load_config
from src.evaluate import metrics_from_config, scores
from src.infer import infer
from src.load_data import audit_data, load_data
from src.train import save_model, train_model
from src.utils import load_model, write_csv
from src.validate import validate_data

ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger(__name__)


def run(config_path: Path, root: Path = ROOT, raw_path: Path | None = None) -> dict[str, float]:
    """Train on development data and score predictions on the reserved sample."""
    cfg = load_config(config_path)
    paths, data = cfg['paths'], cfg['data']
    source = raw_path if raw_path is not None else root / paths['raw_data']
    outputs = [root / paths['processed_data'], root / paths['model'],
               root / paths['predictions'] / 'predictions.csv']
    resolved = [path.resolve() for path in outputs]
    if len(set(resolved)) != len(resolved) or source.resolve() in resolved or config_path.resolve() in resolved:
        raise ValueError('Output paths must be distinct and must not overwrite input data or configuration')
    raw = load_data(source, required_columns=data['required_columns'])
    audit_data(raw)
    clean = clean_data(raw, data['target'], data['numeric'], data['categorical'])
    validate_data(clean, data['required_columns'], data['target'], data['numeric'],
                  cfg['validation'].get('expected_rows'))
    write_csv(clean, root / paths['processed_data'])
    result = train_model(clean, data['target'], data['numeric'], data['categorical'],
                         cfg['variables']['seed'])
    model_path = root / paths['model']
    save_model(result, model_path, data['target'], data['numeric'], data['categorical'])
    reserved = result.splits.reserved
    predictions = infer(load_model(model_path), reserved)
    metrics = scores(reserved[data['target']], predictions['prediction'], metrics_from_config(cfg))
    write_csv(predictions, root / paths['predictions'] / 'predictions.csv')
    logger.info('Reserved-sample metrics: %s', metrics)
    logger.info('Baseline MAE is not measured; the business quality gate cannot be assessed')
    return metrics


def main(argv: Sequence[str] | None = None) -> None:
    """Parse command line options and execute the pipeline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--data', type=Path, help='Override the configured raw CSV path')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    try:
        run(args.config, args.root, args.data)
    except (OSError, ValueError, KeyError) as exc:
        logger.error('Pipeline stopped: %s', exc)
        raise SystemExit(1) from exc


if __name__ == '__main__':
    main()
