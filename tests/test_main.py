"""Exercise the complete pipeline using temporary files."""
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.main import main


def _write_pipeline_inputs(tmp_path: Path, threshold: float) -> Path:
    """Write a small raw CSV and a config gated on mae; return the config path."""
    rows = 80
    frame = pd.DataFrame({
        'age': range(20, 20 + rows), 'sex': ['female', 'male'] * 40,
        'bmi': [20 + i % 13 for i in range(rows)], 'children': [i % 4 for i in range(rows)],
        'smoker': ['yes' if i % 3 == 0 else 'no' for i in range(rows)],
        'region': ['north', 'south'] * 40, 'charges': [1000 + i * 50 for i in range(rows)],
    })
    raw = tmp_path / 'raw.csv'
    frame.to_csv(raw, index=False)
    cfg = {'paths': {'raw_data': 'raw.csv', 'processed_data': 'out/clean.csv',
                     'model': 'out/model.joblib', 'predictions': 'out'},
           'data': {'required_columns': list(frame.columns), 'target': 'charges',
                    'numeric': ['age', 'bmi', 'children'], 'categorical': ['sex', 'smoker', 'region']},
           'validation': {'expected_rows': rows}, 'variables': {'seed': 47},
           'evaluation': {'metrics': ['mae']},
           'quality_gate': {'metric': 'mae', 'threshold': threshold}}
    config = tmp_path / 'config.yaml'
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    return config


def test_main_runs_pipeline(tmp_path: Path, caplog):
    config = _write_pipeline_inputs(tmp_path, threshold=1e9)
    with caplog.at_level('INFO'):
        main(['--root', str(tmp_path), '--config', str(config)])
    assert 'Quality gate passed' in caplog.text
    assert (tmp_path / 'out/model.joblib').is_file()
    assert len(pd.read_csv(tmp_path / 'out/clean.csv')) == 80
    assert len(pd.read_csv(tmp_path / 'out/predictions.csv')) == 8


def test_main_stops_when_quality_gate_fails(tmp_path: Path, caplog):
    config = _write_pipeline_inputs(tmp_path, threshold=0.0)
    with pytest.raises(SystemExit) as exc:
        main(['--root', str(tmp_path), '--config', str(config)])
    assert exc.value.code == 1
    assert 'Quality gate failed' in caplog.text


def test_main_missing_input_stops(tmp_path: Path):
    config = tmp_path / 'config.yaml'
    cfg = {'paths': {'raw_data': 'missing.csv', 'processed_data': 'clean.csv',
                     'model': 'model.joblib', 'predictions': 'out'},
           'data': {'required_columns': []}}
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(SystemExit) as exc:
        main(['--root', str(tmp_path), '--config', str(config)])
    assert exc.value.code == 1


def test_main_refuses_to_overwrite_raw_data(tmp_path: Path):
    raw = tmp_path / 'raw.csv'
    raw.write_text('original', encoding='utf-8')
    config = tmp_path / 'config.yaml'
    cfg = {'paths': {'raw_data': 'raw.csv', 'processed_data': 'raw.csv',
                     'model': 'model.joblib', 'predictions': 'out'}, 'data': {}}
    config.write_text(yaml.safe_dump(cfg), encoding='utf-8')
    with pytest.raises(SystemExit):
        main(['--root', str(tmp_path), '--config', str(config)])
    assert raw.read_text(encoding='utf-8') == 'original'
