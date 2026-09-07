"""Version tracking for judge signature optimization."""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from langdmta_eval.alignment.metrics import AlignmentResult


@dataclass
class SignatureVersion:
    """Record of a judge signature version and its alignment scores."""

    version_id: str
    timestamp: str
    judge_type: str
    optimizer_type: str
    n_training_examples: int
    per_metric_kappa: Dict[str, float]
    overall_kappa: float
    model_path: Optional[str] = None
    notes: str = ""
    config: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def from_alignment_results(
        version_id: str,
        results: Dict[str, AlignmentResult],
        judge_type: str,
        optimizer_type: str,
        n_training_examples: int,
        model_path: Optional[str] = None,
        notes: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> "SignatureVersion":
        """Create a SignatureVersion from alignment results.

        Args:
            version_id: Version identifier (e.g., 'v1_baseline')
            results: Dict mapping metric_name -> AlignmentResult
            judge_type: Judge type used
            optimizer_type: Optimizer type used (or 'none' for baseline)
            n_training_examples: Number of training examples
            model_path: Path to saved DSPy module
            notes: Free-text notes
            config: Serialized OptimizationConfig

        Returns:
            SignatureVersion instance
        """
        per_metric = {}
        kappa_values = []
        for metric, r in results.items():
            k = r.cohens_kappa_weighted
            per_metric[metric] = k if not np.isnan(k) else None
            if not np.isnan(k):
                kappa_values.append(k)

        overall = float(np.mean(kappa_values)) if kappa_values else None

        return SignatureVersion(
            version_id=version_id,
            timestamp=datetime.now().isoformat(),
            judge_type=judge_type,
            optimizer_type=optimizer_type,
            n_training_examples=n_training_examples,
            per_metric_kappa=per_metric,
            overall_kappa=overall,
            model_path=model_path,
            notes=notes,
            config=config or {},
        )


def save_version_history(versions: List[SignatureVersion], path: str) -> None:
    """Save version history to JSON.

    Args:
        versions: List of SignatureVersion records
        path: Output JSON file path
    """
    data = [asdict(v) for v in versions]
    Path(path).write_text(json.dumps(data, indent=2, default=str))


def load_version_history(path: str) -> List[SignatureVersion]:
    """Load version history from JSON.

    Args:
        path: Path to JSON file

    Returns:
        List of SignatureVersion records
    """
    data = json.loads(Path(path).read_text())
    return [SignatureVersion(**d) for d in data]


def version_comparison_table(versions: List[SignatureVersion]) -> pd.DataFrame:
    """Create a comparison table of signature versions (Table 6).

    Args:
        versions: List of SignatureVersion records

    Returns:
        DataFrame with version_id, optimizer, per-metric kappas, overall kappa
    """
    rows = []
    for v in versions:
        row = {
            "version": v.version_id,
            "optimizer": v.optimizer_type,
            "n_examples": v.n_training_examples,
        }
        row.update({f"kappa_{k}": val for k, val in v.per_metric_kappa.items()})
        row["overall_kappa"] = v.overall_kappa
        row["notes"] = v.notes
        rows.append(row)

    return pd.DataFrame(rows)
