"""Testes que provam que o target encoding é CV-safe e faz smoothing correto."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "features"))

import numpy as np
import pandas as pd
import pytest

from target_encoding import (
    compute_target_encoding,
    apply_target_encoding,
    TargetEncoderCV,
)


def test_smoothing_pulls_rare_categories_toward_global_mean():
    """
    Piloto A tem 100 voltas com residual médio -1.0 (sinal forte).
    Piloto B tem 2 voltas com residual médio -5.0 (ruído por poucas amostras).
    Com smoothing=20, piloto B deve ficar perto da média global,
    não em -5.0, porque o prior bayesiano domina.
    """
    df = pd.DataFrame({
        "Driver": ["A"] * 100 + ["B"] * 2,
        "target": [-1.0] * 100 + [-5.0] * 2,
    })
    mapping, gm = compute_target_encoding(df, "target", "Driver", smoothing=20.0)

    # A: (100 * -1.0 + 20 * gm) / 120 — quase não mexe
    assert abs(mapping["A"] - (-1.013)) < 0.02
    # B: (2 * -5.0 + 20 * gm) / 22 — puxado FORTEMENTE do -5
    assert mapping["B"] > -2.0, (
        f"smoothing não puxou B o suficiente: {mapping['B']} "
        f"(sem smoothing seria -5.0)"
    )


def test_unknown_category_falls_back_to_global_mean():
    """Categoria que nunca apareceu no treino deve cair no global_mean."""
    df = pd.DataFrame({"cat": ["A", "A", "B", "B"], "y": [1.0, 2.0, 3.0, 4.0]})
    mapping, gm = compute_target_encoding(df, "y", "cat", smoothing=1.0)

    test_df = pd.DataFrame({"cat": ["A", "B", "C_NEVER_SEEN"]})
    out = apply_target_encoding(test_df, "cat", mapping, gm)

    assert out.iloc[2] == gm, (
        f"categoria nova deveria virar global_mean {gm}, virou {out.iloc[2]}"
    )


def test_cv_safe_never_sees_test_target():
    """
    Teste FORTE de CV-safety: embaralho o target do conjunto de teste
    radicalmente. O encoding aplicado no teste deve ficar IDÊNTICO,
    porque ele foi calculado só a partir do treino. Se mudar, é leakage.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame({
        "Driver": rng.choice(["A", "B", "C", "D"], size=n),
        "Team": rng.choice(["X", "Y", "Z"], size=n),
        "target": rng.normal(0, 1, n),
    })
    tr = df.iloc[:150].copy()
    te = df.iloc[150:].copy()

    enc = TargetEncoderCV(cols=["Driver", "Team"], smoothing=10.0)
    enc.fit(tr, target="target")
    te_enc_orig = enc.transform(te)

    # Envenena o target do teste
    te_poisoned = te.copy()
    te_poisoned["target"] = rng.normal(100, 50, len(te_poisoned))
    te_enc_poisoned = enc.transform(te_poisoned)

    assert np.allclose(te_enc_orig["Driver_te"], te_enc_poisoned["Driver_te"])
    assert np.allclose(te_enc_orig["Team_te"], te_enc_poisoned["Team_te"])


def test_encoder_handles_category_present_only_in_test():
    """Categoria nova no teste não deve quebrar nem virar NaN."""
    tr = pd.DataFrame({
        "Driver": ["A", "B", "A", "B"],
        "target": [1.0, 2.0, 1.5, 2.5],
    })
    te = pd.DataFrame({"Driver": ["A", "B", "ZZZ_NEW"]})

    enc = TargetEncoderCV(cols=["Driver"], smoothing=5.0)
    enc.fit(tr, target="target")
    out = enc.transform(te)

    assert not out["Driver_te"].isna().any()
    assert out["Driver_te"].iloc[2] == pytest.approx(enc.global_means["Driver"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])