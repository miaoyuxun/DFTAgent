"""Tests for XCFunctional class."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from xc.xc_functional import (
    XCFunctional,
    ComponentSpec,
    HFExchange,
    NAME_TO_ID,
    PURE_COMPONENT_ROLES,
)


def test_from_known_pbe():
    xc = XCFunctional.from_known("PBE")
    assert xc.name == "PBE"
    assert len(xc.components) == 2
    assert xc.components[0].name == "GGA_X_PBE"
    assert xc.components[1].name == "GGA_C_PBE"
    assert not xc.is_custom
    assert xc.highest_rung == "GGA"


def test_from_known_pbe0():
    xc = XCFunctional.from_known("PBE0")
    assert xc.name == "PBE0"
    assert xc.hf_exchange.alpha == 0.25
    assert not xc.is_custom


def test_editable_lc_wpbe_scaffold():
    xc = XCFunctional.from_known("HYB_GGA_XC_LC_WPBE")
    assert xc.name == "HYB_GGA_XC_LC_WPBE__editable"
    assert [component.name for component in xc.components] == ["GGA_X_HJS_PBE", "GGA_C_PBE"]
    assert xc.hf_exchange.alpha == 0.0
    assert xc.hf_exchange.beta == 1.0
    assert xc.hf_exchange.omega == 0.4
    assert "GGA_X_HJS_PBE" in xc.to_xc_code()
    assert "LR_HF" in xc.to_xc_code()


def test_expanded_pure_component_catalogue_contains_recovered_components():
    expected = {
        "GGA_X_RPBE",
        "MGGA_X_R4SCAN",
        "LDA_C_PW",
        "LDA_C_PZ",
        "MGGA_C_M06_2X",
        "MGGA_C_R2SCAN",
    }
    assert expected.issubset(PURE_COMPONENT_ROLES)


def test_from_known_b3lyp():
    xc = XCFunctional.from_known("B3LYP")
    assert xc.name == "B3LYP"
    assert xc.hf_exchange.alpha == 0.20
    assert len(xc.components) == 4


def test_from_known_cam_b3lyp():
    xc = XCFunctional.from_known("CAM-B3LYP")
    assert xc.name == "CAM-B3LYP"
    assert xc.hf_exchange.omega == 0.33
    assert xc.hf_exchange.beta == 0.46


def test_from_known_scan():
    xc = XCFunctional.from_known("SCAN")
    assert xc.name == "SCAN"
    assert xc.highest_rung == "MGGA"


def test_from_known_readme_native_aliases():
    for raw_name, expected_name, expected_code in [
        ("TPSSh", "TPSSh", "TPSSh"),
        ("M06 2X", "M06-2X", "M06-2X"),
        ("R2SCAN", "R2SCAN", "R2SCAN"),
        ("r2SCAN0", "r2SCAN0", "R2SCAN0"),
        ("wB97X V", "wB97X-V", "WB97X-V"),
        ("wB97M-V", "wB97M-V", "WB97M-V"),
        ("B97M-V", "B97M-V", "B97M-V"),
    ]:
        xc = XCFunctional.from_known(raw_name)
        assert xc.name == expected_name
        assert xc.metadata.get("native_xc_code") == expected_code
        assert len(xc.components) == 1
        assert xc.components[0].role == "exchange_correlation"


def test_from_known_unknown():
    try:
        XCFunctional.from_known("NONEXISTENT")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_to_xc_code_pbe():
    xc = XCFunctional.from_known("PBE")
    code = xc.to_xc_code()
    assert "PBE" in code
    assert "," in code  # exchange,correlation separator


def test_to_xc_code_empty_side_uses_pyscf_empty_slot():
    xc = XCFunctional(
        name="PBE0_no_corr",
        components=[ComponentSpec(name="GGA_X_PBE", xctype="GGA", mixing_factor=0.78, role="exchange")],
        hf_exchange=HFExchange(alpha=0.22),
    )
    assert xc.to_xc_code() == "0.78*PBE+0.22*HF,"

    xc = XCFunctional(
        name="PBE_corr_only",
        components=[ComponentSpec(name="GGA_C_PBE", xctype="GGA", role="correlation")],
    )
    assert xc.to_xc_code() == ",PBE"


def test_validate_pbe0():
    xc = XCFunctional.from_known("PBE0")
    violations = xc.validate()
    hard_violations = [v for v in violations if v.startswith("HARD:")]
    assert len(hard_violations) == 0, f"PBE0 should have no hard violations, got: {hard_violations}"
    # PBE0 has alpha=0.25 which is above the 0.10 threshold, so no soft SIE warning
    soft_violations = [v for v in violations if v.startswith("SOFT:")]
    assert len(soft_violations) == 0


def test_validate_violation():
    xc = XCFunctional(
        name="BAD",
        components=[ComponentSpec(name="GGA_X_PBE", xctype="GGA", mixing_factor=0.9, role="exchange")],
        hf_exchange=HFExchange(alpha=0.5),
    )
    violations = xc.validate()
    assert len(violations) > 0  # exchange sum 0.9 + 0.5 = 1.4 > 1.0


def test_serialization_roundtrip():
    for name in ["PBE", "PBE0", "B3LYP", "CAM-B3LYP", "SCAN", "SCAN0"]:
        xc = XCFunctional.from_known(name)
        d = xc.to_dict()
        xc2 = XCFunctional.from_dict(d)
        assert xc2.name == xc.name
        assert len(xc2.components) == len(xc.components)
        assert xc2.hf_exchange.alpha == xc.hf_exchange.alpha

        # JSON round-trip
        j = xc.to_json()
        xc3 = XCFunctional.from_json(j)
        assert xc3.name == xc.name


def test_param_vector_pbe0():
    xc = XCFunctional.from_known("PBE0")
    vec = xc.param_vector()
    assert len(vec) > 0
    # Includes PBE X params + PBE C params + alpha/beta/omega
    xc2 = xc.with_param_vector(vec)
    assert xc2.hf_exchange.alpha == xc.hf_exchange.alpha


def test_signature_deterministic():
    xc1 = XCFunctional.from_known("PBE0")
    xc2 = XCFunctional.from_known("PBE0")
    assert xc1.signature() == xc2.signature()


def test_signature_different():
    xc1 = XCFunctional.from_known("PBE0")
    xc2 = XCFunctional.from_known("B3LYP")
    assert xc1.signature() != xc2.signature()


def test_is_custom():
    xc = XCFunctional.from_known("PBE")
    assert not xc.is_custom
    custom = XCFunctional(
        name="CUSTOM",
        components=[ComponentSpec(name="CUSTOM_X", xctype="GGA", role="exchange",
                                   formula="E_x = ...", eval_xc_code="def eval_xc(): pass")],
    )
    assert custom.is_custom


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
