"""Component taxonomy — classify XC components by macro_role and family.

Enables cross-functional knowledge retrieval: ablation evidence for
'exchange.gga' can inform any functional that uses a GGA exchange component.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..xc_functional import XCFunctional, ComponentSpec


# ---------------------------------------------------------------------------
# Macro roles — top-level classification
# ---------------------------------------------------------------------------

MACRO_ROLES = [
    "exchange",
    "correlation",
    "hf_exchange",
    "dispersion",
    "constraint",
    "screening",
    "descriptor",
]

# ---------------------------------------------------------------------------
# Family mapping — fine-grained classification within macro_roles
# ---------------------------------------------------------------------------

# Patterns for classifying libxc component names
_FAMILY_RULES: list[tuple[str, str, str]] = [
    # (prefix_pattern, macro_role, family)
    # Exchange
    ("LDA_X",            "exchange", "exchange.lda"),
    ("GGA_X_",           "exchange", "exchange.gga"),
    ("MGGA_X_",          "exchange", "exchange.mgga"),
    ("HYB_GGA_X_",       "exchange", "exchange.hybrid_gga"),
    ("HYB_MGGA_X_",      "exchange", "exchange.hybrid_mgga"),
    # Correlation
    ("LDA_C_",           "correlation", "correlation.lda"),
    ("GGA_C_",           "correlation", "correlation.gga"),
    ("MGGA_C_",          "correlation", "correlation.mgga"),
    # Compound (exchange + correlation)
    ("HYB_GGA_XC_",      "exchange", "exchange.hybrid_gga"),
    ("HYB_MGGA_XC_",     "exchange", "exchange.hybrid_mgga"),
    # Dispersion
    ("DISP_",            "dispersion", "correlation.dispersion"),
    # Custom
    ("CUSTOM_",          "exchange", "exchange.custom"),
    ("DESC_",            "descriptor", "descriptor.gga"),
]

# Special overrides for known names that don't follow prefix patterns
_SPECIAL_CLASSIFICATIONS: dict[str, tuple[str, str]] = {
    "LDA_X":           ("exchange", "exchange.lda"),
    "LDA_C_VWN":       ("correlation", "correlation.lda"),
    "LDA_C_VWN_RPA":   ("correlation", "correlation.lda"),
    "GGA_X_PBE":       ("exchange", "exchange.gga"),
    "GGA_X_PBE_SOL":   ("exchange", "exchange.gga"),
    "GGA_X_B88":       ("exchange", "exchange.gga"),
    "GGA_C_PBE":       ("correlation", "correlation.gga"),
    "GGA_C_LYP":       ("correlation", "correlation.gga"),
    "MGGA_X_SCAN":     ("exchange", "exchange.mgga"),
    "MGGA_C_SCAN":     ("correlation", "correlation.mgga"),
    "MGGA_X_R2SCAN":   ("exchange", "exchange.mgga"),
    "HYB_GGA_XC_B3LYP":    ("exchange", "exchange.hybrid_gga"),
    "HYB_GGA_XC_PBE0":     ("exchange", "exchange.hybrid_gga"),
    "HYB_GGA_XC_CAM_B3LYP":("exchange", "exchange.hybrid_gga"),
    "HYB_MGGA_XC_WB97X":   ("exchange", "exchange.hybrid_mgga"),
}


@dataclass
class ComponentClassification:
    name: str
    macro_role: str
    family: str


def classify_component(name: str, role: str = "") -> ComponentClassification:
    """Classify a component by its name and optional role.

    Args:
        name: libxc canonical name or custom name.
        role: component role from ComponentSpec (exchange, correlation, etc.).

    Returns:
        ComponentClassification with macro_role and family.
    """
    # Check special overrides first
    if name in _SPECIAL_CLASSIFICATIONS:
        macro_role, family = _SPECIAL_CLASSIFICATIONS[name]
        return ComponentClassification(name=name, macro_role=macro_role, family=family)

    # Check prefix rules
    for prefix, macro_role, family in _FAMILY_RULES:
        if name.startswith(prefix):
            return ComponentClassification(name=name, macro_role=macro_role, family=family)

    # Fallback: use the ComponentSpec role if provided
    if role:
        if role == "exchange":
            return ComponentClassification(name=name, macro_role="exchange", family="exchange.unknown")
        elif role == "correlation":
            return ComponentClassification(name=name, macro_role="correlation", family="correlation.unknown")
        elif role == "exchange_correlation":
            return ComponentClassification(name=name, macro_role="exchange", family="exchange.hybrid")
        elif role == "dispersion":
            return ComponentClassification(name=name, macro_role="dispersion", family="correlation.dispersion")
        elif role == "descriptor_correction":
            return ComponentClassification(name=name, macro_role="descriptor", family="descriptor.gga")

    # Ultimate fallback
    return ComponentClassification(name=name, macro_role="unknown", family="unknown")


def extract_component_families(xc: XCFunctional) -> list[ComponentClassification]:
    """Extract classifications for all components in an XCFunctional."""
    classifications = []
    for comp in xc.components:
        cls = classify_component(comp.name, comp.role)
        classifications.append(cls)
    return classifications


def family_summary(xc: XCFunctional) -> dict[str, list[str]]:
    """Summarize component families present in a functional.

    Returns dict mapping family → list of component names.
    """
    classifications = extract_component_families(xc)
    summary: dict[str, list[str]] = {}
    for cls in classifications:
        if cls.family not in summary:
            summary[cls.family] = []
        summary[cls.family].append(cls.name)
    return summary
