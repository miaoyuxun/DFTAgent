"""XCFunctional: three-layer spec (name, formula, eval_xc) for DFT functionals."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# libxc parameter catalogue (from xc_functional_parameter_space.md §2-8, §14.3)
# key = libxc integer ID
# ext_params: ordered list of (name, default, (lo, hi))
# ---------------------------------------------------------------------------

LIBXC_PARAMS: dict[int, list[tuple[str, float, tuple[float, float]]]] = {
    1: [],   # LDA_X
    7: [],   # LDA_C_VWN (VWN5)
    28: [],  # LDA_C_VWN_RPA (VWN3, used in B3LYP)
    101: [   # GGA_X_PBE — order must match libxc C struct: [kappa, mu]
        ("_kappa", 0.8040,        (0.6,  1.5)),
        ("_mu",    0.21951497268, (0.10, 0.30)),
    ],
    130: [   # GGA_C_PBE — order must match libxc C struct: [beta, gamma, d]
        ("_beta",  0.06672455060314922, (0.03, 0.10)),
        ("_gamma", 0.031090690869654895, (0.01, 0.05)),
        ("_d",     1.0,                   (0.5,  1.5)),
    ],
    106: [   # GGA_X_B88 — order must match libxc C struct
        ("_beta",  0.0042, (0.002, 0.010)),
        ("_gamma", 6.0,    (4.0,  8.0)),
    ],
    131: [   # GGA_C_LYP — order must match libxc C struct
        ("_a", 0.04918, (0.03, 0.07)),
        ("_b", 0.132,   (0.08, 0.20)),
        ("_c", 0.2533,  (0.15, 0.35)),
        ("_d", 0.349,   (0.25, 0.50)),
    ],
    116: [   # GGA_X_PBE_SOL — order must match libxc C struct: [kappa, mu]
        ("_kappa", 0.123, (0.05, 0.30)),
        ("_mu",    0.10,  (0.05, 0.20)),
    ],
    263: [   # MGGA_X_SCAN — order must match libxc C struct
        ("_c1", 0.667, (0.5, 0.8)),
        ("_c2", 0.800, (0.6, 1.0)),
        ("_d",  1.240, (1.0, 1.5)),
        ("_k1", 0.065, (0.04, 0.10)),
    ],
    267: [],  # MGGA_C_SCAN
    497: [],  # MGGA_X_R2SCAN
    # Additional pure exchange/correlation components recovered from the
    # reference benchmark panel. LibXC defaults are used initially.
    9: [],    # LDA_C_PZ
    12: [],   # LDA_C_PW
    49: [],   # GGA_X_PBE_MOL
    74: [],   # MGGA_C_MN12_L
    102: [],  # GGA_X_PBE_R
    117: [],  # GGA_X_RPBE
    132: [],  # GGA_C_P86
    133: [],  # GGA_C_PBE_SOL
    184: [],  # GGA_X_APBE
    186: [],  # GGA_C_APBE
    202: [],  # MGGA_X_TPSS
    203: [],  # MGGA_X_M06_L
    212: [],  # MGGA_X_REVTPSS
    227: [],  # MGGA_X_MN12_L
    231: [],  # MGGA_C_TPSS
    233: [],  # MGGA_C_M06_L
    235: [],  # MGGA_C_M06
    236: [],  # MGGA_C_M06_2X
    493: [],  # MGGA_X_RSCAN
    498: [],  # MGGA_C_R2SCAN
    650: [],  # MGGA_X_R4SCAN
    525: [],  # GGA_X_HJS_PBE (short-range PBE exchange for LC-wPBE)
    # Virtual HLE16 split components. Libxc exposes HCTH/407 and HLE16 only
    # as compound XC functionals. The unwanted channel is zeroed so the two
    # distinct Libxc IDs can be combined as independently editable X and C
    # components while reproducing native HLE16 exactly.
    545: [
        ("_cx0", 1.3523, (1.3523, 1.3523)),
        ("_cx1", -0.64792375, (-0.64792375, -0.64792375)),
        ("_cx2", 4.282025, (4.282025, 4.282025)),
        ("_cx3", -3.2862625, (-3.2862625, -3.2862625)),
        ("_cx4", 2.8606875, (2.8606875, 2.8606875)),
        ("_css0", 0.0, (0.0, 0.0)),
        ("_css1", 0.0, (0.0, 0.0)),
        ("_css2", 0.0, (0.0, 0.0)),
        ("_css3", 0.0, (0.0, 0.0)),
        ("_css4", 0.0, (0.0, 0.0)),
        ("_cos0", 0.0, (0.0, 0.0)),
        ("_cos1", 0.0, (0.0, 0.0)),
        ("_cos2", 0.0, (0.0, 0.0)),
        ("_cos3", 0.0, (0.0, 0.0)),
        ("_cos4", 0.0, (0.0, 0.0)),
    ],
    164: [
        ("_cx0", 0.0, (0.0, 0.0)),
        ("_cx1", 0.0, (0.0, 0.0)),
        ("_cx2", 0.0, (0.0, 0.0)),
        ("_cx3", 0.0, (0.0, 0.0)),
        ("_cx4", 0.0, (0.0, 0.0)),
        ("_css0", 0.593885, (0.593885, 0.593885)),
        ("_css1", -1.20146, (-1.20146, -1.20146)),
        ("_css2", 2.808705, (2.808705, 2.808705)),
        ("_css3", -4.589615, (-4.589615, -4.589615)),
        ("_css4", 3.12399, (3.12399, 3.12399)),
        ("_cos0", 0.294538, (0.294538, 0.294538)),
        ("_cos1", 2.21187, (2.21187, 2.21187)),
        ("_cos2", -9.6109, (-9.6109, -9.6109)),
        ("_cos3", 21.28605, (21.28605, 21.28605)),
        ("_cos4", -21.0026, (-21.0026, -21.0026)),
    ],
    402: [   # HYB_GGA_XC_B3LYP — compound functional
        ("_ax",    0.20, (0.10, 0.30)),
        ("_ss1",   0.72, (0.50, 0.90)),
        ("_ss2",   0.81, (0.60, 0.95)),
    ],
    406: [   # HYB_GGA_XC_PBE0 — compound functional
        ("_alpha", 0.25, (0.10, 0.40)),
    ],
    433: [   # HYB_GGA_XC_CAM_B3LYP — compound functional
        ("_ss2",   0.81, (0.60, 0.95)),
        ("_alpha", 0.65, (0.40, 0.80)),
        ("_beta", -0.46, (-0.80, -0.10)),
        ("_omega", 0.33, (0.20, 0.50)),
    ],
    464: [   # HYB_MGGA_XC_WB97X — compound functional (18 params)
        ("_c0_0",   0.842294,  (0.5, 1.0)),
        ("_c0_1",   0.726479,  (0.4, 1.0)),
        ("_c0_2",   1.0476,    (0.5, 2.0)),
        ("_c0_3",  -5.70635,   (-10.0, 0.0)),
        ("_c0_4",  13.2794,    (5.0, 20.0)),
        ("_c0_5",   1.0,       (0.5, 1.5)),
        ("_c0_6",  -4.33879,   (-8.0, 0.0)),
        ("_c0_7",  18.2308,    (10.0, 30.0)),
        ("_c0_8", -31.743,     (-50.0, -10.0)),
        ("_c0_9",  17.2901,    (10.0, 30.0)),
        ("_c0_10",  1.0,       (0.5, 1.5)),
        ("_c0_11",  2.37031,   (1.0, 5.0)),
        ("_c0_12",-11.3995,    (-20.0, -5.0)),
        ("_c0_13",  6.58405,   (3.0, 10.0)),
        ("_c0_14", -3.78132,   (-6.0, 0.0)),
        ("_c0_15",  1.0,       (0.5, 1.5)),
        ("_c0_16", -0.842294,  (-1.0, 0.0)),
        ("_c0_17",  0.30,      (0.20, 0.50)),
    ],
}

# PySCF short aliases for common libxc names
PYSOF_ALIAS: dict[str, str] = {
    "LDA_X": "SLATER",
    "LDA_C_VWN": "VWN",
    "LDA_C_VWN_RPA": "VWN3",
    "GGA_X_PBE": "PBE",
    "GGA_C_PBE": "PBE",
    "GGA_X_B88": "B88",
    "GGA_C_LYP": "LYP",
    "GGA_X_PBE_SOL": "PBE_SOL",
    "MGGA_X_SCAN": "SCAN",
    "MGGA_C_SCAN": "SCAN",
    "MGGA_X_R2SCAN": "R2SCAN",
    "HYB_GGA_XC_B3LYP": "B3LYP",
    "HYB_GGA_XC_PBE0": "PBE0",
    "HYB_GGA_XC_CAM_B3LYP": "CAM-B3LYP",
    "HYB_MGGA_XC_WB97X": "wB97X",
    "GGA_X_HCTH_407_HLE16": "GGA_XC_HLE16",
    "GGA_C_HCTH_407_HLE16": "GGA_XC_HCTH_407",
}

# Reverse lookup: libxc canonical name -> libxc ID
NAME_TO_ID: dict[str, int] = {
    "LDA_X": 1,
    "LDA_C_VWN": 7,
    "LDA_C_VWN_RPA": 28,
    "GGA_X_PBE": 101,
    "GGA_C_PBE": 130,
    "GGA_X_B88": 106,
    "GGA_C_LYP": 131,
    "GGA_X_PBE_SOL": 116,
    "MGGA_X_SCAN": 263,
    "MGGA_C_SCAN": 267,
    "MGGA_X_R2SCAN": 497,
    "HYB_GGA_XC_B3LYP": 402,
    "HYB_GGA_XC_PBE0": 406,
    "HYB_GGA_XC_CAM_B3LYP": 433,
    "HYB_MGGA_XC_WB97X": 464,
    "GGA_X_HCTH_407_HLE16": 545,
    "GGA_C_HCTH_407_HLE16": 164,
    "LDA_C_PZ": 9,
    "LDA_C_PW": 12,
    "GGA_X_PBE_MOL": 49,
    "MGGA_C_MN12_L": 74,
    "GGA_X_PBE_R": 102,
    "GGA_X_RPBE": 117,
    "GGA_C_P86": 132,
    "GGA_C_PBE_SOL": 133,
    "GGA_X_APBE": 184,
    "GGA_C_APBE": 186,
    "MGGA_X_TPSS": 202,
    "MGGA_X_M06_L": 203,
    "MGGA_X_REVTPSS": 212,
    "MGGA_X_MN12_L": 227,
    "MGGA_C_TPSS": 231,
    "MGGA_C_M06_L": 233,
    "MGGA_C_M06": 235,
    "MGGA_C_M06_2X": 236,
    "MGGA_X_RSCAN": 493,
    "MGGA_C_R2SCAN": 498,
    "MGGA_X_R4SCAN": 650,
    "GGA_X_HJS_PBE": 525,
}

# Additional pure exchange/correlation components observed in the benchmark
# panel. These expand the add/replace operation space without adding tunable
# ext_params for components whose parameter ranges are not curated here.
_OBSERVED_PURE_COMPONENT_IDS: dict[str, int] = {
    "GGA_C_ACGGA": 39,
    "GGA_C_ACGGAP": 176,
    "GGA_C_AM05": 135,
    "GGA_C_BMK": 280,
    "GGA_C_CCDF": 313,
    "GGA_C_CHACHIYO": 309,
    "GGA_C_CS1": 565,
    "GGA_C_FT97": 88,
    "GGA_C_GAM": 33,
    "GGA_C_GAPC": 555,
    "GGA_C_GAPLOC": 556,
    "GGA_C_HCTH_A": 97,
    "GGA_C_HYB_TAU_HCTH": 283,
    "GGA_C_LM": 137,
    "GGA_C_LYPR": 624,
    "GGA_C_MGGAC": 712,
    "GGA_C_N12": 80,
    "GGA_C_N12_SX": 79,
    "GGA_C_OPTC": 200,
    "GGA_C_OP_B88": 87,
    "GGA_C_OP_G96": 85,
    "GGA_C_OP_PBE": 86,
    "GGA_C_OP_PW91": 262,
    "GGA_C_OP_XALPHA": 84,
    "GGA_C_P86VWN": 252,
    "GGA_C_P86VWN_FT": 253,
    "GGA_C_P86_FT": 217,
    "GGA_C_PBEFE": 258,
    "GGA_C_PBEINT": 62,
    "GGA_C_PBELOC": 246,
    "GGA_C_PBE_GAUSSIAN": 322,
    "GGA_C_PBE_JRGX": 138,
    "GGA_C_PBE_MOL": 272,
    "GGA_C_PBE_VWN": 216,
    "GGA_C_PW91": 134,
    "GGA_C_Q2D": 47,
    "GGA_C_REGTPSS": 83,
    "GGA_C_REVTCA": 99,
    "GGA_C_RGE2": 143,
    "GGA_C_SCAN_E0": 553,
    "GGA_C_SG4": 534,
    "GGA_C_SOGGA11": 152,
    "GGA_C_SOGGA11_X": 159,
    "GGA_C_SPBE": 89,
    "GGA_C_TAU_HCTH": 281,
    "GGA_C_TCA": 100,
    "GGA_C_TM_LYP": 559,
    "GGA_C_TM_PBE": 560,
    "GGA_C_W94": 561,
    "GGA_C_WI": 148,
    "GGA_C_WI0": 153,
    "GGA_C_WL": 147,
    "GGA_C_XPBE": 136,
    "GGA_C_ZPBEINT": 61,
    "GGA_C_ZPBESOL": 63,
    "GGA_C_ZVPBEINT": 557,
    "GGA_C_ZVPBELOC": 606,
    "GGA_C_ZVPBESOL": 558,
    "GGA_X_2D_B86": 128,
    "GGA_X_2D_B86_MGC": 124,
    "GGA_X_2D_B88": 127,
    "GGA_X_2D_PBE": 129,
    "GGA_X_AIRY": 192,
    "GGA_X_AK13": 56,
    "GGA_X_AM05": 120,
    "GGA_X_B86": 103,
    "GGA_X_B86_MGC": 105,
    "GGA_X_B86_R": 41,
    "GGA_X_B88M": 570,
    "GGA_X_B88_6311G": 179,
    "GGA_X_BAYESIAN": 125,
    "GGA_X_BCGP": 38,
    "GGA_X_BEEFVDW": 285,
    "GGA_X_BKL1": 338,
    "GGA_X_BKL2": 339,
    "GGA_X_BPCCAC": 98,
    "GGA_X_C09X": 158,
    "GGA_X_CAP": 270,
    "GGA_X_CHACHIYO": 298,
    "GGA_X_DK87_R1": 111,
    "GGA_X_DK87_R2": 112,
    "GGA_X_EB88": 271,
    "GGA_X_ECMV92": 215,
    "GGA_X_EV93": 35,
    "GGA_X_FD_LB94": 604,
    "GGA_X_FD_REVLB94": 605,
    "GGA_X_FT97_A": 114,
    "GGA_X_FT97_B": 115,
    "GGA_X_G96": 107,
    "GGA_X_GAM": 32,
    "GGA_X_GG99": 535,
    "GGA_X_HCTH_A": 34,
    "GGA_X_HTBS": 191,
    "GGA_X_KGG99": 544,
    "GGA_X_KT1": 145,
    "GGA_X_LAG": 193,
    "GGA_X_LAMBDA_CH_N": 44,
    "GGA_X_LAMBDA_LO_N": 45,
    "GGA_X_LAMBDA_OC2_N": 40,
    "GGA_X_LG93": 113,
    "GGA_X_LSPBE": 168,
    "GGA_X_LSRPBE": 169,
    "GGA_X_LV_RPW86": 58,
    "GGA_X_MB88": 149,
    "GGA_X_MPBE": 122,
    "GGA_X_MPW91": 119,
    "GGA_X_N12": 82,
    "GGA_X_NCAP": 180,
    "GGA_X_NCAPR": 324,
    "GGA_X_OL2": 183,
    "GGA_X_OPTB86B_VDW": 171,
    "GGA_X_OPTB88_VDW": 139,
    "GGA_X_OPTPBE_VDW": 141,
    "GGA_X_OPTX": 110,
    "GGA_X_PBEA": 121,
    "GGA_X_PBEFE": 265,
    "GGA_X_PBEINT": 60,
    "GGA_X_PBEK1_VDW": 140,
    "GGA_X_PBEPOW": 539,
    "GGA_X_PBETRANS": 291,
    "GGA_X_PBE_GAUSSIAN": 321,
    "GGA_X_PBE_JSJR": 126,
    "GGA_X_PBE_MOD": 320,
    "GGA_X_PBE_TCA": 59,
    "GGA_X_PW86": 108,
    "GGA_X_PW91": 109,
    "GGA_X_PW91_MOD": 316,
    "GGA_X_Q1D": 734,
    "GGA_X_Q2D": 48,
    "GGA_X_REVSSB_D": 312,
    "GGA_X_RGE2": 142,
    "GGA_X_RPW86": 144,
    "GGA_X_S12G": 495,
    "GGA_X_SG4": 533,
    "GGA_X_SOGGA": 150,
    "GGA_X_SOGGA11": 151,
    "GGA_X_SSB": 91,
    "GGA_X_SSB_D": 92,
    "GGA_X_SSB_SW": 90,
    "GGA_X_VMT84_GE": 68,
    "GGA_X_VMT84_PBE": 69,
    "GGA_X_VMT_GE": 70,
    "GGA_X_VMT_PBE": 71,
    "GGA_X_WC": 118,
    "GGA_X_WPBEH": 524,
    "GGA_X_XPBE": 123,
    "LDA_C_1D_CSC": 18,
    "LDA_C_1D_LOOS": 26,
    "LDA_C_2D_AMGB": 15,
    "LDA_C_2D_PRM": 16,
    "LDA_C_BR78": 552,
    "LDA_C_CHACHIYO": 287,
    "LDA_C_CHACHIYO_MOD": 307,
    "LDA_C_EPC17": 328,
    "LDA_C_EPC17_2": 329,
    "LDA_C_EPC18_1": 330,
    "LDA_C_EPC18_2": 331,
    "LDA_C_GK72": 578,
    "LDA_C_GL": 5,
    "LDA_C_GOMBAS": 24,
    "LDA_C_HL": 4,
    "LDA_C_KARASIEV": 579,
    "LDA_C_KARASIEV_MOD": 308,
    "LDA_C_LP96": 289,
    "LDA_C_MCWEENY": 551,
    "LDA_C_ML1": 22,
    "LDA_C_ML2": 23,
    "LDA_C_OB_PW": 14,
    "LDA_C_OB_PZ": 11,
    "LDA_C_OW": 574,
    "LDA_C_OW_LYP": 573,
    "LDA_C_PK09": 554,
    "LDA_C_PW_MOD": 13,
    "LDA_C_PW_RPA": 25,
    "LDA_C_PZ_MOD": 10,
    "LDA_C_RC04": 27,
    "LDA_C_RPA": 3,
    "LDA_C_RPW92": 684,
    "LDA_C_UPW92": 683,
    "LDA_C_VBH": 17,
    "LDA_C_VWN_1": 28,
    "LDA_C_VWN_2": 29,
    "LDA_C_VWN_3": 30,
    "LDA_C_VWN_4": 31,
    "LDA_C_W20": 317,
    "LDA_C_WIGNER": 2,
    "LDA_C_XALPHA": 6,
    "LDA_X_1D_EXPONENTIAL": 600,
    "LDA_X_1D_SOFT": 21,
    "LDA_X_2D": 19,
    "LDA_X_RAE": 549,
    "LDA_X_REL": 532,
    "LDA_X_SLOC": 692,
    "MGGA_C_B88": 571,
    "MGGA_C_BC95": 240,
    "MGGA_C_CC": 387,
    "MGGA_C_CCALDA": 388,
    "MGGA_C_CF22D": 341,
    "MGGA_C_LAK": 345,
    "MGGA_C_DLDF": 37,
    "MGGA_C_HLTAPW": 699,
    "MGGA_C_KCIS": 562,
    "MGGA_C_KCISK": 638,
    "MGGA_C_M05": 237,
    "MGGA_C_M05_2X": 238,
    "MGGA_C_M06_HF": 234,
    "MGGA_C_M06_SX": 311,
    "MGGA_C_M08_HX": 78,
    "MGGA_C_M08_SO": 77,
    "MGGA_C_M11": 76,
    "MGGA_C_M11_L": 75,
    "MGGA_C_MN12_SX": 73,
    "MGGA_C_MN15": 269,
    "MGGA_C_MN15_L": 261,
    "MGGA_C_PKZB": 239,
    "MGGA_C_R2SCAN01": 642,
    "MGGA_C_REVM06": 306,
    "MGGA_C_REVM06_L": 294,
    "MGGA_C_REVM11": 172,
    "MGGA_C_REVSCAN": 582,
    "MGGA_C_REVSCAN_VV10": 585,
    "MGGA_C_REVTM": 694,
    "MGGA_C_REVTPSS": 241,
    "MGGA_C_RMGGAC": 643,
    "MGGA_C_RPPSCAN": 649,
    "MGGA_C_RREGTM": 391,
    "MGGA_C_RSCAN": 494,
    "MGGA_C_SCAN_RVV10": 292,
    "MGGA_C_SCAN_VV10": 584,
    "MGGA_C_TM": 251,
    "MGGA_C_TPSSLOC": 247,
    "MGGA_C_TPSS_GAUSSIAN": 323,
    "MGGA_C_VSXC": 232,
    "MGGA_X_2D_JS17": 609,
    "MGGA_X_BLOC": 244,
    "MGGA_X_EEL": 326,
    "MGGA_X_GVT4": 204,
    "MGGA_X_GX": 575,
    "MGGA_X_HLTA": 698,
    "MGGA_X_KTBM_0": 735,
    "MGGA_X_KTBM_1": 736,
    "MGGA_X_KTBM_10": 745,
    "MGGA_X_KTBM_11": 746,
    "MGGA_X_KTBM_12": 747,
    "MGGA_X_KTBM_13": 748,
    "MGGA_X_KTBM_14": 749,
    "MGGA_X_KTBM_15": 750,
    "MGGA_X_KTBM_16": 751,
    "MGGA_X_KTBM_17": 752,
    "MGGA_X_KTBM_18": 753,
    "MGGA_X_KTBM_19": 754,
    "MGGA_X_KTBM_2": 737,
    "MGGA_X_KTBM_20": 755,
    "MGGA_X_KTBM_21": 756,
    "MGGA_X_KTBM_22": 757,
    "MGGA_X_KTBM_23": 758,
    "MGGA_X_KTBM_24": 759,
    "MGGA_X_KTBM_3": 738,
    "MGGA_X_KTBM_4": 739,
    "MGGA_X_KTBM_5": 740,
    "MGGA_X_KTBM_6": 741,
    "MGGA_X_KTBM_7": 742,
    "MGGA_X_KTBM_8": 743,
    "MGGA_X_KTBM_9": 744,
    "MGGA_X_KTBM_GAP": 760,
    "MGGA_X_LAK": 342,
    "MGGA_X_LTA": 201,
    "MGGA_X_MBEEF": 249,
    "MGGA_X_MBEEFVDW": 250,
    "MGGA_X_MBR": 716,
    "MGGA_X_MBRXC_BG": 696,
    "MGGA_X_MBRXH_BG": 697,
    "MGGA_X_MCML": 644,
    "MGGA_X_MGGAC": 711,
    "MGGA_X_MN15_L": 260,
    "MGGA_X_MODTPSS": 245,
    "MGGA_X_MS0": 221,
    "MGGA_X_MS1": 222,
    "MGGA_X_MS2": 223,
    "MGGA_X_MS2B": 300,
    "MGGA_X_MS2BS": 301,
    "MGGA_X_MS2_REV": 228,
    "MGGA_X_MSB86BL": 765,
    "MGGA_X_MSPBEL": 761,
    "MGGA_X_MSRPBEL": 763,
    "MGGA_X_MTASK": 724,
    "MGGA_X_MVS": 257,
    "MGGA_X_MVSB": 302,
    "MGGA_X_MVSBS": 303,
    "MGGA_X_PBE_GX": 576,
    "MGGA_X_PKZB": 213,
    "MGGA_X_R2SCAN01": 645,
    "MGGA_X_REGTM": 626,
    "MGGA_X_REGTPSS": 603,
    "MGGA_X_REVM06_L": 293,
    "MGGA_X_REVSCAN": 581,
    "MGGA_X_REVTM": 693,
    "MGGA_X_RMSB86BL": 766,
    "MGGA_X_RMSPBEL": 762,
    "MGGA_X_RMSRPBEL": 764,
    "MGGA_X_RPPSCAN": 648,
    "MGGA_X_RTPSS": 299,
    "MGGA_X_SA_TPSS": 542,
    "MGGA_X_TASK": 707,
    "MGGA_X_TAU_HCTH": 205,
    "MGGA_X_TH": 225,
    "MGGA_X_TLDA": 685,
    "MGGA_X_TM": 540,
    "MGGA_X_VCML": 651,
    "MGGA_X_VT84": 541,
}
NAME_TO_ID.update(_OBSERVED_PURE_COMPONENT_IDS)

LEGACY_PURE_COMPONENT_NAMES: tuple[str, ...] = (
    "LDA_X",
    "LDA_C_VWN",
    "LDA_C_VWN_RPA",
    "GGA_X_PBE",
    "GGA_C_PBE",
    "GGA_X_B88",
    "GGA_C_LYP",
    "GGA_X_PBE_SOL",
    "MGGA_X_SCAN",
    "MGGA_C_SCAN",
    "MGGA_X_R2SCAN",
    "LDA_C_PZ",
    "LDA_C_PW",
    "GGA_X_PBE_MOL",
    "MGGA_C_MN12_L",
    "GGA_X_PBE_R",
    "GGA_X_RPBE",
    "GGA_C_P86",
    "GGA_C_PBE_SOL",
    "GGA_X_APBE",
    "GGA_C_APBE",
    "MGGA_X_TPSS",
    "MGGA_X_M06_L",
    "MGGA_X_REVTPSS",
    "MGGA_X_MN12_L",
    "MGGA_C_TPSS",
    "MGGA_C_M06_L",
    "MGGA_C_M06",
    "MGGA_C_M06_2X",
    "MGGA_X_RSCAN",
    "MGGA_C_R2SCAN",
    "MGGA_X_R4SCAN",
)

LEGACY_PURE_COMPONENT_ROLES: dict[str, str] = {
    name: ("exchange" if "_X_" in name or name == "LDA_X" else "correlation")
    for name in LEGACY_PURE_COMPONENT_NAMES
}

# These names are supported for constructing fixed evaluation baselines, but
# are deliberately not exposed as searchable component arms.
EVALUATION_ONLY_COMPONENT_NAMES = frozenset({"MGGA_C_LAK"})

PURE_COMPONENT_ROLES: dict[str, str] = {
    name: ("exchange" if "_X_" in name or name == "LDA_X" else "correlation")
    for name in NAME_TO_ID
    if (
        name != "GGA_X_HJS_PBE"
        and name not in EVALUATION_ONLY_COMPONENT_NAMES
        and "_XC_" not in name
        and ("_X_" in name or "_C_" in name or name == "LDA_X")
    )
}

COMPONENT_XCTYPES: dict[str, str] = {
    name: (
        "LDA"
        if name.startswith("LDA_")
        else "MGGA"
        if name.startswith("MGGA_") or name.startswith("HYB_MGGA_")
        else "GGA"
    )
    for name in NAME_TO_ID
}

_NATIVE_REFERENCE_FUNCTIONALS: dict[str, tuple[str, str, str]] = {
    "TPSSH": ("TPSSh", "TPSSh", "MGGA"),
    "M06": ("M06", "M06", "MGGA"),
    "M062X": ("M06-2X", "M06-2X", "MGGA"),
    "M06-2X": ("M06-2X", "M06-2X", "MGGA"),
    "R2SCAN": ("R2SCAN", "R2SCAN", "MGGA"),
    "R2SCAN0": ("r2SCAN0", "R2SCAN0", "MGGA"),
    "WB97X-V": ("wB97X-V", "WB97X-V", "MGGA"),
    "WB97XV": ("wB97X-V", "WB97X-V", "MGGA"),
    "WB97M-V": ("wB97M-V", "WB97M-V", "MGGA"),
    "WB97MV": ("wB97M-V", "WB97M-V", "MGGA"),
    "B97M-V": ("B97M-V", "B97M-V", "MGGA"),
    "B97MV": ("B97M-V", "B97M-V", "MGGA"),
}


def _short_alias(canonical: str) -> str:
    return PYSOF_ALIAS.get(canonical, canonical)


def _xctype_rank(xctype: str) -> int:
    return {"LDA": 0, "GGA": 1, "MGGA": 2}.get(xctype, 0)


def _round_float(value: Any, ndigits: int = 8) -> float:
    return round(float(value), ndigits)


def _strip_lr_hf_annotation(xc_code: str) -> str:
    """Mirror the benchmark registration path where LR_HF is carried by hyb."""
    if "," not in xc_code:
        return xc_code
    exchange, correlation = xc_code.split(",", 1)
    kept = [
        part
        for part in exchange.split("+")
        if "LR_HF" not in part
    ]
    return "+".join(part for part in kept if part) + "," + correlation


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ComponentSpec:
    name: str
    formula: str | None = None
    eval_xc_code: str | None = None
    xctype: str = "GGA"
    mixing_factor: float = 1.0
    role: str = "exchange"
    ext_params: dict[str, float] = field(default_factory=dict)
    ext_param_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    descriptor_spec: dict[str, Any] = field(default_factory=dict)

    @property
    def libxc_id(self) -> int | None:
        return NAME_TO_ID.get(self.name)

    @property
    def is_custom(self) -> bool:
        return self.eval_xc_code is not None or bool(self.descriptor_spec)


@dataclass
class HFExchange:
    alpha: float = 0.0
    beta: float = 0.0
    omega: float = 0.0


@dataclass
class XCFunctional:
    name: str
    components: list[ComponentSpec] = field(default_factory=list)
    hf_exchange: HFExchange = field(default_factory=HFExchange)
    role_scales: dict[str, float] = field(
        default_factory=lambda: {"exchange": 1.0, "correlation": 1.0}
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_custom(self) -> bool:
        return any(c.is_custom for c in self.components)

    @property
    def highest_rung(self) -> str:
        r = max((_xctype_rank(c.xctype) for c in self.components), default=0)
        return ["LDA", "GGA", "MGGA"][r]

    # ------------------------------------------------------------------
    # xc_code generation (for REWIRE path)
    # ------------------------------------------------------------------

    def to_xc_code(self) -> str:
        # Separate exchange/correlation components
        exch_parts: list[str] = []
        corr_parts: list[str] = []
        compound_parts: list[str] = []
        for c in self.components:
            if c.role == "descriptor_correction" or c.descriptor_spec:
                continue
            alias = _short_alias(c.name)
            role_scale = (
                float(self.role_scales.get(c.role, 1.0))
                if c.role in {"exchange", "correlation"}
                else 1.0
            )
            effective_mixing = float(c.mixing_factor) * role_scale
            factor = f"{effective_mixing:.6g}*" if effective_mixing != 1.0 else ""
            if c.role == "exchange_correlation":
                compound_parts.append(f"{factor}{alias}")
            elif c.role == "exchange":
                exch_parts.append(f"{factor}{alias}")
            else:
                corr_parts.append(f"{factor}{alias}")

        if compound_parts and not exch_parts and not corr_parts:
            # Pure compound functional
            return ",".join(compound_parts)

        # HF exchange: always render alpha (short-range), and also render
        # beta as LR_HF when range-separation is active (omega > 0).
        # This faithfully transmits the full hf_exchange state rather than
        # silently discarding beta/omega.
        if self.hf_exchange.alpha > 0:
            exch_parts.append(f"{self.hf_exchange.alpha:.6g}*HF")
        if self.hf_exchange.beta > 0 and self.hf_exchange.omega > 0:
            exch_parts.append(f"{self.hf_exchange.beta:.6g}*LR_HF")

        # PySCF accepts empty exchange/correlation sides (",PBE" or "PBE,").
        # Do not render a missing side as literal "0": libxc parses that as
        # functional id 0 and emits "XC functional 0 not found" during runs.
        exch = "+".join(exch_parts)
        corr = "+".join(corr_parts)
        return f"{exch},{corr}"

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> list[str]:
        """Check physical constraints. Returns list of violation strings.

        Violations prefixed with 'HARD:' are unphysical and should block evaluation.
        Violations prefixed with 'SOFT:' are physical limitations but not errors.
        """
        violations: list[str] = []
        tol = 1e-6

        # Sum rules (HARD)
        exchange_scale = float(self.role_scales.get("exchange", 1.0))
        correlation_scale = float(self.role_scales.get("correlation", 1.0))
        for role, scale in (("exchange", exchange_scale), ("correlation", correlation_scale)):
            if scale < 0.8 - tol or scale > 1.2 + tol:
                violations.append(f"HARD: {role} role scale {scale:.4f} outside [0.8, 1.2]")
        exch_sum = sum(c.mixing_factor for c in self.components if c.role == "exchange")
        corr_sum = sum(c.mixing_factor for c in self.components if c.role == "correlation")
        effective_exch_sum = exchange_scale * exch_sum
        effective_corr_sum = correlation_scale * corr_sum
        exchange_limit = 1.0 if abs(exchange_scale - 1.0) <= tol else 1.2
        correlation_limit = 1.0 if abs(correlation_scale - 1.0) <= tol else 1.2
        if effective_exch_sum + self.hf_exchange.alpha > exchange_limit + tol:
            violations.append(
                f"HARD: Effective exchange sum {effective_exch_sum:.4f} + alpha "
                f"{self.hf_exchange.alpha:.4f} > {exchange_limit:.1f}"
            )
        if effective_corr_sum > correlation_limit + tol:
            violations.append(
                f"HARD: Effective correlation sum {effective_corr_sum:.4f} > {correlation_limit:.1f}"
            )

        # HF exchange bounds (HARD)
        if self.hf_exchange.alpha < -tol:
            violations.append(f"HARD: alpha = {self.hf_exchange.alpha} < 0")
        if self.hf_exchange.beta < -tol:
            violations.append(f"HARD: beta = {self.hf_exchange.beta} < 0")
        if self.hf_exchange.omega < -tol:
            violations.append(f"HARD: omega = {self.hf_exchange.omega} < 0")
        if self.hf_exchange.alpha + self.hf_exchange.beta > 1.0 + tol:
            violations.append(f"HARD: alpha+beta = {self.hf_exchange.alpha + self.hf_exchange.beta} > 1.0")

        # Parameter bounds (HARD)
        for c in self.components:
            if c.descriptor_spec:
                from .descriptors import validate_descriptor_spec

                violations.extend(validate_descriptor_spec(c.descriptor_spec))
            for pname, pval in c.ext_params.items():
                if pname in c.ext_param_bounds:
                    lo, hi = c.ext_param_bounds[pname]
                    if pval < lo - tol or pval > hi + tol:
                        violations.append(f"HARD: {c.name}.{pname} = {pval} outside [{lo}, {hi}]")

        # Lieb-Oxford bound (HARD)
        for c in self.components:
            if c.role == "exchange" and "_kappa" in c.ext_params:
                kappa = c.ext_params["_kappa"]
                if kappa > 1.679 + tol:
                    violations.append(
                        f"HARD: {c.name}._kappa={kappa:.4f} > 1.679 (Lieb-Oxford bound violation)"
                    )

        # One-electron SIE (SOFT — only flag if alpha is unusually low)
        # Standard hybrids (alpha=0.20-0.50) have residual SIE by design;
        # this is expected and not actionable. Only flag alpha < 0.10 as
        # a significant SIE concern.
        if self.hf_exchange.alpha < 0.10 - tol and self.hf_exchange.alpha > 0:
            violations.append(
                f"SOFT: alpha={self.hf_exchange.alpha:.3f} < 0.10 (very low HF exchange, significant SIE)"
            )

        descriptor_count = sum(1 for c in self.components if c.descriptor_spec or c.role == "descriptor_correction")
        if descriptor_count > 1:
            violations.append(f"HARD: descriptor component count {descriptor_count} > 1")

        return violations

    # ------------------------------------------------------------------
    # Parameter vector (for optimization)
    # ------------------------------------------------------------------

    def param_vector(self) -> list[float]:
        """Flatten all tunable params into a vector."""
        vec: list[float] = []
        for c in sorted(self.components, key=lambda x: x.name):
            for pname in sorted(c.ext_params):
                vec.append(c.ext_params[pname])
        if self.hf_exchange.alpha > 0 or self.hf_exchange.beta > 0 or self.hf_exchange.omega > 0:
            vec.extend([self.hf_exchange.alpha, self.hf_exchange.beta, self.hf_exchange.omega])
        return vec

    def with_param_vector(self, vec: list[float]) -> XCFunctional:
        """Return a copy with params set from vector."""
        new = copy.deepcopy(self)
        idx = 0
        for c in sorted(new.components, key=lambda x: x.name):
            for pname in sorted(c.ext_params):
                c.ext_params[pname] = vec[idx]
                idx += 1
        if new.hf_exchange.alpha > 0 or new.hf_exchange.beta > 0 or new.hf_exchange.omega > 0:
            new.hf_exchange.alpha = vec[idx]
            new.hf_exchange.beta = vec[idx + 1]
            new.hf_exchange.omega = vec[idx + 2]
        return new

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        def _component_dict(c: ComponentSpec) -> dict:
            return {
                "name": c.name,
                "formula": c.formula,
                "eval_xc_code": c.eval_xc_code,
                "xctype": c.xctype,
                "mixing_factor": c.mixing_factor,
                "role": c.role,
                "ext_params": c.ext_params,
                "ext_param_bounds": {k: list(v) for k, v in c.ext_param_bounds.items()},
                "descriptor_spec": c.descriptor_spec,
            }
        return {
            "name": self.name,
            "components": [_component_dict(c) for c in self.components],
            "hf_exchange": {
                "alpha": self.hf_exchange.alpha,
                "beta": self.hf_exchange.beta,
                "omega": self.hf_exchange.omega,
            },
            "role_scales": {
                "exchange": float(self.role_scales.get("exchange", 1.0)),
                "correlation": float(self.role_scales.get("correlation", 1.0)),
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: dict) -> XCFunctional:
        comps = []
        for cd in d["components"]:
            bounds = {k: tuple(v) for k, v in cd.get("ext_param_bounds", {}).items()}
            comps.append(ComponentSpec(
                name=cd["name"],
                formula=cd.get("formula"),
                eval_xc_code=cd.get("eval_xc_code"),
                xctype=cd.get("xctype", "GGA"),
                mixing_factor=cd.get("mixing_factor", 1.0),
                role=cd.get("role", "exchange"),
                ext_params=cd.get("ext_params", {}),
                ext_param_bounds=bounds,
                descriptor_spec=cd.get("descriptor_spec", {}) or {},
            ))
        hf = d.get("hf_exchange", {})
        role_scales = d.get("role_scales", {}) or {}
        return cls(
            name=d["name"],
            components=comps,
            hf_exchange=HFExchange(
                alpha=hf.get("alpha", 0.0),
                beta=hf.get("beta", 0.0),
                omega=hf.get("omega", 0.0),
            ),
            role_scales={
                "exchange": float(role_scales.get("exchange", 1.0)),
                "correlation": float(role_scales.get("correlation", 1.0)),
            },
            metadata=d.get("metadata", {}),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, s: str) -> XCFunctional:
        return cls.from_dict(json.loads(s))

    def signature(self) -> str:
        """Deterministic hash for deduplication."""
        payload = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def effective_signature(self) -> str:
        """Hash the executable XC expression seen by the benchmark backend.

        This is stricter than a name/history check and looser than the raw
        component schema: two candidates that serialize differently but compile
        to the same LibXC expression, HF tuple, and ext_params share this key.
        """
        payload = json.dumps(self.effective_spec(), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def effective_spec(self) -> dict[str, Any]:
        """Canonical benchmark-facing representation for semantic dedup."""
        native_xc_code = str(self.metadata.get("native_xc_code", "") or "").strip()
        if native_xc_code:
            return {"mode": "native", "xc_code": native_xc_code}

        # A zero-weight LibXC/custom term has no effect on the executable
        # functional.  Excluding it here prevents structure mutations on a
        # dormant component from consuming an evaluation as a "new" point.
        executable = copy.deepcopy(self)
        executable.components = [
            component
            for component in executable.components
            if abs(float(component.mixing_factor)) > 1e-12
        ]
        if executable.is_custom:
            regular_components = [
                c for c in executable.components
                if not c.eval_xc_code and not c.descriptor_spec
            ]
            base = copy.deepcopy(executable)
            base.components = regular_components
            custom_parts = [
                {
                    "name": c.name,
                    "role": c.role,
                    "xctype": c.xctype,
                    "mixing_factor": _round_float(c.mixing_factor),
                    "eval_xc_code_hash": hashlib.sha256((c.eval_xc_code or "").encode()).hexdigest()[:16],
                    "descriptor_spec": c.descriptor_spec,
                }
                for c in sorted(executable.components, key=lambda item: (item.role, item.name))
                if c.eval_xc_code or c.descriptor_spec
            ]
            return {
                "mode": "custom",
                "base_xc_code": _strip_lr_hf_annotation(base.to_xc_code()),
                "base_ext_params": base._effective_ext_params(),
                "components": custom_parts,
                "hyb": executable._effective_hyb_tuple(),
            }
        return {
            "mode": "libxc",
            "xc_code": _strip_lr_hf_annotation(executable.to_xc_code()),
            "hyb": executable._effective_hyb_tuple(),
            "ext_params": executable._effective_ext_params(),
        }

    def _effective_hyb_tuple(self) -> tuple[float, float, float]:
        alpha = _round_float(self.hf_exchange.alpha)
        beta = _round_float(self.hf_exchange.beta)
        omega = _round_float(self.hf_exchange.omega)
        # For ordinary full-range DFA exchange, omega has no physical effect
        # when beta is zero.  Canonicalize that state so an omega-only BO move
        # cannot evade effective-signature deduplication.  HJS-PBE is itself
        # omega dependent and therefore retains omega even at beta == 0.
        has_omega_dependent_component = any(
            component.name == "GGA_X_HJS_PBE" or component.role == "exchange_correlation"
            for component in self.components
        )
        if abs(beta) <= 1e-12 and not has_omega_dependent_component:
            omega = 0.0
        return (alpha, _round_float(alpha + beta), omega)

    def _effective_ext_params(self) -> dict[str, list[float]]:
        ext_params: dict[str, list[float]] = {}
        for comp in self.components:
            xid = NAME_TO_ID.get(comp.name)
            if xid is None:
                continue
            spec = LIBXC_PARAMS.get(xid, [])
            if not spec:
                continue
            values = []
            for pname, default, _bounds in spec:
                values.append(_round_float(comp.ext_params.get(pname, default)))
            ext_params[str(xid)] = values
        return dict(sorted(ext_params.items()))

    def structural_signature(self) -> str:
        """Hash of physical structure only (no metadata).

        Two functionals that differ only in metadata (parent, action, skill)
        but have the same components and HF exchange get the same signature.
        """
        payload = json.dumps({
            "components": [
                {
                    "name": c.name, "xctype": c.xctype,
                    "mixing_factor": c.mixing_factor, "role": c.role,
                    "ext_params": c.ext_params,
                    "descriptor_spec": c.descriptor_spec,
                }
                for c in sorted(self.components, key=lambda x: x.name)
            ],
            "hf_exchange": {
                "alpha": self.hf_exchange.alpha,
                "beta": self.hf_exchange.beta,
                "omega": self.hf_exchange.omega,
            },
            "role_scales": {
                "exchange": float(self.role_scales.get("exchange", 1.0)),
                "correlation": float(self.role_scales.get("correlation", 1.0)),
            },
        }, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    # ------------------------------------------------------------------
    # Factory: well-known functionals
    # ------------------------------------------------------------------

    @classmethod
    def from_known(cls, name: str) -> XCFunctional:
        """Build an XCFunctional for a well-known functional."""
        builders = {
            "PBE": cls._build_pbe,
            "PBE0": cls._build_pbe0,
            "B3LYP": cls._build_b3lyp,
            "CAMB3LYP": cls._build_cam_b3lyp,
            "WB97X": cls._build_wb97x,
            "SCAN": cls._build_scan,
            "SCAN0": cls._build_scan0,
            "HYBGGAXCLCWPBE": cls._build_lc_wpbe,
            "HYBGGAXCLCWPBEEDITABLE": cls._build_lc_wpbe,
            "LCWPBE": cls._build_lc_wpbe,
            "HLE16SPLIT": cls._build_hle16_split,
        }
        key = cls._normalize_known_name(name)
        if key not in builders:
            native = _NATIVE_REFERENCE_FUNCTIONALS.get(key)
            if native is not None:
                display_name, xc_code, xctype = native
                return cls._build_native(display_name, xc_code=xc_code, xctype=xctype)
            raise ValueError(f"Unknown functional: {name}. Choose from: {list(builders.keys())}")
        return builders[key]()

    @staticmethod
    def _normalize_known_name(name: str) -> str:
        return (
            name.upper()
            .replace("Ω", "W")
            .replace("ω", "W")
            .replace("-", "")
            .replace(" ", "")
            .replace("_", "")
        )

    # --- builders ---

    @staticmethod
    def _pbe_x(mix: float = 1.0) -> ComponentSpec:
        return ComponentSpec(
            name="GGA_X_PBE", xctype="GGA", mixing_factor=mix, role="exchange",
            ext_params={"_kappa": 0.8040, "_mu": 0.21951497268},
            ext_param_bounds={"_kappa": (0.6, 1.5), "_mu": (0.10, 0.30)},
        )

    @staticmethod
    def _pbe_c(mix: float = 1.0) -> ComponentSpec:
        return ComponentSpec(
            name="GGA_C_PBE", xctype="GGA", mixing_factor=mix, role="correlation",
            ext_params={"_beta": 0.06672455060314922, "_gamma": 0.031090690869654895, "_d": 1.0},
            ext_param_bounds={"_beta": (0.03, 0.10), "_gamma": (0.01, 0.05), "_d": (0.5, 1.5)},
        )

    @classmethod
    def _build_pbe(cls) -> XCFunctional:
        return cls(name="PBE", components=[cls._pbe_x(), cls._pbe_c()])

    @classmethod
    def _build_pbe0(cls) -> XCFunctional:
        return cls(
            name="PBE0",
            components=[cls._pbe_x(0.75), cls._pbe_c()],
            hf_exchange=HFExchange(alpha=0.25),
        )

    @classmethod
    def _build_b3lyp(cls) -> XCFunctional:
        return cls(
            name="B3LYP",
            components=[
                ComponentSpec(name="LDA_X", xctype="LDA", mixing_factor=0.08, role="exchange"),
                ComponentSpec(
                    name="GGA_X_B88", xctype="GGA", mixing_factor=0.72, role="exchange",
                    ext_params={"_beta": 0.0042, "_gamma": 6.0},
                    ext_param_bounds={"_beta": (0.002, 0.010), "_gamma": (4.0, 8.0)},
                ),
                ComponentSpec(
                    name="GGA_C_LYP", xctype="GGA", mixing_factor=0.81, role="correlation",
                    ext_params={"_a": 0.04918, "_b": 0.132, "_c": 0.2533, "_d": 0.349},
                    ext_param_bounds={"_a": (0.03, 0.07), "_b": (0.08, 0.20),
                                      "_c": (0.15, 0.35), "_d": (0.25, 0.50)},
                ),
                ComponentSpec(name="LDA_C_VWN_RPA", xctype="LDA", mixing_factor=0.19, role="correlation"),
            ],
            hf_exchange=HFExchange(alpha=0.20),
        )

    @classmethod
    def _build_cam_b3lyp(cls) -> XCFunctional:
        return cls(
            name="CAM-B3LYP",
            components=[
                ComponentSpec(
                    name="HYB_GGA_XC_CAM_B3LYP", xctype="GGA", mixing_factor=1.0,
                    role="exchange_correlation",
                    ext_params={
                        "_ss2": 0.81, "_alpha": 0.65, "_beta": -0.46, "_omega": 0.33,
                    },
                    ext_param_bounds={
                        "_ss2": (0.60, 0.95), "_alpha": (0.40, 0.80),
                        "_beta": (-0.80, -0.10), "_omega": (0.20, 0.50),
                    },
                ),
            ],
            hf_exchange=HFExchange(alpha=0.19, beta=0.46, omega=0.33),
        )

    @classmethod
    def _build_wb97x(cls) -> XCFunctional:
        return cls(
            name="wB97X",
            components=[
                ComponentSpec(
                    name="HYB_MGGA_XC_WB97X", xctype="MGGA", mixing_factor=1.0,
                    role="exchange_correlation",
                    ext_params={
                        "_c0_0": 0.842294, "_c0_1": 0.726479, "_c0_2": 1.0476,
                        "_c0_3": -5.70635, "_c0_4": 13.2794, "_c0_5": 1.0,
                        "_c0_6": -4.33879, "_c0_7": 18.2308, "_c0_8": -31.743,
                        "_c0_9": 17.2901, "_c0_10": 1.0, "_c0_11": 2.37031,
                        "_c0_12": -11.3995, "_c0_13": 6.58405, "_c0_14": -3.78132,
                        "_c0_15": 1.0, "_c0_16": -0.842294, "_c0_17": 0.3,
                    },
                    ext_param_bounds={
                        "_c0_0": (0.5, 1.0), "_c0_1": (0.4, 1.0), "_c0_2": (0.5, 2.0),
                        "_c0_3": (-10.0, 0.0), "_c0_4": (5.0, 20.0), "_c0_5": (0.5, 1.5),
                        "_c0_6": (-8.0, 0.0), "_c0_7": (10.0, 30.0), "_c0_8": (-50.0, -10.0),
                        "_c0_9": (10.0, 30.0), "_c0_10": (0.5, 1.5), "_c0_11": (1.0, 5.0),
                        "_c0_12": (-20.0, -5.0), "_c0_13": (3.0, 10.0), "_c0_14": (-6.0, 0.0),
                        "_c0_15": (0.5, 1.5), "_c0_16": (-1.0, 0.0), "_c0_17": (0.20, 0.50),
                    },
                ),
            ],
            hf_exchange=HFExchange(alpha=0.157706, beta=0.842294, omega=0.30),
        )

    @classmethod
    def _build_scan(cls) -> XCFunctional:
        return cls(
            name="SCAN",
            components=[
                ComponentSpec(
                    name="MGGA_X_SCAN", xctype="MGGA", mixing_factor=1.0, role="exchange",
                    ext_params={"_c1": 0.667, "_c2": 0.800, "_d": 1.240, "_k1": 0.065},
                    ext_param_bounds={"_c1": (0.5, 0.8), "_c2": (0.6, 1.0),
                                      "_d": (1.0, 1.5), "_k1": (0.04, 0.10)},
                ),
                ComponentSpec(name="MGGA_C_SCAN", xctype="MGGA", mixing_factor=1.0, role="correlation"),
            ],
        )

    @classmethod
    def _build_scan0(cls) -> XCFunctional:
        base = cls._build_scan()
        base.name = "SCAN0"
        base.hf_exchange = HFExchange(alpha=0.25)
        base.components[0].mixing_factor = 0.75
        return base

    @classmethod
    def _build_lc_wpbe(cls) -> XCFunctional:
        """Editable LC-wPBE scaffold equivalent to HYB_GGA_XC_LC_WPBE."""
        return cls(
            name="HYB_GGA_XC_LC_WPBE__editable",
            components=[
                ComponentSpec(
                    name="GGA_X_HJS_PBE",
                    xctype="GGA",
                    mixing_factor=1.0,
                    role="exchange",
                ),
                cls._pbe_c(),
            ],
            hf_exchange=HFExchange(alpha=0.0, beta=1.0, omega=0.4),
            metadata={
                "decomposed_from": "HYB_GGA_XC_LC_WPBE",
                "equivalence_check_recommended": True,
            },
        )

    @classmethod
    def _build_hle16_split(cls) -> XCFunctional:
        """Editable X/C decomposition that is numerically equivalent to HLE16."""
        components = []
        for name, role in (
            ("GGA_X_HCTH_407_HLE16", "exchange"),
            ("GGA_C_HCTH_407_HLE16", "correlation"),
        ):
            param_spec = LIBXC_PARAMS[NAME_TO_ID[name]]
            components.append(
                ComponentSpec(
                    name=name,
                    xctype="GGA",
                    mixing_factor=1.0,
                    role=role,
                    ext_params={key: default for key, default, _bounds in param_spec},
                    # HLE16 coefficients define the virtual component and are
                    # intentionally fixed; mixing/role scales remain editable.
                    ext_param_bounds={key: bounds for key, _default, bounds in param_spec},
                )
            )
        return cls(
            name="GGA_XC_HLE16__split_editable",
            components=components,
            metadata={
                "decomposed_from": "GGA_XC_HLE16",
                "decomposition": "1.25*HCTH/407 exchange + 0.50*HCTH/407 correlation",
                "equivalence_check_recommended": True,
            },
        )

    @classmethod
    def _build_native(cls, display_name: str, *, xc_code: str, xctype: str = "MGGA") -> XCFunctional:
        return cls(
            name=display_name,
            components=[
                ComponentSpec(
                    name=xc_code,
                    xctype=xctype,
                    mixing_factor=1.0,
                    role="exchange_correlation",
                )
            ],
            metadata={"native_xc_code": xc_code},
        )

    # ------------------------------------------------------------------
    # Dynamic libxc enumeration
    # ------------------------------------------------------------------

    @classmethod
    def enumerate_libxc(cls) -> dict[str, dict]:
        """Enumerate all available libxc functionals with metadata.

        Returns dict of {name: {libxc_id, xc_type, hybrid_coeff, rsh_coeff,
        nlc_coeff, is_hybrid, is_rsh, is_nlc, role}}.
        """
        from pyscf.dft import libxc
        func_dict = libxc.available_libxc_functionals()
        result = {}
        for name, fid in func_dict.items():
            # Skip kinetic energy functionals
            if "_K_" in name or name.startswith("LDA_K") or name.startswith("GGA_K") or name.startswith("MGGA_K"):
                continue

            # Determine role from name
            role = "unknown"
            if "_X_" in name or name == "LDA_X":
                role = "exchange"
            elif "_C_" in name:
                role = "correlation"
            elif "_XC_" in name:
                role = "exchange_correlation"

            try:
                xc_type = libxc.xc_type(fid)
                is_hybrid = libxc.is_hybrid_xc(fid)
                hybrid_coeff = libxc.hybrid_coeff(fid) if is_hybrid else 0.0
                rsh_coeff = libxc.rsh_coeff(fid) if is_hybrid else (0.0, 0.0, 0.0)
                is_nlc = libxc.is_nlc(fid)
                nlc_coeff = libxc.nlc_coeff(fid) if is_nlc else ()
            except Exception:
                xc_type = "GGA"
                is_hybrid = False
                hybrid_coeff = 0.0
                rsh_coeff = (0.0, 0.0, 0.0)
                is_nlc = False
                nlc_coeff = ()

            is_rsh = rsh_coeff[0] > 0 if isinstance(rsh_coeff, tuple) and len(rsh_coeff) >= 1 else False

            result[name] = {
                "libxc_id": int(fid),
                "xc_type": xc_type,
                "hybrid_coeff": float(hybrid_coeff),
                "rsh_coeff": list(rsh_coeff) if isinstance(rsh_coeff, tuple) else [0.0, 0.0, 0.0],
                "nlc_coeff": list(nlc_coeff) if isinstance(nlc_coeff, tuple) else [],
                "is_hybrid": bool(is_hybrid),
                "is_rsh": bool(is_rsh),
                "is_nlc": bool(is_nlc),
                "role": role,
            }
        return result

    @classmethod
    def params_from_libxc(cls, name: str) -> dict:
        """Extract functional parameters dynamically from libxc.

        Returns dict with: hybrid_coeff, rsh_coeff, nlc_coeff, xc_type, is_hybrid, etc.
        """
        from pyscf.dft import libxc
        fid = libxc.XC_CODES.get(name)
        if fid is None:
            # Try uppercase
            fid = libxc.XC_CODES.get(name.upper())
        if fid is None:
            return {}

        info = {}
        try:
            info["xc_type"] = libxc.xc_type(fid)
            info["is_hybrid"] = libxc.is_hybrid_xc(fid)
            info["hybrid_coeff"] = float(libxc.hybrid_coeff(fid))
            info["rsh_coeff"] = list(libxc.rsh_coeff(fid))
            info["is_nlc"] = libxc.is_nlc(fid)
            info["nlc_coeff"] = list(libxc.nlc_coeff(fid))
        except Exception:
            pass
        info["libxc_id"] = int(fid)
        return info
