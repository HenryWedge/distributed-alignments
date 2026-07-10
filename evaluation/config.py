import os

EVALUATION_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(EVALUATION_DIR)
DATASETS_DIR = os.path.join(CODE_DIR, "datasets")
RESULTS_DIR = os.path.join(EVALUATION_DIR, "results")

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DATASETS = {
    #"BPIC2017": {
    #    "path": os.path.join(DATASETS_DIR, "BPI Challenge 2017.xes"),
    #    "location_key": "org:resource",
    #},
    "Sepsis": {
        "path": os.path.join(DATASETS_DIR, "Sepsis.xes"),
        "location_key": "org:group",
    },
    "TrafficFine": {
        "path": os.path.join(DATASETS_DIR, "Road_Traffic_Fine_Management_Process.xes"),
        "location_key": "concept:name",
    },
    "PermitLog": {
        "path": os.path.join(DATASETS_DIR, "PermitLog.xes"),
        "location_key": "org:resource",
    },
    "SmartFactory": {
        "path": os.path.join(DATASETS_DIR, "MainProcess.xes"),
        "location_key": "org:resource",
    }
}

# ---------------------------------------------------------------------------
# Default experiment parameters
# ---------------------------------------------------------------------------

DEFAULT_N_TRAINING = 50
DEFAULT_N_TEST = 200

DEFAULT_FIXED = {
    "max_depth": None,
    "context_depth": None,
    "predecessor_threshold": 1.0,
    "n_training": DEFAULT_N_TRAINING,
    "n_test": DEFAULT_N_TEST,
}

# ---------------------------------------------------------------------------
# Standard metric definitions
# ---------------------------------------------------------------------------

STANDARD_METRICS = [
    ("avg_step_loss", "Avg Step Loss", "lower is better"),
    ("max_step_loss", "Max Step Loss", "lower is better"),
    ("exact_pct", "Exact Match [%]", "higher is better"),
    ("total_entrypoints", "Entrypoints", "lower is better"),
    ("final_states", "Cache States", "lower is better"),
    ("total_route", "Route Calls", "lower is better"),
    ("total_remote", "Remote Calls", "lower is better"),
    ("total_compute", "Explored States", "lower is better"),
    ("total_cost_evals", "Cost Evals", "lower is better"),
    ("total_queried", "Total Queries", "lower is better"),
    ("elapsed_s", "Elapsed Time [s]", "lower is better"),
]

QUICK_METRICS = [
    ("avg_step_loss", "Avg Step Loss", "lower is better"),
    ("exact_pct", "Exact Match [%]", "higher is better"),
    ("total_route", "Route Calls", "lower is better"),
    ("total_remote", "Remote Calls", "lower is better"),
    ("final_states", "Cache States", "lower is better"),
    ("elapsed_s", "Elapsed Time [s]", "lower is better"),
]
