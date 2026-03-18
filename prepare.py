"""
Prepare cached single-cell classification data for autoresearch experiments.

Usage:
    uv run prepare.py
"""

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Project-local cache paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "data_cache")
DATASET_PATH = os.path.join(CACHE_DIR, "dataset.pt")
META_PATH = os.path.join(CACHE_DIR, "meta.json")

# ---------------------------------------------------------------------------
# Fixed dataset configuration
# ---------------------------------------------------------------------------

DEFAULT_GROUP = "Set1"
DEFAULT_RANDOM_STATE = 42
LABELS_TO_KEEP = ["PARENT", "TREM2 KO", "R47H", "PLCG2 KO", "P522R"]
RESAMPLE_TARGETS = {"train": 6560, "val": 820, "test": 820}
RESULT_COLUMN = "label_encoded"
LABEL_COLUMN = "label"
DROP_SOURCE_COLUMNS = ["Compound", "Concentration", "Cell Type", "Cell Count", "Unnamed: 28"]
FEATURE_COLUMNS_TO_ANALYZE = [
    "Row",
    "Column",
    "Field",
    "Cells Selected - Cell Area [µm²]",
    "Cells Selected - Cell Roundness",
    "Cells Selected - Cell Ratio Width to Length",
    "Cells Selected - Object No in Cells",
    "Cells Selected - Intensity Cell Alexa 568 Mean",
    "Cells Selected - Total Spot Area",
    "Cells Selected - Relative Spot Intensity",
    "Cells Selected - Number of Spots",
    "Cells Selected - Number of Spots per Area of Cell",
    "Cells Selected - Total Spot Area (2)",
    "Cells Selected - Relative Spot Intensity (2)",
    "Cells Selected - Number of Spots (2)",
    "Cells Selected - Number of Spots per Area of Cell (2)",
]
MODEL_DROP_COLUMNS = [
    "Row",
    "Column",
    "Field",
    LABEL_COLUMN,
    RESULT_COLUMN,
    "Cells Selected - Object No in Cells",
]

GROUP_RULES = {
    "Set2": {"20250429-120531", "20250429-164932", "20250429-193408"},
    "Set1": {"20250430-030744", "20250430-004254", "20250429-221658"},
}

LABEL_MAP = {
    "20250429-120531": {
        2: "PARENT",
        3: "TREM2 KO",
        4: "R47H",
        5: "H157Y",
        6: "PLCG2 KO",
        7: "P522R",
        8: "P522R HET",
        9: "SHIP1 KO",
        10: "ABI3 KO",
        11: "S209F",
    },
    "20250429-164932": {
        2: "PARENT",
        3: "TREM2 KO",
        4: "R47H",
        5: "H157Y",
        6: "PLCG2 KO",
        7: "P522R",
        8: "P522R HET",
        9: "SHIP1 KO",
        10: "ABI3 KO",
        11: "S209F",
    },
    "20250429-193408": {
        2: "PARENT",
        3: "TREM2 KO",
        4: "R47H",
        5: "H157Y",
        6: "PLCG2 KO",
        7: "P522R",
        8: "P522R HET",
        9: "SHIP1 KO",
        10: "ABI3 KO",
        11: "S209F",
    },
    "20250430-030744": {
        2: "PARENT",
        3: "TREM2 KO",
        4: "R47H",
        5: "H157Y",
        6: "PLCG2 KO",
        7: "P522R",
        8: "P522R HET",
        9: "SHIP1 KO",
        10: "ABI3 KO",
        11: "S209F",
    },
    "20250430-004254": {
        2: "PARENT",
        3: "TREM2 KO",
        4: "R47H",
        5: "H157Y",
        6: "PLCG2 KO",
        7: "P522R",
        8: "P522R HET",
        9: "SHIP1 KO",
        10: "ABI3 KO",
        11: "S209F",
    },
    "20250429-221658": {
        2: "PARENT",
        3: "TREM2 KO",
        4: "R47H",
        5: "H157Y",
        6: "PLCG2 KO",
        7: "P522R",
        8: "P522R HET",
        9: "SHIP1 KO",
        10: "ABI3 KO",
        11: "S209F",
    },
}

ROW_SELECTION_MAP = {
    "20250429-120531": [2, 3, 4],
    "20250429-164932": [5, 6, 7],
    "20250429-193408": [2, 3, 4],
    "20250430-030744": [2, 3, 4],
    "20250430-004254": [5, 6, 7],
    "20250429-221658": [2, 3, 4],
}


def sanitize_column_name(column_name):
    return (
        column_name.replace("[", "")
        .replace("]", "")
        .replace("µm²", "um2")
        .replace("<", "")
        .replace(">", "")
    )


def resolve_raw_data_root(explicit_root=None):
    checked_paths = []
    candidate_paths = [
        explicit_root,
        os.environ.get("AUTORESEARCH_RAW_DATA_ROOT"),
        "/Users/ph23568/Documents/UoB/Trem2/ExtractedFeatures",
        "/home/b35am/berlin.b35am/Trem2/Data/ExtractedFeatures",
    ]
    for path in candidate_paths:
        if not path:
            continue
        checked_paths.append(path)
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        "Could not find the raw data directory. Pass --root_path or set "
        f"AUTORESEARCH_RAW_DATA_ROOT. Checked: {checked_paths}"
    )


def discover_grouped_paths(root_path):
    baseline_folders = sorted(
        folder_name
        for folder_name in os.listdir(root_path)
        if os.path.isdir(os.path.join(root_path, folder_name)) and folder_name.startswith("2025")
    )

    evaluation_folders = []
    for folder_name in baseline_folders:
        folder_path = os.path.join(root_path, folder_name)
        subfolders = sorted(
            subfolder
            for subfolder in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, subfolder))
        )
        if subfolders:
            evaluation_folders.append(os.path.join(folder_path, subfolders[0]))

    cells_selected_paths = {}
    for folder_path in evaluation_folders:
        matches = sorted(name for name in os.listdir(folder_path) if "Cells Selected" in name)
        if not matches:
            print(f"Warning: no 'Cells Selected' file found in {folder_path}")
            continue
        file_path = os.path.join(folder_path, matches[0])
        parent_dir = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        parts = parent_dir.split("_")
        if len(parts) >= 3:
            run_id = parts[0]
            cells_selected_paths[run_id] = file_path

    grouped_paths = defaultdict(list)
    for run_id, file_path in cells_selected_paths.items():
        matched_group = None
        for group_name, run_ids in GROUP_RULES.items():
            if run_id in run_ids:
                matched_group = group_name
                grouped_paths[group_name].append({run_id: file_path})
                break
        if matched_group is None:
            print(f"Warning: run {run_id} did not match any configured group")

    return grouped_paths


def load_group_frames(grouped_paths, group_name):
    if group_name not in grouped_paths:
        raise ValueError(f"Group '{group_name}' not found. Available groups: {sorted(grouped_paths.keys())}")

    dataframes = {}
    for item in grouped_paths[group_name]:
        run_id, file_path = next(iter(item.items()))
        dataframe = pd.read_csv(file_path, sep="\t", skiprows=9)
        dataframe = dataframe.drop(columns=DROP_SOURCE_COLUMNS)
        dataframe = dataframe[FEATURE_COLUMNS_TO_ANALYZE].copy()

        rows = ROW_SELECTION_MAP.get(run_id)
        if rows is not None:
            dataframe = dataframe[dataframe["Row"].isin(rows)].copy()

        dataframe[LABEL_COLUMN] = dataframe["Column"].map(LABEL_MAP[run_id])
        dataframes[run_id] = dataframe.reset_index(drop=True)

    return dataframes


def split_by_row(merged_dataframe, random_state):
    train_parts, val_parts, test_parts = [], [], []
    for _, group in merged_dataframe.groupby("Row"):
        train_group, temp_group = train_test_split(group, test_size=0.2, random_state=random_state)
        val_group, test_group = train_test_split(temp_group, test_size=0.5, random_state=random_state)
        train_parts.append(train_group)
        val_parts.append(val_group)
        test_parts.append(test_group)

    train_dataframe = pd.concat(train_parts).reset_index(drop=True)
    val_dataframe = pd.concat(val_parts).reset_index(drop=True)
    test_dataframe = pd.concat(test_parts).reset_index(drop=True)
    return train_dataframe, val_dataframe, test_dataframe


def resample_by_class(dataframe, target_size, label_col=RESULT_COLUMN, random_state=DEFAULT_RANDOM_STATE):
    resampled_frames = []
    for _, group in dataframe.groupby(label_col):
        group_size = len(group)
        if group_size > target_size:
            sampled = group.sample(n=target_size, replace=False, random_state=random_state)
        elif group_size < target_size:
            sampled = group.sample(n=target_size, replace=True, random_state=random_state)
        else:
            sampled = group
        resampled_frames.append(sampled)

    resampled_dataframe = pd.concat(resampled_frames, axis=0)
    resampled_dataframe = resampled_dataframe.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return resampled_dataframe


def build_split_dataframe(features, encoded_labels, raw_labels):
    return pd.concat(
        [
            features.reset_index(drop=True),
            encoded_labels.reset_index(drop=True),
            raw_labels.reset_index(drop=True),
        ],
        axis=1,
    )


def dataframe_to_tensors(features_dataframe, labels_series):
    features_tensor = torch.tensor(features_dataframe.values, dtype=torch.float32)
    labels_tensor = torch.tensor(labels_series.values, dtype=torch.long)
    return features_tensor, labels_tensor


def prepare_dataset(raw_data_root, group_name, force=False):
    if os.path.exists(DATASET_PATH) and not force:
        print(f"Cache already exists at {DATASET_PATH}")
        print("Use --force to rebuild it.")
        return

    grouped_paths = discover_grouped_paths(raw_data_root)
    dataframes = load_group_frames(grouped_paths, group_name)

    merged_dataframe = pd.concat(dataframes.values(), ignore_index=True)
    merged_dataframe = merged_dataframe[merged_dataframe[LABEL_COLUMN].isin(LABELS_TO_KEEP)].copy()

    label_encoder = LabelEncoder()
    merged_dataframe[RESULT_COLUMN] = label_encoder.fit_transform(merged_dataframe[LABEL_COLUMN])
    merged_dataframe.columns = [sanitize_column_name(name) for name in merged_dataframe.columns]

    train_dataframe, val_dataframe, test_dataframe = split_by_row(merged_dataframe, DEFAULT_RANDOM_STATE)

    x_train_raw = train_dataframe.drop(MODEL_DROP_COLUMNS, axis=1)
    y_train_raw = train_dataframe[RESULT_COLUMN]
    y_train_label = train_dataframe[LABEL_COLUMN]

    x_val_raw = val_dataframe.drop(MODEL_DROP_COLUMNS, axis=1)
    y_val_raw = val_dataframe[RESULT_COLUMN]
    y_val_label = val_dataframe[LABEL_COLUMN]

    x_test_raw = test_dataframe.drop(MODEL_DROP_COLUMNS, axis=1)
    y_test_raw = test_dataframe[RESULT_COLUMN]
    y_test_label = test_dataframe[LABEL_COLUMN]

    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train_raw), columns=x_train_raw.columns)
    x_val_scaled = pd.DataFrame(scaler.transform(x_val_raw), columns=x_val_raw.columns)
    x_test_scaled = pd.DataFrame(scaler.transform(x_test_raw), columns=x_test_raw.columns)

    train_pre_resample = build_split_dataframe(x_train_scaled, y_train_raw, y_train_label)
    val_pre_resample = build_split_dataframe(x_val_scaled, y_val_raw, y_val_label)
    test_pre_resample = build_split_dataframe(x_test_scaled, y_test_raw, y_test_label)

    train_sampled = resample_by_class(train_pre_resample, RESAMPLE_TARGETS["train"])
    val_sampled = resample_by_class(val_pre_resample, RESAMPLE_TARGETS["val"])
    test_sampled = resample_by_class(test_pre_resample, RESAMPLE_TARGETS["test"])

    x_train = train_sampled.drop([LABEL_COLUMN, RESULT_COLUMN], axis=1)
    y_train = train_sampled[RESULT_COLUMN]
    x_val = val_sampled.drop([LABEL_COLUMN, RESULT_COLUMN], axis=1)
    y_val = val_sampled[RESULT_COLUMN]
    x_test = test_sampled.drop([LABEL_COLUMN, RESULT_COLUMN], axis=1)
    y_test = test_sampled[RESULT_COLUMN]

    train_features, train_labels = dataframe_to_tensors(x_train, y_train)
    val_features, val_labels = dataframe_to_tensors(x_val, y_val)
    test_features, test_labels = dataframe_to_tensors(x_test, y_test)
    test_original_features, test_original_labels = dataframe_to_tensors(x_test_scaled, y_test_raw)

    dataset = {
        "train_features": train_features,
        "train_labels": train_labels,
        "val_features": val_features,
        "val_labels": val_labels,
        "test_features": test_features,
        "test_labels": test_labels,
        "test_original_features": test_original_features,
        "test_original_labels": test_original_labels,
        "feature_names": x_train.columns.tolist(),
        "class_names": label_encoder.classes_.tolist(),
        "group_name": group_name,
        "raw_data_root": raw_data_root,
        "source_run_ids": sorted(dataframes.keys()),
        "resample_targets": RESAMPLE_TARGETS,
    }

    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(dataset, DATASET_PATH)

    metadata = {
        "cache_dir": CACHE_DIR,
        "dataset_path": DATASET_PATH,
        "group_name": group_name,
        "raw_data_root": raw_data_root,
        "source_run_ids": sorted(dataframes.keys()),
        "class_names": label_encoder.classes_.tolist(),
        "feature_names": x_train.columns.tolist(),
        "input_dim": len(x_train.columns),
        "split_shapes": {
            "train": list(x_train.shape),
            "val": list(x_val.shape),
            "test": list(x_test.shape),
            "test_original": list(x_test_scaled.shape),
        },
        "resample_targets": RESAMPLE_TARGETS,
    }
    with open(META_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    print(f"Saved dataset cache to {DATASET_PATH}")
    print(f"Saved metadata to {META_PATH}")
    print(f"Using raw data root: {raw_data_root}")
    print(f"Group: {group_name}")
    print(f"Input dim: {metadata['input_dim']}")
    print(f"Classes: {metadata['class_names']}")
    for split_name, shape in metadata["split_shapes"].items():
        print(f"{split_name}_shape: {shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare cached single-cell classification data")
    parser.add_argument("--root_path", type=str, default=None, help="Raw feature directory")
    parser.add_argument("--group", type=str, default=DEFAULT_GROUP, help="Configured group name")
    parser.add_argument("--force", action="store_true", help="Rebuild the cache even if it already exists")
    arguments = parser.parse_args()

    raw_data_root = resolve_raw_data_root(arguments.root_path)
    prepare_dataset(raw_data_root=raw_data_root, group_name=arguments.group, force=arguments.force)
