from __future__ import annotations
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from config import (RAW_DATA_PATH, TRAIN_DATA_PATH, VALIDATION_DATA_PATH, TEST_DATA_PATH,
                    DATA_DIR, RANDOM_SEED)

REQ_COLS = ["NEW_llama_output", "text"]

def load_and_split_data(test_size=0.1, validation_size=0.1, force_resplit=False, test_run=False):
    if TRAIN_DATA_PATH.exists() and VALIDATION_DATA_PATH.exists() and TEST_DATA_PATH.exists() and not force_resplit:
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        val_df = pd.read_csv(VALIDATION_DATA_PATH)
        test_df = pd.read_csv(TEST_DATA_PATH)
    else:
        try:
            df = pd.read_excel(RAW_DATA_PATH)
        except Exception as e:
            print(f"[ERROR] Failed to load: {RAW_DATA_PATH} -> {e}")
            sys.exit(1)
        for c in REQ_COLS:
            if c not in df.columns:
                print(f"[ERROR] Missing required column: {c}")
                sys.exit(1)
        if "bug_id" not in df.columns:
            df.insert(0, "bug_id", range(len(df)))

        train_val, test_df = train_test_split(df, test_size=test_size, random_state=RANDOM_SEED)
        train_df, val_df = train_test_split(
            train_val, test_size=validation_size/(1-test_size), random_state=RANDOM_SEED
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_csv(TRAIN_DATA_PATH, index=False)
        val_df.to_csv(VALIDATION_DATA_PATH, index=False)
        test_df.to_csv(TEST_DATA_PATH, index=False)

    if test_run:
        train_df = train_df.head(100).copy()
        val_df = val_df.head(10).copy()
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
