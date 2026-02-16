#!/usr/bin/env python3
"""
Generate a synthetic airline operations dataset for HW3 Q3 with the same
SCM-consistent structure used by the notebook fallback. Saves to
dataset/airline_synthetic.csv so the notebook can load it if no real
airline CSV is provided.

Run from HW3 root: python scripts/generate_airline_synthetic.py
"""

import os
import sys

import numpy as np
import pandas as pd

# Resolve HW3 root (parent of scripts/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HW3_ROOT = os.path.dirname(SCRIPT_DIR)
DATASET_DIR = os.path.join(HW3_ROOT, "dataset")
OUT_PATH = os.path.join(DATASET_DIR, "airline_synthetic.csv")

AIRLINE_COLS = [
    "Booking_Mode",
    "Marketing_Budget",
    "Website_Visits",
    "Ticket_Price",
    "Tickets_Sold",
    "Sales_Revenue",
    "Operating_Expenses",
    "Profit",
]


def generate(seed: int = 0, n: int = 365) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    booking = rng.binomial(1, 0.22, size=n)
    marketing = 1200 + 850 * booking + rng.normal(0, 120, size=n)
    website = 12000 + 2.4 * marketing + 2800 * booking + rng.normal(0, 900, size=n)
    ticket_price = 420 + 170 * booking + rng.normal(0, 35, size=n)
    tickets_sold = 1800 + 0.30 * website - 2.0 * ticket_price + 900 * booking + rng.normal(0, 300, size=n)
    tickets_sold = np.clip(tickets_sold, 100, None)
    sales = ticket_price * tickets_sold + rng.normal(0, 40000, size=n)
    op_exp = 900000 + 170 * marketing + 130 * tickets_sold + rng.normal(0, 30000, size=n)
    profit = sales - op_exp
    return pd.DataFrame({
        "Booking_Mode": booking.astype(bool),
        "Marketing_Budget": marketing,
        "Website_Visits": website,
        "Ticket_Price": ticket_price,
        "Tickets_Sold": tickets_sold,
        "Sales_Revenue": sales,
        "Operating_Expenses": op_exp,
        "Profit": profit,
    })


def main() -> None:
    seed = int(os.environ.get("HW3_AIRLINE_SEED", "0"))
    df = generate(seed=seed)
    os.makedirs(DATASET_DIR, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print("Saved:", OUT_PATH)
    print("Rows:", len(df))
    print("Columns:", list(df.columns))


if __name__ == "__main__":
    main()
    sys.exit(0)
