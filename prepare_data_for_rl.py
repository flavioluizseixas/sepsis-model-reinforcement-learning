import pandas as pd
import numpy as np

def discretize_actions(df, iv_col='iv_input', vaso_col='vaso_input'):
    """
    Discretiza variáveis contínuas de fluido IV e vasopressor em 5 níveis cada (5x5 = 25 ações).
    """
    df = df.copy()

    iv_bins = pd.qcut(df[iv_col].fillna(0), 5, labels=False, duplicates='drop')
    vaso_bins = pd.qcut(df[vaso_col].fillna(0), 5, labels=False, duplicates='drop')

    iv_bins = iv_bins.fillna(0).astype(int)
    vaso_bins = vaso_bins.fillna(0).astype(int)

    df["action"] = (iv_bins * 5 + vaso_bins).astype(int)
    return df


def assign_rewards(df, cohort_df):
    """
    Atribui recompensa terminal com base na mortalidade:
    +1 se sobreviveu, -1 se morreu (apenas no último timestamp por stay_id).
    """
    df = df.copy()
    cohort_df = cohort_df.copy()

    cohort_df["mortality_icu"] = cohort_df["deathtime"].notna() & (cohort_df["deathtime"] <= cohort_df["outtime"])
    cohort_df["mortality_icu"] = cohort_df["mortality_icu"].astype(int)

    df = df.merge(cohort_df[["stay_id", "outtime", "mortality_icu"]], on="stay_id", how="left")
    df["is_terminal"] = df["hour"] == df.groupby("stay_id")["hour"].transform("max")

    df["reward"] = 0
    df.loc[df["is_terminal"] & (df["mortality_icu"] == 0), "reward"] = 1
    df.loc[df["is_terminal"] & (df["mortality_icu"] == 1), "reward"] = -1

    return df.drop(columns=["outtime", "mortality_icu", "is_terminal"])


def build_transitions(df, state_cols):
    """
    Constrói lista de tuplas (s, a, r, s') agrupadas por stay_id.
    """
    df = df.sort_values(["stay_id", "hour"]).reset_index(drop=True)
    transitions = []

    for _, group in df.groupby("stay_id"):
        group = group.reset_index(drop=True)
        for i in range(len(group) - 1):
            s = group.loc[i, state_cols].values
            a = group.loc[i, "action"]
            r = group.loc[i, "reward"]
            s_next = group.loc[i + 1, state_cols].values
            transitions.append((s, a, r, s_next))

    return transitions
