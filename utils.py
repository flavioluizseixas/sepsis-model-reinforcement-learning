import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class SepsisEnv:
    def __init__(self, data, state_cols=None):
        self.data = data.copy()
        self.state_cols = state_cols if state_cols else self._infer_state_cols()
        self.buffer = self._build_buffer()
        self.state_dim = len(self.state_cols)
        self.action_dim = self.data["action"].nunique()

    def _infer_state_cols(self):
        return [col for col in self.data.columns if col not in ["stay_id", "hour", "action", "reward"]]

    def _build_buffer(self):
        transitions = []
        total_groups = self.data["stay_id"].nunique()
        print(f"Construindo buffer para {total_groups} pacientes...")
        for idx, (_, group) in enumerate(self.data.groupby("stay_id")):
            if idx % 100 == 0:
                print(f"Processando grupo {idx}/{total_groups}")
            group = group.sort_values("hour").reset_index(drop=True)
            for i in range(len(group) - 1):
                try:
                    s = group.loc[i, self.state_cols].values.astype(np.float32)
                    a = group.loc[i, "action"]
                    r = group.loc[i, "reward"]
                    s_next = group.loc[i + 1, self.state_cols].values.astype(np.float32)
                    transitions.append((s, a, r, s_next))
                except Exception as e:
                    print(f"Erro em stay_id={group.loc[i, 'stay_id']} index={i}: {e}")
        print(f"Total de transições coletadas: {len(transitions)}")
        return transitions


    def get_buffer(self):
        return self.buffer


def preprocess_data(df):
    """
    Preenche valores ausentes com forward fill e normaliza colunas numéricas (exceto stay_id, hour).
    """
    df = df.copy()
    df.ffill(inplace=True)
    numeric_cols = df.columns.difference(['stay_id', 'hour'])
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    return df
