import pandas as pd
import numpy as np
from dddqn import DDDQNAgent
from utils import preprocess_data, SepsisEnv
from sklearn.model_selection import train_test_split
import prepare_data_for_rl as prep
import pickle

# =========================
# ======= RL PIPELINE =====
# =========================

# 6. Carregar e normalizar
print("🔄 Pré-processando para RL...")
df = pd.read_csv("output/mimiciv_sepsis_processed.csv")
data = preprocess_data(df)
data = data.fillna(0)

# 7. Ações e recompensas
data = prep.discretize_actions(data, iv_col="iv_input", vaso_col="vaso_input")
df_cohort = pd.read_parquet("output/df_cohort.parquet")
data = prep.assign_rewards(data, df_cohort)

print(data["reward"].value_counts(dropna=False))
print(data["reward"].isna().sum())

# 8. Treino/teste
# Separar IDs de pacientes
stay_ids = data["stay_id"].unique()
train_ids, test_ids = train_test_split(stay_ids, test_size=0.2, random_state=42)

# Filtrar dados
train_data = data[data["stay_id"].isin(train_ids)].copy()
test_data = data[data["stay_id"].isin(test_ids)].copy()

print(f"Número de linhas treino: {len(train_data)}")
print(f"Número de pacientes treino: {train_data['stay_id'].nunique()}")
print(f"Número de linhas teste: {len(test_data)}")
print(f"Número de pacientes teste: {test_data['stay_id'].nunique()}")

#train_data = train_data.groupby("stay_id").head(5)  # pega até 5 timesteps por paciente
#env = SepsisEnv(sample_data)

# 9. Ambiente offline
env = SepsisEnv(train_data)
# Salvar
with open("output/sepsis_env.pkl", "wb") as f:
    pickle.dump(env, f)
# Carregar depois
#with open("sepsis_env.pkl", "rb") as f:
#    env_loaded = pickle.load(f)

# 10. Agente
agent = DDDQNAgent(state_dim=env.state_dim, action_dim=env.action_dim, lr=1e-4, replay_buffer=env.get_buffer())

# 11. Treinamento
print("🚀 Treinando agente...")
agent.train(episodes=200)

# 12. Avaliação
print("📊 Avaliando agente...")
env_test = SepsisEnv(test_data)
# Salvar
with open("output/sepsis_env_test.pkl", "wb") as f:
    pickle.dump(env_test, f)
# Carregar depois
#with open("sepsis_env.pkl", "rb") as f:
#    env_loaded = pickle.load(f)

agent.evaluate(env_test)