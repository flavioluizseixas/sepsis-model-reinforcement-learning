import pandas as pd
import numpy as np
from google.cloud import bigquery
from tqdm import tqdm
from backup.dddqn import DDDQNAgent
from backup.utils import preprocess_data, SepsisEnv
import prepare_data_for_rl as prep
from sklearn.model_selection import train_test_split

# 1. Coorte Sepse
print("🔎 Carregando coorte com sepse...")
client = bigquery.Client()
query_cohort = """
WITH sepsis_patients AS (
  SELECT DISTINCT ie.subject_id, ie.stay_id, ie.hadm_id, ie.intime, ie.outtime
  FROM `physionet-data.mimiciv_3_1_icu.icustays` ie
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.diagnoses_icd` d
    ON ie.hadm_id = d.hadm_id
  WHERE d.icd_code LIKE 'A41%'
)
SELECT sp.*, p.gender, p.anchor_age, p.anchor_year, a.deathtime
FROM sepsis_patients sp
JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON p.subject_id = sp.subject_id
JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a ON a.hadm_id = sp.hadm_id
"""
df_cohort = client.query(query_cohort).to_dataframe()
df_cohort.to_parquet("output/df_cohort.parquet", index=False)
print(f"✅ {len(df_cohort)} pacientes com sepse identificados")

stay_ids = [int(x) for x in df_cohort['stay_id'].unique()]
all_data = []

# 2. Variáveis clínicas
item_ids = {
    "HR": 220045, "SBP": 220050, "DBP": 220051, "Temp": 223761,
    "RR": 220210, "SpO2": 220277, "WBC": 220546,
    "Creatinine": 220615, "BUN": 220615, "GCS": 223900,
}

print("📥 Extraindo sinais vitais...")
for name, itemid in tqdm(item_ids.items()):
    query = f"""
        SELECT subject_id, stay_id, charttime, valuenum AS value, '{name}' AS variable
        FROM `physionet-data.mimiciv_3_1_icu.chartevents`
        WHERE itemid = {itemid}
          AND valuenum IS NOT NULL
          AND stay_id IN UNNEST(@stay_ids)
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids)]
    )
    df_item = client.query(query, job_config=job_config).to_dataframe()
    all_data.append(df_item)

# 3. Fluido IV
print("💧 Extraindo iv_input...")
query_iv = """
SELECT subject_id, stay_id, starttime AS charttime, amount AS value, 'iv_input' AS variable
FROM `physionet-data.mimiciv_3_1_icu.inputevents`
WHERE amount IS NOT NULL
  AND stay_id IN UNNEST(@stay_ids)
"""
df_iv = client.query(query_iv, job_config=bigquery.QueryJobConfig(
    query_parameters=[bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids)]
)).to_dataframe()
df_iv["charttime"] = pd.to_datetime(df_iv["charttime"])
all_data.append(df_iv)

# 4. Vasopressores
print("💉 Extraindo vaso_input...")
vaso_ids = [221906, 221289, 222315, 221749, 221662]
query_vaso = """
SELECT subject_id, stay_id, starttime AS charttime, rate AS value, 'vaso_input' AS variable
FROM `physionet-data.mimiciv_3_1_icu.inputevents`
WHERE itemid IN UNNEST(@item_ids)
  AND rate IS NOT NULL
  AND stay_id IN UNNEST(@stay_ids)
"""
df_vaso = client.query(query_vaso, job_config=bigquery.QueryJobConfig(
    query_parameters=[
        bigquery.ArrayQueryParameter("item_ids", "INT64", vaso_ids),
        bigquery.ArrayQueryParameter("stay_ids", "INT64", stay_ids)
    ]
)).to_dataframe()
df_vaso["charttime"] = pd.to_datetime(df_vaso["charttime"])
all_data.append(df_vaso)

# 5. Unir, agregar por hora e salvar
print("🧼 Processando janelas por hora...")
df_all = pd.concat(all_data)
df_all = df_all.merge(df_cohort[["stay_id", "intime"]], on="stay_id")
df_all["hour"] = ((pd.to_datetime(df_all["charttime"]) - df_all["intime"]) / pd.Timedelta(hours=1)).astype(int)
df_all = df_all[df_all["hour"].between(0, 80)]
df_wide = df_all.pivot_table(index=["stay_id", "hour"], columns="variable", values="value", aggfunc="mean").reset_index()
df_wide.to_csv("output/mimiciv_sepsis_processed.csv", index=False)
print("✅ Dados salvos em output/mimiciv_sepsis_processed.csv")