"""Useful constants and helper functions."""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve

# All variables that we initially flagged as potential predictors of weaning
PREDICTORS_1 = [
    'hour','PARALYSIS','CHRONIC_PULMONARY','OBESITY','elixhauser_score',
    'age','gender','admission_type', 'admission_location','insurance',
    'marital_status','ethnicity','language','first_careunit','imputed_IBW',
    'hour_baseline','imputed_height'
]
PREDICTORS_2 = [f'last_{c}' for c in (
    'peep_set','tidal_volume_set','resp_rate','peak_insp_pressure','plateau_pressure',
    'mean_airway_pressure','minutes_vol','pao2fio2ratio','driving_pressure', 'po2','pco2','aado2_calc',
    'ph','baseexcess','totalco2','temperature','fio2_chartevents', 'urine_output',
    'lactate','glucose','gcs','gcs_verbal','gcs_motor','gcs_eyes','gcs_unable',
    'CREATININE','PLATELET','PTT','BUN','WBC','sbp','dbp','mbp','spo2','heart_rate',
    'respiration_24hours','coagulation_24hours','liver_24hours','cardiovascular_24hours',
    'cns_24hours','renal_24hours','any_vaso','amount','weight','imputed_TV_standardized',
    'opioid','benzo','propofol','dex'
)]

ALL_WEANING_PREDICTORS = PREDICTORS_1 + PREDICTORS_2

def group_by_stay(df: pd.DataFrame, col: str = None):
    """Group the given dataframe by the 'stay_id' column.
    
    Returns either the grouped dataframe or optionally, just one grouped column."""
    grouped = df.groupby('stay_id', axis='rows', group_keys=False)
    return grouped[col] if col else grouped
