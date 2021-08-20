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

def create_calibration_curve(model, X, y, plt_title, save_path=None):
    probability_estimates = model.predict_proba(X)[:, 1]
    prob_true, prob_pred = calibration_curve(y, probability_estimates, n_bins=10)

    _ , ax = plt.subplots(nrows=1, ncols=1)
    ax.axline([0, 0], [1, 1], color='black', alpha=0.3)
    plt.title(plt_title)
    plt.plot(prob_pred, prob_true)
    if save_path:
        plt.savefig(save_path, facecolor='white')

def remove_outliers(df: pd.DataFrame):
    """Process the data and remove outliers, inplace."""
    df.loc[:, 'amount'] = df['amount']/1000
    df.loc[:, 'last_amount'] = df['last_amount']/1000

    df.loc[df['driving_pressure'] <= 0, 'driving_pressure'] = np.nan

    df.loc[df['mbp'] > 200, 'mbp'] = np.nan
    df.loc[df['rr_set_set'] <= 0, 'rr_set_set'] = np.nan
    df.loc[df['peak_insp_pressure'] <= 0, 'peak_insp_pressure'] = np.nan
    df.loc[df['mean_airway_pressure'] <= 0, 'mean_airway_pressure'] = np.nan
    df.loc[df['inspiratory_time'] <= 0, 'inspiratory_time'] = np.nan
    df.loc[df['aado2_calc'] <= 0, 'aado2_calc'] = np.nan
    df.loc[df['imputed_TV_standardized'] < 2, 'imputed_TV_standardized'] = np.nan
    df.loc[df['imputed_TV_standardized'] > 12, 'imputed_TV_standardized'] = np.nan
    df.loc[df['tidal_volume_set'] < 200, 'imputed_TV_standardized'] = np.nan
    df.loc[df['tidal_volume_set'] < 200, 'imputed_TV_standardized'] = np.nan
    df.loc[df['tidal_volume_set'] < 200, 'tidal_volume_set'] = np.nan

    df.loc[df['plateau_pressure'] > 45, 'plateau_pressure'] = np.nan

    condition = (np.log(df['peak_insp_pressure']) > 4) | (np.log(df['peak_insp_pressure']) < 1)
    df.loc[condition, 'peak_insp_pressure'] = np.nan

    df.loc[df['mean_airway_pressure'] > 50, 'mean_airway_pressure'] = np.nan
    df.loc[df['inspiratory_time'] > 1.5, 'inspiratory_time'] = np.nan
    df.loc[:, 'pao2fio2ratio'] = np.minimum(df['pao2fio2ratio'], 600)

    df.loc[df['last_driving_pressure'] <= 0, 'last_driving_pressure'] = np.nan
    df.loc[df['last_mbp'] > 200, 'last_mbp'] = np.nan
    df.loc[df['last_rr_set_set'] <= 0, 'last_rr_set_set'] = np.nan
    df.loc[df['last_peak_insp_pressure'] <= 0, 'last_peak_insp_pressure'] = np.nan
    df.loc[df['last_mean_airway_pressure'] <= 0, 'last_mean_airway_pressure'] = np.nan
    df.loc[df['last_inspiratory_time'] <= 0, 'last_inspiratory_time'] = np.nan
    df.loc[df['last_aado2_calc'] <= 0, 'last_aado2_calc'] = np.nan
    df.loc[df['last_imputed_TV_standardized'] < 2, 'last_imputed_TV_standardized'] = np.nan
    df.loc[df['last_imputed_TV_standardized'] > 12, 'last_imputed_TV_standardized'] = np.nan
    df.loc[df['last_tidal_volume_set'] < 200, 'last_imputed_TV_standardized'] = np.nan
    df.loc[df['last_tidal_volume_set'] < 200, 'last_tidal_volume_set'] = np.nan

    df.loc[df['last_plateau_pressure'] > 45, 'last_plateau_pressure'] = np.nan

    condition = (np.log(df['last_peak_insp_pressure']) > 4) | (np.log(df['last_peak_insp_pressure']) < 1)
    df.loc[condition, 'last_peak_insp_pressure'] = np.nan

    df.loc[df['last_mean_airway_pressure'] > 50, 'last_mean_airway_pressure'] = np.nan
    df.loc[df['last_inspiratory_time'] > 1.5, 'last_inspiratory_time'] = np.nan
    df.loc[:, 'last_pao2fio2ratio'] = np.minimum(df['last_pao2fio2ratio'], 600)

def remove_extremes(df: pd.DataFrame):
    df['rate_std'] = np.minimum(df['rate_std'], 10)
    df['last_rate_std'] = np.minimum(df['last_rate_std'], 10)

    df['sbp'] = np.minimum(np.maximum(df['sbp'], -4), 4)
    df['last_sbp'] = np.minimum(np.maximum(df['last_sbp'], -4), 4)

    df['dbp'] = np.minimum(df['dbp'], 5)
    df['last_dbp'] = np.minimum(df['last_dbp'], 5)

    df['mbp'] = np.minimum(np.maximum(df['mbp'], -4), 4)
    df['last_mbp'] = np.minimum(np.maximum(df['last_mbp'], -4), 4)

    df['resp_rate'] = np.minimum(df['resp_rate'], 4)
    df['last_resp_rate'] = np.minimum(df['last_resp_rate'], 4)

    df['temperature'] = np.minimum(np.maximum(df['temperature'], -4), 4)
    df['last_temperature'] = np.minimum(np.maximum(df['last_temperature'], -4), 4)

    df['heart_rate'] = np.minimum(np.maximum(df['heart_rate'], -3), 4)
    df['last_heart_rate'] = np.minimum(np.maximum(df['last_heart_rate'], -3), 4)

    df['spo2'] = np.maximum(df['spo2'], -4)
    df['last_spo2'] = np.maximum(df['last_spo2'], 4)

    df['po2'] = np.minimum(df['po2'], 5)
    df['last_po2'] = np.minimum(df['last_po2'], 5)

    df['pco2'] = np.minimum(df['pco2'], 5)
    df['last_pco2'] = np.minimum(df['last_pco2'], 5)

    df['inspiratory_time'] = np.minimum(np.maximum(df['inspiratory_time'], -3), 4)
    df['last_inspiratory_time'] = np.minimum(np.maximum(df['last_inspiratory_time'], -3), 4)

    df['imputed_TV_standardized'] = np.maximum(df['imputed_TV_standardized'], 4)
    df['last_imputed_TV_standardized'] = np.maximum(df['last_imputed_TV_standardized'], 4)

    df['tidal_volume_set'] = np.minimum(df['tidal_volume_set'], 650)
    df['last_tidal_volume_set'] = np.minimum(df['last_tidal_volume_set'], 650)

    df['mean_airway_pressure'] = np.minimum(df['mean_airway_pressure'], 4)
    df['last_mean_airway_pressure'] = np.minimum(df['last_mean_airway_pressure'], 4)

    df['amount'] = np.minimum(df['amount'], 10)
    df['last_amount'] = np.minimum(df['last_amount'], 10)

    df['driving_pressure'] = np.minimum(df['driving_pressure'], 30)
    df['last_driving_pressure'] = np.minimum(df['last_driving_pressure'], 30)

    df['ph'] = np.maximum(df['ph'], -4)
    df['last_ph'] = np.maximum(df['last_ph'], -4)

    df['peep_set'] = np.maximum(np.minimum(df['peep_set'], -1), 4)
    df['last_peep_set'] = np.maximum(np.minimum(df['last_peep_set'], -1), 4)