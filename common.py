"""Useful constants and helper functions."""

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from sklearn.calibration import calibration_curve

# All variables that we initially flagged as potential predictors of weaning
ALL_WEANING_PREDICTORS = [
    'tidal_volume_set', 'tidal_volume_observed', 'plateau_pressure', 'fio2',
    'ventilator_type', 'peep_set', 'ventilator_mode', 'total_peep_level',
    'peak_insp_pressure', 'mean_airway_pressure', 'inspiratory_time', 'spo2', 'po2',
    'pco2', 'aado2_calc', 'pao2fio2ratio', 'ph', 'baseexcess', 'BICARBONATE', 'totalco2',
    'LACTATE', 'gcs', 'gcsmotor', 'gcsverbal', 'gcseyes', 'heartrate', 'sysbp', 'diasbp',
    'meanbp', 'resprate', 'tempc', 'PARALYSIS', 'CHRONIC_PULMONARY', 'OBESITY',
    'driving_pressure', 'imputed_TV_standardized', 'HEMATOCRIT', 'HEMOGLOBIN',
    'carboxyhemoglobin', 'methemoglobin', 'ANIONGAP', 'ALBUMIN', 'BANDS', 'BILIRUBIN',
    'CREATININE', 'PLATELET', 'PTT', 'INR', 'PT', 'BUN', 'WBC', 'urine_output', 'GLUCOSE',
    'weight', 'HYPERTENSION', 'DIABETES_UNCOMPLICATED', 'DIABETES_COMPLICATED',
    'HYPOTHYROIDISM', 'LIVER_DISEASE', 'AIDS', 'LYMPHOMA', 'METASTATIC_CANCER',
    'SOLID_TUMOR', 'RHEUMATOID_ARTHRITIS', 'COAGULOPATHY', 'DEFICIENCY_ANEMIAS',
    'ALCOHOL_ABUSE', 'DRUG_ABUSE', 'DEPRESSION', 'elixhauser_score', 'first_careunit',
    'admission_type', 'insurance', 'admission_location', 'marital_status', 'ethnicity',
    'age', 'gender', 'imputed_height', 'imputed_IBW'
]

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

    df.loc[df['meanbp'] > 200, 'meanbp'] = np.nan
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
    df.loc[df['last_meanbp'] > 200, 'last_meanbp'] = np.nan
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

    df['sysbp'] = np.minimum(np.maximum(df['sysbp'], -4), 4)
    df['last_sysbp'] = np.minimum(np.maximum(df['last_sysbp'], -4), 4)

    df['diasbp'] = np.minimum(df['diasbp'], 5)
    df['last_diasbp'] = np.minimum(df['last_diasbp'], 5)

    df['meanbp'] = np.minimum(np.maximum(df['meanbp'], -4), 4)
    df['last_meanbp'] = np.minimum(np.maximum(df['last_meanbp'], -4), 4)

    df['resprate'] = np.minimum(df['resprate'], 4)
    df['last_resprate'] = np.minimum(df['last_resprate'], 4)

    df['tempc'] = np.minimum(np.maximum(df['tempc'], -4), 4)
    df['last_tempc'] = np.minimum(np.maximum(df['last_tempc'], -4), 4)

    df['heartrate'] = np.minimum(np.maximum(df['heartrate'], -3), 4)
    df['last_heartrate'] = np.minimum(np.maximum(df['last_heartrate'], -3), 4)

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