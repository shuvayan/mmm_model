import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from config.core import config
from processing.features import AdstockImputer, SaturationImputer
from processing.features import Mapper
from processing.features import SeasonEncoder, HolidayEncoder

ollies_pipe = Pipeline([

    ######### Imputation ###########
    ('adstock_imputation', AdstockImputer(variable = config.process_config.weekday_var, 
                                          date_var= config.process_config.date_var)),
    ('saturation_imputation', SaturationImputer(variable = config.process_config.weathersit_var)),
    
    ######### Mapper ###########
    ('map_yr', Mapper(variable = config.process_config.yr_var, mappings = config.process_config.yr_mappings)),
    
    ('map_mnth', Mapper(variable = config.process_config.mnth_var, mappings = config.process_config.mnth_mappings)),
    
    ('map_season', Mapper(variable = config.process_config.season_var, mappings = config.process_config.season_mappings)),
    
    ('map_weathersit', Mapper(variable = config.process_config.weathersit_var, mappings = config.process_config.weathersit_mappings)),
    
    ('map_holiday', Mapper(variable = config.process_config.holiday_var, mappings = config.process_config.holiday_mappings)),
    
    ('map_workingday', Mapper(variable = config.process_config.workingday_var, mappings = config.process_config.workingday_mappings)),
    
    ('map_hr', Mapper(variable = config.process_config.hr_var, mappings = config.process_config.hr_mappings)),
    
    ######## Handle outliers ########
    ('handle_outliers_temp', OutlierHandler(variable = config.process_config.temp_var)),
    ('handle_outliers_atemp', OutlierHandler(variable = config.process_config.atemp_var)),
    ('handle_outliers_hum', OutlierHandler(variable = config.process_config.hum_var)),
    ('handle_outliers_windspeed', OutlierHandler(variable = config.process_config.windspeed_var)),

    ######## One-hot encoding ########
    ('encode_weekday', WeekdayOneHotEncoder(variable = config.process_config.weekday_var)),

    # Scale features
    ('scaler', StandardScaler()),
    
    # Regressor
    ('model_rf', RandomForestRegressor(n_estimators = config.process_config.n_estimators, 
                                       max_depth = config.process_config.max_depth,
                                      random_state = config.process_config.random_state))
    
    ])
