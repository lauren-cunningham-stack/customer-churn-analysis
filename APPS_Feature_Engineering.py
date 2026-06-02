# Generated from: APPS_Feature_Engineering.ipynb
# Converted at: 2026-04-03T02:17:17.573Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# %%  --------------------------------------------  class: PREPPING / TRANSFORMING / FITTING  --------------------------------------
# # Pipeline
import os 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statistics as s
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split as tts
from feature_engine.outliers import Winsorizer
from feature_engine.transformation import LogCpTransformer
from feature_engine.transformation import YeoJohnsonTransformer
from feature_engine.transformation import PowerTransformer
from sklearn.preprocessing import RobustScaler as RS
from feature_engine.transformation import BoxCoxTransformer
import sklearn
import shap
from scipy.stats import skew
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score as AS
from sklearn.metrics import confusion_matrix
from feature_engine.outliers import Winsorizer
from sklearn.inspection import permutation_importance as PI
import tensorflow as tf
from keras import layers, models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Nadam
from scikeras.wrappers import KerasClassifier
import pickle
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score



import sys
print(sys.executable)

other_app_list = ['usage_app_other','usage_voice_d2d_outgoing','usage_voice_d2nd_outgoing','usage_voice_d2d_incoming','usage_voice_nd2d_incoming',
                  'usage_pack_data','usage_pack_vas']
youtube_list = ['usage_app_youtube_daily','usage_voice_d2d_outgoing','usage_voice_d2nd_outgoing','usage_voice_d2d_incoming','usage_voice_nd2d_incoming',
                  'usage_pack_data','usage_pack_vas']
facebook_list = ['usage_app_facebook_daily','usage_voice_d2d_outgoing','usage_voice_d2nd_outgoing','usage_voice_d2d_incoming','usage_voice_nd2d_incoming',
                  'usage_pack_data','usage_pack_vas']
tiktok_list = ['usage_app_tiktok_daily','usage_voice_d2d_outgoing','usage_voice_d2nd_outgoing','usage_voice_d2d_incoming','usage_voice_nd2d_incoming',
                  'usage_pack_data','usage_pack_vas']
whatsapp_list = ['usage_app_whatsapp_daily','usage_voice_d2d_outgoing','usage_voice_d2nd_outgoing','usage_voice_d2d_incoming','usage_voice_nd2d_incoming',
                  'usage_pack_data','usage_pack_vas']
helakuru_list = ['usage_app_helakuru_daily','usage_voice_d2d_outgoing','usage_voice_d2nd_outgoing','usage_voice_d2d_incoming','usage_voice_nd2d_incoming',
                  'usage_pack_data','usage_pack_vas']

class Preping_df:
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.cutoff_day = None
        self.categorical_col = None
        self.numeric_col = None
        self.day_column = None
        self.last_day_col = None
        self.apps = {}
        self.XY_apps = {}
        self.train_test_apps_split = {}
        
    def app_split(self):
        '''
        Split the dataset into apps dataframe.
        '''
        print("Splitting the Dataframe into Apss ...\n")  
        DataFrame = self.dataframe.copy()       
        self.apps['app_other'] = DataFrame[DataFrame['usage_type'].isin(other_app_list)]
        self.apps['youtube'] = DataFrame[DataFrame['usage_type'].isin(youtube_list)]
        self.apps['facebook'] = DataFrame[DataFrame['usage_type'].isin(facebook_list)]
        self.apps['tiktok'] = DataFrame[DataFrame['usage_type'].isin(tiktok_list)]
        self.apps['whatsapp'] = DataFrame[DataFrame['usage_type'].isin(whatsapp_list)]
        self.apps['helakuru'] = DataFrame[DataFrame['usage_type'].isin(helakuru_list)]
        return self

    def clean_df(self):   
        '''
        Will clean this data by: 
             - converting any null-like values & whitespace strings into np.nan
             - remove np.nans
             - drop duplicates
        '''
        print("Cleaning dataframe...\n")         

        for app_name, app_df in self.apps.items():              
            # Replace common null-like values with np.nan
            null_markers = ["", " ", "  ", "NA", "NaN", "nan", "null", "None", None]
            app_df = app_df.replace(null_markers, np.nan)
    
            # Convert pure whitespace strings to NaN
            app_df = app_df.replace(r'^\s*$', np.nan, regex=True)
    
            # Fill NaNs with 0 (if that's the business rule)
            app_df = app_df.fillna(0)       
        
            # Drop duplicates
            before = len(app_df)
            app_df = app_df.drop_duplicates()
            after = len(app_df)
            print(f"Dropped {before - after} duplicates. Remaining rows: {after}")
            
            # Update Dictionary 
            self.apps[app_name] = app_df
        return self  

    def churn_df(self):
        '''
        Create churn labels for each app based on user inactiviy patterns
        
        Churn Definition: 
            - users whose last active day <= cutoff_day
            - Or users inactive for  >= inactivity_window days
        '''
        print("Creating Chrun Col for each App...\n")    
        self.last_day_col = len(df.filter(regex = '^Day_*').columns)
        self.day_column = [f'Day_{i}' for i in range(1, self.last_day_col)]
        cut_off_extra_day_cols = [f'Day_{i}' for i in range(81,self.last_day_col)]
        self.cut_off_day_cols = [f'Day_{i}' for i in range(1,self.last_day_col)]
        self.categorical_col = ['usage_type']
        self.cutoff_day = self.last_day_col - 10
        inactivity_window = 20
        
        for app_name, app_df in self.apps.items(): 
            self.numeric_col = [col for col in app_df.columns if col not in self.categorical_col]
            # Last active day (in actual day numbers, not zero-indexed)
            def get_last_active_day(row, day_cols):
                for i in reversed(range(len(day_cols))):
                    if row[day_cols[i]] > 0:
                        return int(i + 1)   # ensure scalar int, not tuple
                return 0  # if no activity at all
            app_df['last_active_day'] = app_df.apply(lambda row: get_last_active_day(row, self.day_column), axis=1)
    
            # Churn definition
            app_df['churn'] = ((self.cutoff_day - app_df['last_active_day']) >= inactivity_window).astype(int)
            self.apps[app_name] = app_df.drop(cut_off_extra_day_cols, axis = 1)
        return self

    def downsample_subset_df(self):
        '''
         The dataframe is too large for the CPU to handle, thus we must downsample it. 
         For the Model to not pick churner each time, we must match the same amount of churners vs. nonchurners for the model to actually learn
        '''
        print('Downsample each app\'s datasets.')
        for app_name, app_df in self.apps.items():
             # Split the Df in Minority / Majority 
            df_majority = app_df[app_df['churn'] == 0]
            df_minority = app_df[app_df['churn'] == 1]
    
            # Downsample majority class
            df_majority_downsampled = resample(df_majority,
                                      replace = False,              # sample without replacement
                                      n_samples = len(df_minority), # match minority class size
                                      random_state = 42) 
            # Combine minority class with downsampled majority class
            df_downsampled = pd.concat([df_majority_downsampled, df_minority])
    
            # Shuffle the new dataset
            df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)
    
            # subset the whole dataframe to 10%
            subset_df_frac = df_downsampled.sample(frac=0.1, random_state=42) 
    
            # remove aggregated cols to stop data leakage
            agg_columns = ['customer_id','voice_outgoing_usage_sum', 'voice_incoming_usage_sum', 
                                          'pack_usage_sum', 'ave_usage_per_day','total_active_days', 'total_inactive_days', 'last_active_day',
                                          'num_of_usage_burst', 'longest_inactive_gap', 'total_usage', 'app_usage_sum']
            columns_to_drop = [col for col in subset_df_frac.columns if col in agg_columns]
            app_df = subset_df_frac.drop(columns=columns_to_drop)
            self.apps[app_name] = app_df
        return self

    def XY_split(self):
        '''
        Defining what the x and y datasets are for each apps before train test split. 
        '''
        print("Splitting Dataframe into X and Y...\n")
        
        for app_name, app_df in self.apps.items():
            x = app_df.drop(columns = ['churn'])
            x = x[[f'Day_{i}' for i in range(1, self.cutoff_day +1)] + ['usage_type']]
            y = app_df['churn']
            self.categorical_col = ['usage_type']
            self.numeric_col = [col for col in x.columns if col not in self.categorical_col]
            self.XY_apps[app_name] = {'x': x, 'y': y}
        return self 

    def encoding_df(self):
        '''
        There are categorical cols that must be converted using encoding. 
        '''
        print("Encoding the Usage Types...\n")
        for app_name, xy in self.XY_apps.items():
            x = xy['x']
            y = xy['y']
            
            # ct for a DF that i want to convert back into a DF after
            ct = ColumnTransformer(
                transformers=[
                    ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_col),
                    ('num', 'passthrough', self.numeric_col)
                    ]
                )
            # Transform_fit over the x_df
            x = ct.fit_transform(x)
    
            # convert x_df back into a DF since rn its in a array
            ohe_features = ct.named_transformers_['cat'].get_feature_names_out(self.categorical_col)
            all_features = list(ohe_features) + self.numeric_col
            x_encoded = pd.DataFrame(x, columns=all_features)  
    
            # combine the encoded x_df and y_df to create the whole df 
            DataFrame_encoded = pd.concat([x_encoded, y], axis = 1)
            self.XY_apps[app_name] = {'x': x_encoded, 'y':y, 'dataframe':DataFrame_encoded}
            
        return self

    def XY_traintest_split(self):
        print("Train Test Split...\n")  
        for app_name, xy in self.XY_apps.items():
            x = xy['x']
            y = xy['y']
            remove_col = [f'usage_type_usage_{app_name}' if app_name == 'app_other' else f'usage_type_usage_app_{app_name}_daily']
            x = x.drop(columns = remove_col)
            # split the train test sets
            x_train, x_test,y_train,y_test = tts(x,y, test_size = 0.3, random_state = 42)

            self.train_test_apps_split[app_name] = {
                'x_train': x_train,
                'x_test' : x_test,
                'y_train': y_train,
                'y_test' : y_test,
                'x'      : x,
                'y'      : y
                }
        return self 

    def run(self):
        self.app_split()
        self.clean_df()
        self.churn_df()
        self.downsample_subset_df()
        self.XY_split()
        self.encoding_df()
        self.XY_traintest_split()
        return self

#  ______________________________________________________________________________________________________________________________________________
class transform_methods(Preping_df):
    def __init__(self, dataframe, train_test_apps_split):
        '''
        Class that will inherit the Preping Class to then transform the apps dataframes use multiple transformers. 
        '''
        Preping_df.__init__(self,dataframe)
        self.train_test_apps_split = train_test_apps_split
        self.aug_df = None
        self.day_cols = None
        self.apps_split = {}

    def transform_nonzero_with_fitting(self, x_train, x_test = None, transformer_class = None, **kwargs):
        """
        Applies a transformer (fit + transform) to non-zero values only.
        Parameters:
            - X_train, X_test: DataFrames
            - columns: list of column names to transform
            - transformer_class: e.g., PowerTransformer, Winsorizer
            - kwargs: arguments for the transformer (e.g., method='yeo-johnson')
        Returns:
            - X_train_ar, X_test_ar: transformed NumPy arrays
            - X_train_df, X_test_df: transformed DataFrames
        """

        # Create list of days columns at the cutoff day
        last_day_col = len(x_train.filter(regex = '^Day_*').columns)
        self.day_cols = [f'Day_{i}' for i in range(1,last_day_col+1)]
        columns = self.day_cols
        #  -------------------  TRAIN  -------------------------
        for col in columns: 
            transformer = transformer_class(**kwargs)
            # Create mask of non-zero values
            mask_train = x_train[col] > 0
            # Extract non-zero values or True values on df[row:col]
            train_input = x_train.loc[mask_train, [col]]
    
            # Apply transformer to the non-zero df[row:col]
            transformed_train = transformer.fit_transform(train_input)
    
            # Convert arrays to DataFrames so you can safely use .loc
            if not isinstance(transformed_train, pd.DataFrame):
                transformed_train = pd.DataFrame(transformed_train, columns=[col], index=train_input.index)
                
            # Assign back to both DataFrame and Array versions
            x_train.loc[mask_train, col] = transformed_train[col].values.flatten()
            
            #  -------------------  TEST  -------------------------
            if x_test is not None: 
                mask_test = x_test[col] > 0
                test_input = x_test.loc[mask_test, [col]]
                transformed_test = transformer.transform(test_input)
                if not isinstance(transformed_train, pd.DataFrame):
                    transformed_test = pd.DataFrame(transformed_test, columns=[col], index=test_input.index)
                x_test.loc[mask_test, col] = transformed_test[col].values.flatten()

        self.day_cols = columns
        return (x_train, x_test) if x_test is not None else (x_train, None)

    def create_binary_usage_columns(self,x_train, x_test = None, prefix='agu_'):
        """
        Create binary usage columns from daily usage data.
        Parameters:
            - df (pd.DataFrame): Original DataFrame with daily usage columns.
            - day_cols (list of str): Names of the 90 daily usage columns.
            - prefix (str): Prefix to use for binary column names (default: 'bin_').
        Returns:
            - pd.DataFrame: A new DataFrame with binary columns added.
        """
        # Create binary (0/1) version of each daily column
        binary_train = (x_train[self.day_cols] > 0).astype(int)

        # Rename binary columns
        binary_train.columns = [f'{prefix}{col}' for col in self.day_cols]

        # Combine original df with binary columns
        x_train_aug = pd.concat([x_train, binary_train], axis=1)

        if x_test is not None: 
            binary_test =  (x_test[self.day_cols] > 0).astype(int)
            binary_test.columns  = [f'{prefix}{col}' for col in self.day_cols]
            x_test_aug =  pd.concat([x_test, binary_test], axis=1)
            
        return (x_train_aug, x_test_aug) if x_test is not None else (x_train_aug, None)
    
    def Winsorization_T(self):
        """
        Because of usage_type encoded has LOW variance we have to filter that out and then keep it after. 
        """
        for app_name, tts in self.train_test_apps_split.items():
            print(f'Winsorization: IQR and MAD for {app_name}... ')
            y = tts['y']
            x = tts['x']
            x_train = tts['x_train']
            x_test = tts['x_test']
            y_train = tts['y_train']
            y_test = tts['y_test']
            
            # list the columns
            encoded_cols = list(x.columns)
    
            # keep only numeric columns with >2 unique values for winsorization
            numeric_cols_for_winsor = [col for col in encoded_cols if x[col].nunique() > 2 ]
            X_winsor = x[numeric_cols_for_winsor]
    
            # This Transformation only takes DataFrames
            winsor_IQR = Winsorizer(capping_method = 'iqr',
                                 tail = 'both',
                                 variables = X_winsor.columns.tolist()
                               )
            winsor_MAD = Winsorizer(capping_method='quantiles', 
                                    tail='both',
                                    fold=0.02,
                                    variables = X_winsor.columns.tolist()
                                   )
            x_iqr = winsor_IQR.fit_transform(x)
            x_train_iqr = winsor_IQR.fit_transform(x_train)
            x_test_iqr = winsor_IQR.transform(x_test)
            x_mad = winsor_MAD.fit_transform(x)
            x_train_mad = winsor_MAD.fit_transform(x_train)
            x_test_mad = winsor_MAD.transform(x_test)
            x_non_iqr, _ = self.transform_nonzero_with_fitting(x,None,Winsorizer,capping_method = 'iqr',tail = 'both' )
            x_train_non_iqr, x_test_non_iqr = self.transform_nonzero_with_fitting(x_train,x_test,Winsorizer,capping_method = 'iqr',tail = 'both' )
            x_non_mad, _ = self.transform_nonzero_with_fitting(x,None,Winsorizer,capping_method='quantiles',tail='both',fold=0.02)
            x_train_non_mad, x_test_non_mad = self.transform_nonzero_with_fitting(x_train,x_test,Winsorizer,capping_method='quantiles',tail='both',fold=0.02)
            print(f'Non-Binary Winsorization: IQR and MAD for {app_name}... ')
            x_non_aug_iqr, _ = self.create_binary_usage_columns(x_non_iqr, None)
            x_non_aug_mad, _ = self.create_binary_usage_columns(x_non_mad,None)
            x_train_non_aug_iqr, x_test_non_aug_iqr = self.create_binary_usage_columns(x_train_non_iqr,x_test_non_iqr)
            x_train_non_aug_mad, x_test_non_aug_mad = self.create_binary_usage_columns(x_train_non_mad,x_test_non_mad)
            print(f'Adding Binary Cols for: {app_name}... ')
            
            self.apps_split[app_name] = {
                'iqr':{
                    'tts': {
                        'x_train': x_train_iqr,'x_test' : x_test_iqr, 'x_train_non': x_train_non_iqr, 'x_test_non' : x_test_non_iqr,
                        'x_train_non_aug' : x_train_non_aug_iqr, 'x_test_non_aug' : x_test_non_aug_iqr,'y_train': y_train , 'y_test':y_test,
                        'x':x_iqr,'x_non':x_non_iqr,'x_non_aug':x_non_aug_iqr, 'y':y},
                    'model': {}},
                'mad':{
                    'tts': { 
                        'x_train': x_train_mad,'x_test' : x_test_mad, 'x_train_non': x_train_non_mad, 'x_test_non' : x_test_non_mad,
                        'x_train_non_aug' :x_train_non_aug_mad, 'x_test_non_aug' : x_test_non_aug_mad,'y_train': y_train , 'y_test':y_test,
                        'x':x_mad,'x_non':x_non_mad, 'x_non_aug':x_non_aug_mad, 'y':y},
                    'model': {}}}
        return self 

    def Log_SquareRoot_T(self):
        for app_name, tts in self.train_test_apps_split.items():
            print(f'LogCP and Inverse Hyperbolicsine: {app_name} ... ')
            y = tts['y']
            x = tts['x']
            x_train = tts['x_train']
            x_test = tts['x_test']
            y_train = tts['y_train']
            y_test = tts['y_test']
            
            logCP = LogCpTransformer(variables = x.columns.tolist(), C = "auto")
            class ArcsinhTransformer: 
                def fit(self, X):
                    return self
                def transform(self, X):
                    return np.arcsinh(X)
                def fit_transform(self,X):
                    return self.transform(X)
    
            # LogCP transformation on the Train / Test data
            x_log = pd.DataFrame(logCP.fit_transform(x), columns = x.columns, index = x.index)
            x_train_log = pd.DataFrame(logCP.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
            x_test_log = pd.DataFrame(logCP.transform(x_test), columns = x_test.columns, index = x_test.index)
            
            x_non_log, _ = self.transform_nonzero_with_fitting(x, None, LogCpTransformer, C = "auto")
            x_train_non_log, x_test_non_log = self.transform_nonzero_with_fitting(x_train, x_test, LogCpTransformer, C = "auto")

            x_non_aug_log, _ = self.create_binary_usage_columns(x_non_log, None)
            x_train_non_aug_log, x_test_non_aug_log = self.create_binary_usage_columns(x_train_non_log, x_test_non_log)
            print(f'LogCP non-zero & binary cols: {app_name} ... ')
            
            # Inverse Hyperbolic sine
            x_ihs = pd.DataFrame(np.arcsinh(x), columns = x.columns, index = x.index)
            x_train_ihs = pd.DataFrame(np.arcsinh(x_train), columns = x_train.columns, index = x_train.index)
            x_test_ihs = pd.DataFrame(np.arcsinh(x_test), columns = x_test.columns, index = x_test.index)
            
            x_non_ihs, _ = self.transform_nonzero_with_fitting(x, None, ArcsinhTransformer)
            x_train_non_ihs, x_test_non_ihs = self.transform_nonzero_with_fitting(x_train, x_test, ArcsinhTransformer)

            x_non_aug_ihs, _ = self.create_binary_usage_columns(x_non_ihs,None)
            x_train_non_aug_ihs, x_test_non_aug_ihs = self.create_binary_usage_columns(x_train_non_ihs,x_test_non_ihs)
            print(f'Hyperbolic non-zero & binary cols: {app_name} ... ')
            self.apps_split[app_name].update({
                'log':{
                    'tts': {'x_train': x_train_log, 'x_test' : x_test_log, 'x_train_non' : x_train_non_log, 'x_test_non' : x_test_non_log,
                            'x_train_non_aug' : x_train_non_aug_log, 'x_test_non_aug' : x_test_non_aug_log,'y_train': y_train , 'y_test':y_test,
                            'x':x_log,'x_non':x_non_log,'x_non_aug':x_non_aug_log, 'y':y},
                    'model': {}},
                'ihs':{
                    'tts': {'x_train': x_train_ihs, 'x_test' : x_test_ihs, 'x_train_non' : x_train_non_ihs, 'x_test_non' : x_test_non_ihs,
                            'x_train_non_aug' : x_train_non_aug_ihs, 'x_test_non_aug' : x_test_non_aug_ihs,'y_train': y_train , 'y_test':y_test,
                            'x':x_ihs,'x_non':x_non_ihs,'x_non_aug':x_non_aug_ihs, 'y':y},
                    'model': {}}})
        return self

    def Yeo_Johnson_T(self):
        for app_name, tts in self.train_test_apps_split.items():
            print(f'Yeo Johnson on {app_name}  ... ')
            y = tts['y']
            x = tts['x']
            x_train = tts['x_train']
            x_test = tts['x_test']
            y_train = tts['y_train']
            y_test = tts['y_test']
            
            # Feature_Engine ONLY takes DataFrames
            yeo = YeoJohnsonTransformer()
            x_yeo = pd.DataFrame(yeo.fit_transform(x), columns = x.columns, index = x.index)
            x_train_yeo = pd.DataFrame(yeo.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
            x_test_yeo = pd.DataFrame(yeo.transform(x_test), columns = x_test.columns, index = x_test.index)
            x_non_yeo, _ = self.transform_nonzero_with_fitting(x, None,YeoJohnsonTransformer)
            x_train_non_yeo, x_test_non_yeo = self.transform_nonzero_with_fitting(x_train, x_test,YeoJohnsonTransformer)
            x_non_aug_yeo, _ = self.create_binary_usage_columns(x_non_yeo,None)
            x_train_non_aug_yeo, x_test_non_aug_yeo = self.create_binary_usage_columns(x_train_non_yeo,x_test_non_yeo)
            print(f'Yeo Johnson non-zero & binary cols: {app_name} ... ')
            self.apps_split[app_name].update({
                'yeo':{
                    'tts': {'x_train': x_train_yeo, 'x_test' : x_test_yeo, 'x_train_non' : x_train_non_yeo, 'x_test_non' : x_test_non_yeo,
                            'x_train_non_aug' : x_train_non_aug_yeo, 'x_test_non_aug' : x_test_non_aug_yeo,'y_train': y_train , 'y_test':y_test,
                            'x':x_yeo, 'x_non':x_non_yeo,'x_non_aug':x_non_aug_yeo,'y':y},
                    'model': {}}})
        return self

    def Power_T(self):
        for app_name, tts in self.train_test_apps_split.items():
            print(f'Power on {app_name}  ... ')
            y = tts['y']
            x = tts['x']
            x_train = tts['x_train']
            x_test = tts['x_test']
            y_train = tts['y_train']
            y_test = tts['y_test']
            
            # Feature_Engine ONLY takes DataFrames
            power = PowerTransformer()
            x_power = pd.DataFrame(power.fit_transform(x), columns = x.columns, index = x.index)
            x_train_power = pd.DataFrame(power.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
            x_test_power =  pd.DataFrame(power.transform(x_test), columns = x_test.columns, index = x_test.index)
            
            x_non_power, _ = self.transform_nonzero_with_fitting(x, None, PowerTransformer)
            x_train_non_power, x_test_non_power = self.transform_nonzero_with_fitting(x_train, x_test, PowerTransformer)

            x_non_aug_power, _ = self.create_binary_usage_columns(x_non_power, None)
            x_train_non_aug_power, x_test_non_aug_power = self.create_binary_usage_columns(x_train_non_power, x_test_non_power)
            print(f'Power non-zero & binary cols: {app_name} ... ')
            self.apps_split[app_name].update({
                    'power':{
                        'tts': {'x_train': x_train_power, 'x_test' : x_test_power, 'x_train_non' : x_train_non_power, 'x_test_non' : x_test_non_power,
                                 'x_train_non_aug':x_train_non_aug_power,'x_test_non_aug':x_test_non_aug_power,'y_train': y_train , 'y_test':y_test,
                                 'x':x_power, 'x_non':x_non_power,'x_non_aug':x_non_aug_power, 'y':y},
                        'model': {}}})
        return self

    def Box_Cox_T(self):
        # Feature_Engine ONLY takes DataFrames on NON-ZERO data
        self.boxcox = BoxCoxTransformer()
        for app_name, tts in self.train_test_apps_split.items():
            print(f'Box-Cox on {app_name}  ... ')
            y = tts['y']
            x = tts['x']
            x_train = tts['x_train']
            x_test = tts['x_test']
            y_train = tts['y_train']
            y_test = tts['y_test']

            x_non_boxcox, _ = self.transform_nonzero_with_fitting(x, None,BoxCoxTransformer)
            x_train_non_boxcox, x_test_non_boxcox = self.transform_nonzero_with_fitting(x_train, x_test,BoxCoxTransformer)

            x_non_aug_boxcox, _ = self.create_binary_usage_columns(x_non_boxcox, None)
            x_train_non_aug_boxcox, x_test_non_aug_boxcox = self.create_binary_usage_columns(x_train_non_boxcox, x_test_non_boxcox)
            print(f'Box-Cox non-zero & binary cols: {app_name} ... ')
            self.apps_split[app_name].update({
                    'boxcox':{
                        'tts': { 'x_train': None , 'x_test' : None , 'x_train_non': x_train_non_boxcox, 'x_test_non' : x_test_non_boxcox,
                                 'x_train_non_aug':x_train_non_aug_boxcox,'x_test_non_aug':x_test_non_aug_boxcox,'y_train': y_train , 'y_test':y_test,
                                 'x_non':x_non_boxcox,'x_non_aug':x_non_aug_boxcox, 'y':y},
                        'model': {}}})
        return self

    def run(self):
        self.Winsorization_T()
        self.Log_SquareRoot_T()
        self.Yeo_Johnson_T()
        self.Box_Cox_T()
        return self

#_____________________________________________________________________________________________________________________________________________________

class model_evaluation(transform_methods):
    def __init__(self,apps_split):
        self.apps_split = apps_split
        self.results_table = pd.DataFrame()
        self.combo_tts_dict = {}
        self.top_combinations_table = pd.DataFrame()
        self.top_combination_tts = {}
        
    def evaluate(self):
        """Calculate metrics and store results in results_table"""
        print('Accurring Results Table ...')
            
        for app_name, transformer_dict in self.apps_split.items(): 
            for transformer_name, contents in transformer_dict.items():
                suffix_list = ['', '_non','_non_aug']
                
                for suffix in suffix_list:
                    if transformer_name == 'boxcox' and suffix == '':
                        suffix_list = ['_non','_non_aug']
                        continue 
                    else:
                        tts = contents['tts']
                        Y_test = tts['y_test']
                        Y_train = tts['y_train']
                        X_train = tts[f'x_train{suffix}']
                        X_test = tts[f'x_test{suffix}']
                        y = tts['y']
                        x = tts[f'x{suffix}']
                        combo_tts_dict = pd.DataFrame()
                        
                        for model, model_dict in contents['model'].items():
                            for Y_pred_suffix, y_pred_values in model_dict['y_preds'].items():
                                Y_pred = y_pred_values
                                combo_key = f'{app_name}_{model}_{transformer_name}{suffix}'
                                
                               # Recall, Precision, F2 Score, PR AUC
                                recall_model = round(100 * recall_score(Y_test, Y_pred), 2)
                                f2 = round(100 * fbeta_score(Y_test, Y_pred, beta = 2), 2)
                                precision = round(100 * precision_score(Y_test, Y_pred), 2)
                                PR_AUC = round(100 * average_precision_score(Y_test, Y_pred), 2)
                                    
                                # Percentage of False Negatives
                                y_pred_flat = np.ravel(Y_pred)
                                tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_flat).flatten()
                                recall = round((tp / (tp + fn))*100,2)
                                percent_fn = round(((fn/(tn+tp))*100),2)
        
                                # Averaging Scores
                                avg_scores = round((recall_model+f2+precision+PR_AUC)/4, 2)
                                            
                                # Results Table
                                new_row = pd.DataFrame({
                                    'App_name': [f'{app_name}'],
                                    'transformer': [f'{transformer_name}'],
                                    'Model': [f'{model}'],
                                    'Y-Pred Type': [f'{suffix}'],
                                    'Combo_key': [combo_key],
                                    'PR_AUC': [PR_AUC],
                                    'F2 Score': [f2],
                                    'Precision': [precision],
                                    'Recall': [recall_model],
                                    'avg_scores': [avg_scores],
                                    'Percent_fn': [percent_fn]})
                                
                                tts_dict = {
                                    'x_train':X_train,
                                    'x_test': X_test,
                                    'y_train': Y_train, 
                                    'y_test': Y_test, 
                                    'x': x,
                                    'y':y}
             
                                # Append the new row
                                self.results_table = pd.concat([self.results_table, new_row])
                                self.combo_tts_dict[combo_key] = tts_dict
        
        return self
        
    def top_results(self):
        # Top Combos
        print('Gathering Top Combinations...')
        # Find top Combinations
        top_combo = self.results_table.copy().drop(columns = ['PR_AUC', 'F2 Score', 'Precision','Recall'])
        top_combo = top_combo.groupby(['App_name','Model']).apply(lambda x: x.sort_values(by = ['avg_scores','Percent_fn'], ascending = [False,True]).head(1))
        self.top_combinations_table = top_combo.reset_index(drop=True)        
        # Get the Train/Test sets of Top Combinations
        print('Gathering Top Combinations Train / Test List...')
        c_tts = self.combo_tts_dict.copy()
        combo_name_list = top_combo['Combo_key'].tolist()
        self.top_combination_tts = {key:value for key, value in c_tts.items() if key in combo_name_list}
        return self
        
class XGB_combination_fitting(model_evaluation):  
    def __init__(self, apps_split):
        self.apps_split = apps_split
        
    def XGB_combination_fit(self):
        for app_name, transformer_dict in self.apps_split.items(): 
            for transformer_name, contents in transformer_dict.items():
                print(f'Fitting XGBoost on {app_name}_{transformer_name} ...')
                # stores the y_preds
                preds = {}
                tts = contents['tts'] 
                suffix_list = ['', '_non','_non_aug']
                
                for suffix in suffix_list:
                    if transformer_name == 'boxcox' and suffix == '':
                        suffix_list = ['_non','_non_aug']
                        continue 
                    else:
                            X_train = tts[f'x_train{suffix}']
                            X_test = tts[f'x_test{suffix}']
                            Y_train = tts['y_train']
                            Y_test  = tts['y_test']
                    
                            # Define the parameters for XGBoost Classifier
                            params = {
                            'objective': 'binary:logistic',
                            'base_score': 0.5,
                            'eta': 0.1,
                            'max_depth': 5,
                            'subsample': 0.7,
                            'colsample_bytree': 0.7,
                            'gamma': 0.1,
                            'lambda': 1,
                            'alpha': 0,
                            'eval_metric': 'logloss',
                            'tree_method': 'hist'
                                        }
                            # Initialize XGBoost Classifier with parameters
                            XGB = xgb.XGBClassifier(**params, n_estimators=100)
                        
                            # Train the Model
                            XGB_model = XGB.fit(X_train, Y_train)
                            Y_pred = XGB_model.predict(X_test).reshape(-1, 1)
                            preds[f'Y_pred{suffix}'] = Y_pred
    
                # Update the ANN y_preds dictionary
                self.apps_split[app_name][transformer_name]['model'].update({'XGB': {'y_preds': preds}})
        return self


class ANN_combination_fitting(model_evaluation):    
    def __init__(self, apps_split):
        self.apps_split = apps_split
        
    def ANN_combination_fit(self):
        for app_name, transformer_dict in self.apps_split.items(): 
            for transformer_name, contents in transformer_dict.items():
                print(f'Fitting ANN on {app_name}_{transformer_name} ...')
                # stores the y_preds
                preds = {}
                tts = contents['tts'] 
                suffix_list = ['', '_non','_non_aug']
                
                for suffix in suffix_list:
                    if transformer_name == 'boxcox' and suffix == '':
                        suffix_list = ['_non','_non_aug']
                        continue 
                    else:
                            X_train = tts[f'x_train{suffix}']
                            X_test = tts[f'x_test{suffix}']
                            Y_train = tts['y_train']
                            Y_test  = tts['y_test']
    
                            #  --------------------- Initializing Ann  --------------------------
                            ANN = tf.keras.models.Sequential([
                                layers.Dense(units = 6, activation = 'relu'),
                                layers.Dense(units = 6, activation = 'relu'),
                                layers.Dense(units = 1, activation = 'sigmoid')
                            ])
                            ANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])
                                
                            # Train the model
                            ANN.fit(X_train, Y_train, batch_size = 32, epochs = 10, verbose = 0)
                            Y_pred = ANN.predict(X_test)
                            Y_pred = (Y_pred > 0.5) 
                            preds[f'Y_pred{suffix}'] = Y_pred

                # Update the ANN y_preds dictionary
                self.apps_split[app_name][transformer_name]['model'].update({'ANN': {'y_preds': preds}})
        return self
# %%  -------------------------------------------------- class: FEATURE ANALYSIS  --------------------------------------------------

class feature_analysis(model_evaluation):
    def __init__(self, top_combination_tts):
        self.top_combination_tts = top_combination_tts
        self.shap_df = pd.DataFrame()
        self.shap_daily_values = {}

    def XGB_build(self):
        params = {
                            'objective': 'binary:logistic',
                            'base_score': 0.5,
                            'eta': 0.1,
                            'max_depth': 5,
                            'subsample': 0.7,
                            'colsample_bytree': 0.7,
                            'gamma': 0.1,
                            'lambda': 1,
                            'alpha': 0,
                            'eval_metric': 'logloss',
                            'tree_method': 'hist'
                                        }
        # Initialize XGBoost Classifier with parameters
        XGB = xgb.XGBClassifier(**params, n_estimators=100)
        return XGB

    def build_ANN_model(self):
        #  --------------------- Initializing Ann  --------------------------
        ANN = tf.keras.models.Sequential()
        ANN.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
        ANN.add(tf.keras.layers.Dense(units = 6, activation = 'relu'))
        ANN.add(tf.keras.layers.Dense(units = 1, activation = 'sigmoid'))
        ANN.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics =['accuracy'])
        return ANN

    def feature_results(self):
        ann_df = []
        usage_type = ['usage_type_usage_pack_data','usage_type_usage_pack_vas','usage_type_usage_voice_d2d_incoming',
                      'usage_type_usage_voice_d2d_outgoing','usage_type_usage_voice_d2nd_outgoing','usage_type_usage_voice_nd2d_incoming']

        for combo_key, combo_items in self.top_combination_tts.items():
            combo_key_name = combo_key

            # churners TTS
            x_sample = combo_items['x'].reset_index(drop=True)
            x_sample = x_sample.astype(np.float32)
            y_sample = combo_items['y'].reset_index(drop=True)
            y_sample = y_sample.apply(pd.to_numeric, errors = 'coerce').fillna(0)
            y_sample_churn = y_sample[y_sample == 1].reset_index(drop=True).astype(np.float32)
            y_sample_nonchurn = y_sample[y_sample == 0].reset_index(drop=True).astype(np.float32)
            x_sample_churn = x_sample[y_sample == 1].reset_index(drop=True).astype(np.float32)
            x_sample_nonchurn = x_sample[y_sample == 0].reset_index(drop=True).astype(np.float32)
            
            last_day_col = len(x_sample.filter(regex = '^Day_*').columns)
            days_col = [f'Day_{i}' for i in range(1,last_day_col+1)]
            aug_days_col = [f'aug_Day_{i}' for i in range(1,last_day_col+1)]

            for X,Y in [(x_sample_churn,y_sample_churn)]:
                if X is x_sample_churn:
                    churn_non = 'Churners'
                else:
                    churn_non = 'Non-Churners'

                
                if not (X.filter(regex = 'aug').columns).empty: 
                    sub_group_list = [usage_type, days_col, aug_days_col]
                else:
                    sub_group_list = [usage_type, days_col]
                        
                for sub_group in sub_group_list:
                    if sub_group == usage_type:
                        subgroup = 'Usage Type'
                    elif sub_group == days_col:
                        subgroup = 'Days Column'
                    elif sub_group == aug_days_col:
                        subgroup = 'Binary Days Column' 
                        
                    print('Working on: ', combo_key_name, '\t\t', subgroup)
                    
                    if 'ann' in combo_key.lower() :
                        # Use Stratified Sample as the background distribution
                        X_train_SHAP,X_test_SHAP,Y_train_SHAP,Y_test_SHAP = tts(X[sub_group],Y,train_size = 50,stratify = Y,random_state = 42)

                        feature_name = X_test_SHAP.columns.tolist()
                        # Summariaze X_train to K samples
                        background_data = shap.sample(X_train_SHAP, 50)
                        
                        model = self.build_ANN_model()
                        model.fit(X_train_SHAP,Y_train_SHAP, epochs = 50, verbose = 0)
                        explainer = shap.KernelExplainer(model.predict, background_data)
                                         
                    else:
                        # Use Stratified Sample as the background distribution
                        X_train_SHAP,X_test_SHAP,Y_train_SHAP,Y_test_SHAP = tts(x_sample[sub_group],y_sample,train_size = 50,stratify = y_sample,random_state = 42)
                        feature_name = X_test_SHAP.columns.tolist()
                        for i in feature_name:
                            X_train_SHAP[i] = X_train_SHAP[i]
                            X_test_SHAP[i] = X_test_SHAP[i]
                        Y_train_SHAP = Y_train_SHAP
                        Y_test_SHAP = Y_test_SHAP
                        model = xgb.XGBClassifier(n_estimators = 100,
                                                 base_score = 0.5,
                                                 tree_method = 'hist',
                                                 random_state = 4)
                        print(model.get_xgb_params())
                        model.fit(X_train_SHAP,Y_train_SHAP)
                        explainer = shap.TreeExplainer(model)

                    # Test splits
                    X_test_sample = shap.sample(X[sub_group],50)
                    X_test_sample = X_test_sample.apply(pd.to_numeric, errors = 'coerce').fillna(0)
                        
                    # Calculate SHAP values for the entire dataset
                    shap_values = explainer.shap_values(X_test_sample)
                    shap_values = list(map(lambda x: abs(x), shap_values))
                    shap_values = np.mean(shap_values, axis = 0)
                    pairs = [(f,v) for f,v in zip(feature_name, shap_values)]
                    zipped_pairs = list(zip(feature_name, shap_values))

                    sort_pairs = sorted(pairs, key=lambda x: x[1], reverse = True)
                    sort_pairs = sort_pairs[:3]
                    top_feature_name = [name for name,val in sort_pairs]
                    top_feature_values = [val for name,val in sort_pairs]
                    app_name = None
                    model_name = None
                    if 'app_other' in combo_key.lower():
                        app_name = 'Other Apps'
                    elif 'youtube' in combo_key.lower():
                        app_name = 'YouTube'
                    elif 'facebook' in combo_key.lower():
                        app_name = 'Facebook'
                    elif 'tiktok' in combo_key.lower():
                        app_name = 'TikTok'
                    elif 'whatsapp' in combo_key.lower():
                        app_name = 'WhatsApp'
                    elif 'helakuru' in combo_key.lower():
                        app_name = 'Helakuru'

                    if 'ann' in combo_key.lower():
                        model_name = 'ANN'
                    elif 'xgb' in combo_key.lower():
                        model_name = 'XGB'
    
                    shap_new_row = pd.DataFrame({
                                        'App Name': [app_name],
                                        'Model Name': [model_name],
                                        'churners / Non':[churn_non],
                                        'feature_name':[top_feature_name],
                                        'shap_values': [top_feature_values]
                                        })
                    
                    self.shap_df = pd.concat([self.shap_df,shap_new_row], ignore_index = True)   
                    shap_daily_values_df = pd.DataFrame(zipped_pairs, columns = ['Days', 'SHAP Values'])
                    shap_daily_values_df = shap_daily_values_df.sort_values(by = ['SHAP Values'], ascending = [False]).head(10)
            self.shap_daily_values.setdefault(app_name,{}).setdefault(model_name,{})[churn_non] = shap_daily_values_df
        return self.shap_df


# %%  ------------------------------------------------ initiating: PREPING DATAFRAMES  --------------------------------------------------------
DF = pd.read_csv(r'other\Real_World_Customer_Churn_dataset RAW.csv')
df = pd.DataFrame(DF)
prep_df = Preping_df(df).run()

# ## Transform Apps Train / Test sets
trans = transform_methods(prep_df.dataframe, prep_df.train_test_apps_split).run()
# %%  ------------------------------------------- initiating: TRANSFORMING / FITTING DATAFRAMES --------------------------------------------
# ## Modeling / Evaluating 
XGB_model = XGB_combination_fitting(trans.apps_split).XGB_combination_fit()
ann_model = ANN_combination_fitting(trans.apps_split).ANN_combination_fit()
models_evaluation = model_evaluation(trans.apps_split)
models_evaluation.evaluate()
models_evaluation.top_results()
result_table = models_evaluation.results_table
top_combo_table = models_evaluation.top_combinations_table
combo_dict = models_evaluation.combo_tts_dict
top_combo_dict = models_evaluation.top_combination_tts
print(top_combo_table.drop(['Combo_key','Y-Pred Type'], axis = 1))
# %%  --------------------------------------------------- initiating: FEATURE ANALYSIS  ---------------------------------------------------------------
# shap
Feature_Analysis = feature_analysis(top_combo_dict)
Feature_Analysis.XGB_build()
Feature_Analysis.build_ANN_model()
Feature_Analysis.feature_results()          
# %%  ---------------------------------------------  GRAB FEATURE ANALYSIS DATAFRAME / DICTS -------------------------------------------
SHAP_DF = Feature_Analysis.shap_df
dropped_shap_df = SHAP_DF.drop(index = SHAP_DF.index[::2])
final_shap_df = dropped_shap_df.drop('shap_values', axis =1)
shap_daily_values = Feature_Analysis.shap_daily_values                                        # SHAP DAILY VALUES
print('Grabbed Dataframes / Dicts')


# %%  ------------------------------------------------------ SAVE CSV / DICTIONARY FILES  ---------------------------------------------------------
# SAVE DICTIONARY shap daily values
with open(r'files/shap_daily_values.pk1', 'wb') as f: 
    pickle.dump(shap_daily_values,f)

final_shap_df.to_csv(r'outputs/final_shap_df.csv', index = False)
print(final_shap_df)
print('Saved final shap dataframe \n\n')

# ONLY churners / nonchurners SHAP DF 
ANN_shap = final_shap_df[final_shap_df['Model Name'].str.contains('ANN')] 
print('WHOLE ann shap: \n ',ANN_shap,' \n\n')
ANN_shap_churners = ANN_shap.loc[ANN_shap['churners / Non'] == 'Churners']
print('CHURNERS ann shap: \n ',ANN_shap_churners,' \n\n')
ANN_shap_churners = ANN_shap_churners.drop(columns=['Model Name','churners / Non'], axis=1)
print('ANN_shap_churners \n', ANN_shap_churners)  
ANN_shap_churners.to_csv(r'outputs/ANN_shap_values.csv', index = False)
# %%
