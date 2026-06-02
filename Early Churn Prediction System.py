# Generated from: Early Churn Prediction System.ipynb
# Converted at: 2026-04-03T02:18:16.718Z
# Next step (optional): refactor into modules & generate tests with RunCell
# Quick start: pip install runcell

# %%  ----------------------------------------------------- Packages  ----------------------------------------------
# ## Pipeline
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
import pandas as pd
import numpy as np
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
from sklearn.linear_model import LogisticRegression
from scikeras.wrappers import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import precision_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
import pickle
import os

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
# %% ------------------------------------------------ Preping Model  -----------------------------------------
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
        self.categorical_col = ['usage_type']
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
            app_df['churn'] = ((app_df['last_active_day']) >= inactivity_window).astype(int)
        return self

    def downsample_subset_df(self):
        '''
         The dataframe is too large for the CPU to handle, thus we must downsample it. 
         For the Model to not pick churner each time, we must match the same amount of churners vs. nonchurners for the model to actually learn
        '''
        print('Downsample each app\'s datasets.')
        for app_name, app_df in self.apps.items():
            print(f'{app_name} Df Shape:', app_df.shape )
             # Split the Df in Minority / Majority 
            df_minority = app_df[app_df['churn'] == 0] 
            df_majority = app_df[app_df['churn'] == 1]
            print(f'Number of churners {len(df_majority)}')
            print(f'Number of NON churners: {len(df_minority)}')
    
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
            x = x[[f'Day_{i}' for i in range(1, self.last_day_col +1)] + ['usage_type']]
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
            x = x.drop(columns = remove_col, axis = 1)
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
    def rename_key_names(self):
        name_map = {
                'app_other':'Other Apps',
                'youtube':'YouTube',
                'facebook': 'Facebook',
                'tiktok': 'TikTok',
                'whatsapp': 'WhatsApp',
                'helakuru': 'Helakuru'
            }
        self.apps = {
                name_map.get(k,k):v for k,v in self.apps.items()
            }
        self.XY_apps = {
                name_map.get(k,k):v for k,v in self.XY_apps.items()
            }
        self.train_test_apps_split = {
                name_map.get(k,k):v for k,v in self.train_test_apps_split.items()
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
        self.rename_key_names()
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
            x_train = tts['x_train']
            # list the columns
            encoded_cols = list(x_train.columns)
            
            # keep only numeric columns with >2 unique values for winsorization
            numeric_cols_for_winsor = [col for col in encoded_cols if x_train[col].nunique() > 2 ]
            
            x_train = x_train[numeric_cols_for_winsor].copy()
            x_test = tts['x_test'][numeric_cols_for_winsor].copy()
            y_train = tts['y_train']
            y_test = tts['y_test']
 
            winsor_MAD = Winsorizer(capping_method='quantiles', 
                                    tail='both',
                                    fold=0.02,
                                    variables = x_train.columns.tolist()
                                   )

            x_train_mad = winsor_MAD.fit_transform(x_train)
            x_test_mad = winsor_MAD.transform(x_test)
            x_train_non_mad, x_test_non_mad = self.transform_nonzero_with_fitting(x_train,x_test,Winsorizer,capping_method='quantiles',tail='both',fold=0.02)
            print(f'Non-Binary Winsorization: IQR and MAD for {app_name}... ')
            x_train_non_aug_mad, x_test_non_aug_mad = self.create_binary_usage_columns(x_train_non_mad,x_test_non_mad)
            print(f'Adding Binary Cols for: {app_name}... ')
            
            self.apps_split[app_name] = {
                'mad':{
                    'tts': { 
                        'x_train': x_train_mad,'x_test' : x_test_mad, 'x_train_non': x_train_non_mad, 'x_test_non' : x_test_non_mad,
                        'x_train_non_aug' :x_train_non_aug_mad, 'x_test_non_aug' : x_test_non_aug_mad,'y_train': y_train , 'y_test':y_test},
                    'model': {}}}
        return self 

    def Log_SquareRoot_T(self):
        for app_name, tts in self.train_test_apps_split.items():
            print(f'LogCP and Inverse Hyperbolicsine: {app_name} ... ')
            x = tts['x']
            x_train = tts['x_train'].copy()
            x_test = tts['x_test'].copy()
            y_train = tts['y_train']
            y_test = tts['y_test']
            
            class ArcsinhTransformer: 
                def fit(self, X):
                    return self
                def transform(self, X):
                    return np.arcsinh(X)
                def fit_transform(self,X):
                    return self.transform(X)

            # Inverse Hyperbolic sine
            x_train_ihs = pd.DataFrame(np.arcsinh(x_train), columns = x_train.columns, index = x_train.index)
            x_test_ihs = pd.DataFrame(np.arcsinh(x_test), columns = x_test.columns, index = x_test.index)
            
            x_train_non_ihs, x_test_non_ihs = self.transform_nonzero_with_fitting(x_train, x_test, ArcsinhTransformer)

            x_train_non_aug_ihs, x_test_non_aug_ihs = self.create_binary_usage_columns(x_train_non_ihs,x_test_non_ihs)
            print(f'Hyperbolic non-zero & binary cols: {app_name} ... ')
            self.apps_split[app_name].update({
                'ihs':{
                    'tts': {'x_train': x_train_ihs, 'x_test' : x_test_ihs, 'x_train_non' : x_train_non_ihs, 'x_test_non' : x_test_non_ihs,
                            'x_train_non_aug' : x_train_non_aug_ihs, 'x_test_non_aug' : x_test_non_aug_ihs,'y_train': y_train , 'y_test':y_test},
                    'model': {}}})
        return self

    def Yeo_Johnson_T(self):
        for app_name, tts in self.train_test_apps_split.items():
            print(f'Yeo Johnson on {app_name}  ... ')
            x_train = tts['x_train'].copy()
            x_test = tts['x_test'].copy()
            y_train = tts['y_train']
            y_test = tts['y_test']
            
            # Feature_Engine ONLY takes DataFrames
            yeo = YeoJohnsonTransformer()
            x_train_yeo = pd.DataFrame(yeo.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
            x_test_yeo = pd.DataFrame(yeo.transform(x_test), columns = x_test.columns, index = x_test.index)
            x_train_non_yeo, x_test_non_yeo = self.transform_nonzero_with_fitting(x_train, x_test,YeoJohnsonTransformer)
            x_train_non_aug_yeo, x_test_non_aug_yeo = self.create_binary_usage_columns(x_train_non_yeo,x_test_non_yeo)
            print(f'Yeo Johnson non-zero & binary cols: {app_name} ... ')
            self.apps_split[app_name].update({
                'yeo':{
                    'tts': {'x_train': x_train_yeo, 'x_test' : x_test_yeo, 'x_train_non' : x_train_non_yeo, 'x_test_non' : x_test_non_yeo,
                            'x_train_non_aug' : x_train_non_aug_yeo, 'x_test_non_aug' : x_test_non_aug_yeo,'y_train': y_train , 'y_test':y_test},
                    'model': {}}})
        return self

    def Power_T(self):
        for app_name, tts in self.train_test_apps_split.items():
            print(f'Power on {app_name}  ... ')
            x_train = tts['x_train'].copy()
            x_test = tts['x_test'].copy()
            y_train = tts['y_train']
            y_test = tts['y_test']
            
            # Feature_Engine ONLY takes DataFrames
            power = PowerTransformer()
            x_train_power = pd.DataFrame(power.fit_transform(x_train), columns = x_train.columns, index = x_train.index)
            x_test_power =  pd.DataFrame(power.transform(x_test), columns = x_test.columns, index = x_test.index)
            x_train_non_power, x_test_non_power = self.transform_nonzero_with_fitting(x_train, x_test, PowerTransformer)
            x_train_non_aug_power, x_test_non_aug_power = self.create_binary_usage_columns(x_train_non_power, x_test_non_power)
            print(f'Power non-zero & binary cols: {app_name} ... ')
            self.apps_split[app_name].update({
                    'power':{
                        'tts': {'x_train': x_train_power, 'x_test' : x_test_power, 'x_train_non' : x_train_non_power, 'x_test_non' : x_test_non_power,
                                 'x_train_non_aug':x_train_non_aug_power,'x_test_non_aug':x_test_non_aug_power,'y_train': y_train , 'y_test':y_test},
                        'model': {}}})
        return self

    def Box_Cox_T(self):
        # Feature_Engine ONLY takes DataFrames on NON-ZERO data
        self.boxcox = BoxCoxTransformer()
        for app_name, tts in self.train_test_apps_split.items():
            print(f'Box-Cox on {app_name}  ... ')
            x_train = tts['x_train'].copy()
            x_test = tts['x_test'].copy()
            y_train = tts['y_train']
            y_test = tts['y_test']

            x_train_non_boxcox, x_test_non_boxcox = self.transform_nonzero_with_fitting(x_train, x_test,BoxCoxTransformer)
            x_train_non_aug_boxcox, x_test_non_aug_boxcox = self.create_binary_usage_columns(x_train_non_boxcox, x_test_non_boxcox)
            print(f'Box-Cox non-zero & binary cols: {app_name} ... ')
            self.apps_split[app_name].update({
                    'boxcox':{
                        'tts': { 'x_train': None , 'x_test' : None , 'x_train_non': x_train_non_boxcox, 'x_test_non' : x_test_non_boxcox,
                                 'x_train_non_aug':x_train_non_aug_boxcox,'x_test_non_aug':x_test_non_aug_boxcox,'y_train': y_train , 'y_test':y_test},
                        'model': {}}})
        return self

    def run(self):
        self.Winsorization_T()
        self.Log_SquareRoot_T()
        self.Yeo_Johnson_T()
        self.Box_Cox_T()
        return self

#  %%_________________________________________________________________  Model Evaluation  ____________________________________________________________________________________

class model_evaluation(transform_methods):
    def __init__(self,apps_split):
        self.apps_split = apps_split
        self.results_table = pd.DataFrame()
        self.top_combinations_table = pd.DataFrame()
        self.top_combination_tts = {}

    def XGB_model(self):
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

    def Elastic_Net_model(self):
        #  Define the parametrs for Log Regression Elastic Net 
        params = {
                    'penalty' : 'elasticnet',
                    'solver' : 'saga',
                    'l1_ratio' : 0.5, # mixing L1 and L2 penalty ( L1 = 0, L2 = 1)
                    'C': 1.0,
                    'max_iter' : 800,
                    'random_state' : 42
                }
        elastic_net = LogisticRegression(**params)
        return elastic_net
        
    def evaluate(self):
        """ Calculating Tradesoffs of lowering the threshold on the model to achieve optimal False Negative Scores.
            To gauge how well our models are catching churners while lowering the threshold we will be using: 
                - Recall
                - Precision 
                - F2 Score
                - PR AUC
            Reasons: 
                : Directly measures how many churners are catched 
                : Tells how many flagged users are actually churners
                : Weight recall focused (betta > 1)
                : Focuses on positive class (churn)
            Won't be using these Units: 
                - Accuracy 
                - F1 Score
                - ROC AUC
            Reasons: 
                : misleading for churn
                : is used for a balanced threshold (p > 0.5)
                : 
        
        """
        print('Accurring Results Table ...')
        range_window=range(1,15)
            
        for app_name, transformer_dict in self.apps_split.items(): 
            print(app_name)
            for transformer_name, contents in transformer_dict.items():
                print(transformer_name)
                tts = contents['tts']
                suffix_list = ['_non'] if transformer_name == 'boxcox' else ['']
                for suffix in suffix_list:
                    for day_range in range_window: 
                        interval_days = [f'Day_{i}' for i in range(1, day_range+1)]
                        X_train = tts[f'x_train{suffix}'][interval_days]
                        X_test = tts[f'x_test{suffix}'][interval_days]
                        Y_test = tts['y_test']
                        Y_train = tts['y_train']  
                        # Lower False Negative with a tradeoff of upping Flase Positives
                        threshold_list = [0.1,0.2,0.3,0.4,0.5]
                        for threshold in threshold_list:

                            # train XGB Model
                            xgb = self.XGB_model()
                            xgb_model = xgb.fit(X_train,Y_train)
                            Y_pred_XGB = xgb_model.predict(X_test)
                            y_praba_XGB = xgb_model.predict_proba(X_test)[:,1]
                            Y_pred_XGB = (y_praba_XGB>=threshold).astype(int)

                            # train Elastic Net Model
                            elastic_net = self.Elastic_Net_model()
                            elastic_net.fit(X_train,Y_train)
                            Y_pred_Elastic_Net = elastic_net.predict(X_test)
                            y_praba_E = elastic_net.predict_proba(X_test)[:,1]
                            Y_pred_Elastic_Net = (y_praba_E>=threshold).astype(int)

                            for model_name, Y_pred, Y_praba in (('XGB', Y_pred_XGB,y_praba_XGB), ('Elastic_Net',Y_pred_Elastic_Net,y_praba_E)):                                    
                                # Recall, Precision, F2 Score, PR AUC
                                recall_model = round(100 * recall_score(Y_test, Y_pred), 2)
                                f2 = round(100 * fbeta_score(Y_test, Y_pred, beta = 2), 2)
                                precision = round(100 * precision_score(Y_test, Y_pred), 2)
                                PR_AUC = round(100 * average_precision_score(Y_test, Y_praba), 2)
                                
                                # Percentage of False Negatives
                                y_pred_flat = np.ravel(Y_pred)
                                tn, fp, fn, tp = confusion_matrix(Y_test, y_pred_flat).flatten()
                                recall = 0 if (tp + fn) ==0 else round((tp / (tp + fn))*100,2)
                                percent_fn = 0 if (fn+tp) == 0 else round(((fn/(fn+tp))*100),2)
                                                        
                                # Results Table
                                new_row = pd.DataFrame({
                                                'App_name': [f'{app_name}'],
                                                'transformer': [f'{transformer_name}'],
                                                'Model': [f'{model_name}'],
                                                'Threshold': [threshold],
                                                'Days': [day_range],
                                                'recall': [recall_model],
                                                'F2 Score': [f2],
                                                'Precision': [precision], 
                                                'PR AUC': [PR_AUC],
                                                'False Negatives': [percent_fn]})
                                # Append the new row
                                self.results_table = pd.concat([self.results_table, new_row])
        print('...✅')
        return self
        
    def top_results(self):
        # Top Combos
        print('Gathering Top Combinations...')
        modify_results_table = self.results_table.copy()
        
        top_combo = modify_results_table.groupby(['App_name','Model','Days','Threshold']).apply(lambda x: x.sort_values(by = ['False Negatives','PR AUC','recall'], ascending = [True,False,False]).head(1))
        top_combo = top_combo.reset_index(drop=True)

        # Merge the two top tables
        self.top_combinations_table = top_combo
        print('...✅')
        return self

    def run(self):
        self.XGB_model()
        self.Elastic_Net_model()
        self.evaluate()
        return self

# ### Preping dataframe

# %%  ---------------------------------------  INITIATING: DATA PREPPING / TRANSFORMATION --------------------------------------------
DF = pd.read_csv(r'files/Real_World_Customer_Churn_dataset RAW.csv')
df = pd.DataFrame(DF)
prep_df = Preping_df(df).run()
trans = transform_methods(prep_df.dataframe, prep_df.train_test_apps_split).run()
apps_split = trans.apps_split
#print(apps_split.items())
# %%
# -------------------------------------------- INITIATING: Evaluating Models  --------------------------------------------------
model_evaluation = model_evaluation(apps_split).run()
model_evaluation.top_results()
result_tab = model_evaluation.results_table
top_combinations_table = model_evaluation.top_combinations_table
print(top_combinations_table)


top_combinations_table.to_csv('outputs/early_churn_prediciton_table.csv', index = False)
# %%
