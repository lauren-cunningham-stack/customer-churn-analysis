# %% Class Preping the dataframe
import statistics as s
from sklearn.utils import resample
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
import shap
from scipy.stats import skew
from sklearn.inspection import permutation_importance as PI
import pickle
import ast
import re
import pandas as pd
import numpy as np
import matplotlib as plt
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

class PrepingDF:
    def __init__(self,dataframe):
        self.dataframe = dataframe
        self.cutoff_day = None
        self.categorical_col = None
        self.numeric_col = None
        self.day_column = None
        self.cut_off_day_cols = None

        self.apps = {}
        self.churners = {}
        self.nonchurners = {}
        self.cohorts_churners_Day_1 = {}
        self.cohorts_nonchurners_Day_1 = {}
        self.retention_churners_Day_1 = {}
        self.retention_nonchurners_Day_1 = {}
        self.cohort_summary_churners_Day_1 = {}
        self.cohort_summary_nonchurners_Day_1 = {}
        
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
        
        #print("Cleaning dataframe...\n")         

        for app_name, app_df in self.apps.items():              
            # Replace common null-like values with np.nan
            #print(f'Cleaning {app_name} . . .')
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

            remove_rows = [f'usage_type_usage_{app_name}' if app_name == 'app_other' else f'usage_type_usage_app_{app_name}_daily']
            app_df = app_df.drop(app_df[app_df['usage_type'].isin(remove_rows)].index)

            # Update Dictionary 
            self.apps[app_name] = app_df
        print('1/6 Done ✅')
        return self                                          

    def churn_df(self):
        '''
        Create churn labels for each app based on user inactiviy patterns
        
        Churn Definition: 
            - users whose last active day <= cutoff_day
            - Or users inactive for  >= inactivity_window days
        '''
        print("Creating Chrun Col for each App...\n")    
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
        for app_name, app_df in self.apps.items(): 
            self.last_day_col = len(app_df.filter(regex = '^Day_*').columns) # 90
            self.day_column = [f'Day_{i}' for i in range(1, self.last_day_col+1)] # 90
            self.cutoff_day = self.last_day_col - 10 
            inactivity_window = 20
            
            cut_off_extra_day_cols = [f'Day_{i}' for i in range(81,self.last_day_col + 1)] #['Day_81', 'Day_82', 'Day_83', 'Day_84', 'Day_85', 'Day_86', 'Day_87', 'Day_88', 'Day_89'] 
            self.cut_off_day_cols = [f'Day_{i}' for i in range(1,self.cutoff_day+1)]
            self.categorical_col = ['usage_type']
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
        print('2/6 Done ✅')
        return self

    def first_day_activity_level(self):
        '''
        We want to see if there are any correlations on how much usage (hours) in each app for the users on the first day.
        '''
        #print("Creating first_day Col for each App...\n")    
        for app_name, app_df in self.apps.items(): 
            app_df['Day_1'] = pd.to_numeric(app_df['Day_1'], errors = 'coerce')
            #print(f'App Name {app_name}: ',app_df['Day_1'].dtypes)
            conditions = [
                (app_df['Day_1'] == 0), 
                (app_df['Day_1'] > 0) & (app_df['Day_1'] <= 30),
                (app_df['Day_1'] > 30)
            ]
            choices = ['Low','Medium','High']
            app_df['first_day_activity_level'] = np.select(conditions, choices, default = None)
            self.apps[app_name] = app_df  
        print('3/6 Done ✅')
        return self
    
    def separate_apps_churners_nonchurners(self):
        '''
        By separating into churners and nonchurners we could further see the behavior patterns for each of these users to limit churners.
        '''
        print("Creating Churners vs Non-Churner Dataframe for each App...\n")         
        for app_name, app_df in self.apps.items(): 
            # store separated dataframe 
            churners = app_df.loc[app_df['churn'] == 1].copy()
            nonchurners = app_df.loc[app_df['churn'] == 0].copy()

            # store the new separated dataframe into __init__ variables
            self.churners[app_name] = churners
            self.nonchurners[app_name] = nonchurners
        print('4/6 Done ✅')
        return self

    def cohorts_subgroubs(self):
        '''
        The cohorts we will to group would be based on:
            - app
            - churners vs. nonchurners
            - activity levels on the first day
        '''
        
        print('Creating Cohorts of activity levels on the first day for each groups')

        for app_name, churners_app_df in self.churners.items(): 
            # sort out activity level
            high = churners_app_df.loc[churners_app_df['first_day_activity_level'] == 'High']
            medium = churners_app_df.loc[churners_app_df['first_day_activity_level'] == 'Medium']
            low = churners_app_df.loc[churners_app_df['first_day_activity_level'] == 'Low']

            # store new cohorts dataframe back into __init__ variables 
            self.cohorts_churners_Day_1.setdefault(app_name,{})['high'] = high
            self.cohorts_churners_Day_1.setdefault(app_name,{})['medium'] = medium
            self.cohorts_churners_Day_1.setdefault(app_name,{})['low'] = low
            print(app_name)

        for nonchurners_app_name, nonchurners_app_df in self.nonchurners.items(): 
            # sort out activity level
            high = nonchurners_app_df.loc[nonchurners_app_df['first_day_activity_level'] == 'High']
            medium = nonchurners_app_df.loc[nonchurners_app_df['first_day_activity_level'] == 'Medium']
            low = nonchurners_app_df.loc[nonchurners_app_df['first_day_activity_level'] == 'Low']

            # store new cohorts dataframe back into __init__ variables 
            self.cohorts_nonchurners_Day_1.setdefault(nonchurners_app_name,{})['high'] = high
            self.cohorts_nonchurners_Day_1.setdefault(nonchurners_app_name,{})['medium'] = medium
            self.cohorts_nonchurners_Day_1.setdefault(nonchurners_app_name,{})['low'] = low
        print('5/6 Done ✅')
        return self

    def retention_summary_churners_Day_1(self):
        for app_name in self.cohorts_churners_Day_1.keys():
            print(app_name)
            cohort_by_apps = self.cohorts_churners_Day_1[app_name]
            for cohort_name, cohort_df in cohort_by_apps.items():
                print(cohort_name)
                tot_users = cohort_df['customer_id'].nunique()
                cohort_retention = {}
                for day in self.cut_off_day_cols:
                    daily_active_users = (cohort_df[day]>0).sum()
                    if tot_users == 0:
                        retention = 0
                    else: 
                        retention = (daily_active_users/tot_users)*100
                    cohort_retention[day] = retention
                retention_df = pd.DataFrame.from_dict(cohort_retention,columns = [cohort_name],orient = 'index').fillna(0)   
                self.retention_churners_Day_1.setdefault(app_name,{})[cohort_name] = retention_df
        print('6/6 Done ✅')
        return self

    def retention_summary_nonchurners_Day_1(self):
        for app_name in self.cohorts_nonchurners_Day_1.keys():
            print(app_name)
            cohort_by_apps = self.cohorts_nonchurners_Day_1[app_name]
            for cohort_name, cohort_df in cohort_by_apps.items():
                print(cohort_name)
                tot_users = cohort_df['customer_id'].nunique()
                cohort_retention = {}
                for day in self.cut_off_day_cols:
                    daily_active_users = (cohort_df[day]>0).sum()
                    if tot_users == 0:
                        retention = 0
                    else: 
                        retention = (daily_active_users/tot_users)*100
                    cohort_retention[day] = retention
                retention_df = pd.DataFrame.from_dict(cohort_retention,columns = [cohort_name],orient = 'index').fillna(0)   
                self.retention_nonchurners_Day_1.setdefault(app_name,{})[cohort_name] = retention_df
        print('6/6 Done ✅')
        return self

    def run(self): 
        self.app_split()
        self.clean_df()
        self.churn_df()
        self.first_day_activity_level()
        self.separate_apps_churners_nonchurners()
        self.cohorts_subgroubs()
        self.retention_summary_churners_Day_1()
        self.retention_summary_nonchurners_Day_1()
        return self

#  %% -----------------------------------------------------------------Retentiong on CRITICAL DAYS ----------------------------------------------------------------------
class CriticalEngagementShiftAnalyzer(PrepingDF): 
    def __init__(self,apps, last_day_col,retention_churners_Day_1):
        self.apps = apps
        self.last_day_col = last_day_col
        self.retention_Day1 = retention_churners_Day_1
        
        self.ann_shap_values = pd.DataFrame()
        self.Non_Onboarders = pd.DataFrame()
        self.ANN_shap_dict = {}
        self.critical_days_cols = {}
        self.CriticalEngagementDayActivity = {}
        self.cohorts_churners_tfd = {}             # tfd = top feature days (segment)
        self.cohort_summary_churners_tfd = {}
        self.retention_tfd = {}
    
        self.one_week_window_DF = pd.DataFrame()
        self.two_week_window_DF = pd.DataFrame()

    def critical_shap_days_labeling(self): 
        
        ann_shap_values = pd.read_csv(r'outputs/ANN_shap_values.csv')
        print(ann_shap_values)

        self.ANN_shap_dict = ann_shap_values.set_index('App Name')['feature_name'].apply(ast.literal_eval).to_dict()
        
        for app_name, feature_list in self.ANN_shap_dict.items():                    
            for top_day in feature_list:
                apps = self.apps.copy()
                app_df = apps[app_name].drop('first_day_activity_level', axis =1)
                app_df = app_df.drop('usage_type', axis = 1)
                top_day_numbers = int(top_day.replace('Day_', ''))
                days_cols = [f'Day_{i}' for i in range(top_day_numbers,81)]
                self.critical_days_cols[top_day] = days_cols
                
                churners = pd.DataFrame(app_df.loc[app_df['churn'] == 1].copy())
                churners_first_day_shift = churners[days_cols + ['customer_id']]
                
                conditions = [
                    (churners_first_day_shift[top_day] == 0), 
                    (churners_first_day_shift[top_day] > 0) & (churners_first_day_shift[top_day] <= 60),
                    (churners_first_day_shift[top_day] > 20)]
                
                choices = ['Failed Onboarding','Medium','High']
                churners_first_day_shift[f'{top_day}_activity_level'] = np.select(conditions, choices, default = None)   
                self.CriticalEngagementDayActivity.setdefault(app_name,{})[top_day] = churners_first_day_shift
        print('1/4 ...✅')
        return self

    def subgroubs_critical_activity(self):
        '''
        The cohorts we will to group would be based on:
            - app
            - churners
            - activity levels the new shifted first day which are the criticals 
        '''
        print('Creating Cohorts of activity levels on the first day for each groups')
        for app_name, critical_days_list in self.CriticalEngagementDayActivity.items():
            for critical_days, churners_app_df in critical_days_list.items():    
                # sort out activity level
                high = churners_app_df.loc[churners_app_df[f'{critical_days}_activity_level'] == 'High']
                medium = churners_app_df.loc[churners_app_df[f'{critical_days}_activity_level'] == 'Medium']
                low = churners_app_df.loc[churners_app_df[f'{critical_days}_activity_level'] == 'Low']

                self.cohorts_churners_tfd.setdefault(app_name,{}).setdefault(critical_days,{})['high'] = high
                self.cohorts_churners_tfd.setdefault(app_name,{}).setdefault(critical_days,{})['medium'] = medium
                self.cohorts_churners_tfd.setdefault(app_name,{}).setdefault(critical_days,{})['low'] = low
        
        print('2/4 ...✅')
        return self

    def retention_activity_churners(self):      
        for app_name in self.cohorts_churners_tfd.keys():
            cohort_tfd = self.cohorts_churners_tfd[app_name]
            for critical_days, critical_days_levels in cohort_tfd.items():
                for levels, cohort_dfs in critical_days_levels.items():
                    cohort_dfs
                    tot_users = cohort_dfs['customer_id'].nunique()
                    #print(f'{app_name} on {critical_days} for {levels} level total Users: ', tot_users)
                    retention_per_day = {}
                    retention_levels = {}
                    critical_days_cols = self.critical_days_cols[critical_days]
                    for day in critical_days_cols: 
                        daily_active_users = (cohort_dfs[day]>0).sum()
                        #print('Daily Active Users: ', daily_active_users)
                        
                        if tot_users == 0:
                            retention = 0
                        else: 
                            retention = (daily_active_users/tot_users)*100
                            #print('Daily Retention: ', retention)
                        retention_per_day[day] = retention
                    retention_df = pd.DataFrame.from_dict(retention_per_day,columns = [levels],orient = 'index').fillna(0)                    
                    self.retention_tfd.setdefault(app_name,{}).setdefault(critical_days,{})[levels] = retention_df
        print('3/4 ...✅')
        return self     

    def percent_diff_between_dayshifts(self):
        retention_Day1 = self.retention_Day1.copy() #find the difference in retention points with the new critical day shifts
        retention_tfd = self.retention_tfd.copy()
        one_week_rows = []
        two_week_rows = []
        for app_name, critical_days_list in retention_tfd.items():
            day1_levels = retention_Day1[app_name]   # ['high', 'medium', 'low']
            for critical_days, critical_days_levels in critical_days_list.items():
                for level in day1_levels:
                    critical_day_df = critical_days_levels[level]
                    day1_df = day1_levels[level]

                    # Calc the percent diff for each mode: low, medium, high
                    critical_day_df_ResetIndex = critical_day_df.reset_index(drop=True)
                    day1_df_ResetIndex = day1_df.reset_index(drop=True)
                    differences_one_week = (((critical_day_df_ResetIndex.iloc[1:9,:].sub(day1_df_ResetIndex.iloc[1:9,:]))/day1_df_ResetIndex.iloc[1:9,:] + 1e-6)*100)
                    differences_two_week = (((critical_day_df_ResetIndex.iloc[1:14,:].sub(day1_df_ResetIndex.iloc[1:14,:]))/day1_df_ResetIndex.iloc[1:14,:] + 1e-6)*100)

                    # Mean Across Days
                    avg_one_week = round(differences_one_week.mean().mean(),2)
                    avg_two_week = round(differences_two_week.mean().mean(),2)
                    one_week_rows.append({
                        'App Name': app_name,
                        'Critical Day': critical_days,
                        'Activity Level': level, 
                        'Average Percent Difference': avg_one_week
                    })
                    two_week_rows.append({
                        'App Name': app_name,
                        'Critical Day': critical_days,
                        'Activity Level': level, 
                        'Average Percent Difference': avg_two_week
                    })
        self.one_week_window_DF = pd.DataFrame(one_week_rows)
        self.two_week_window_DF = pd.DataFrame(two_week_rows)
        print('4/4 ...✅')
        return self

    def run(self):
        self.critical_shap_days_labeling()
        self.subgroubs_critical_activity()
        self.retention_activity_churners()
        self.percent_diff_between_dayshifts()
        return self


# %% --------------------------------------  Initiating: DATA PREPPING / FIRST DAY ANALYZER -----------------------------------------------------
#DF = pd.read_csv('C:\\Users\\Laure\\OneDrive\\Desktop\\DS Projects\\Churn_Prediction_Project\\3. Uploaded Data\\real_world_customer_churn_dataset.csv')
DF = pd.read_csv(r'files/Real_World_Customer_Churn_dataset RAW.csv')
df = pd.DataFrame(DF)
df = PrepingDF(df).run()
print('Finished cleaning')
retention_Day1 = df.retention_churners_Day_1
nonchurners_retention_day1 = df.retention_nonchurners_Day_1
print('Finished Day 1 Retention')


# %% ------------------------------------------  Initiating: CRITICAL DAY ANALYZER ----------------------------------------------
A = CriticalEngagementShiftAnalyzer(df.apps,df.last_day_col, df.retention_churners_Day_1).run()
Retention_TFD = A.retention_tfd
print('Finished CRITICAL day Retention')

# %% -------------------------------------------------------  Initiating: TABLES ------------------------------------------------
A_table = A.CriticalEngagementDayActivity
ONE_week_window_DIFF = A.one_week_window_DF
TWO_week_window_DIFF = A.two_week_window_DF
print('Finished WINDOWS Diff of Retention')

retention_ALL_results = {
    'Day_1_retention': retention_Day1,
    'Day_1_retention_NONchurners': nonchurners_retention_day1,
    'Day_CRIT_retention':Retention_TFD,
    'A_Table': A_table,
    '1_week_ret_diff': ONE_week_window_DIFF,
    '2_week_ret_diff': TWO_week_window_DIFF
}

# Store the dicts in a table
with open(r'outputs/retention_results.pk1', 'wb') as f:
    pickle.dump(retention_ALL_results,f)
print('SAVED the dicts of the Retention Tables!')
# %%
