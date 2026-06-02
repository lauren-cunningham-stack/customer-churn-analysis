#  %%  --------------------------------------------------  PREPPING / TRANSFORMING / FITTING (CLASSES)  --------------------------------------------
from sklearn.utils import resample
import ast
from sklearn.model_selection import train_test_split as tts
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines import CoxPHFitter
import pandas as pd
import numpy as np
import matplotlib as plt
import jinja2


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


class Prep_df:
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

        self.cut_off_day_cols = None
        
    def app_split(self):
        '''
        Split the dataset into apps dataframe.
        '''
        print("Splitting the Dataframe into Apss ...\n")  

        DataFrame = self.dataframe.copy()
        self.apps['Other Apps'] = DataFrame[DataFrame['usage_type'].isin(other_app_list)]
        self.apps['YouTube'] = DataFrame[DataFrame['usage_type'].isin(youtube_list)]
        self.apps['Facebook'] = DataFrame[DataFrame['usage_type'].isin(facebook_list)]
        self.apps['TikTok'] = DataFrame[DataFrame['usage_type'].isin(tiktok_list)]
        self.apps['WhatsApp'] = DataFrame[DataFrame['usage_type'].isin(whatsapp_list)]
        self.apps['Helakuru'] = DataFrame[DataFrame['usage_type'].isin(helakuru_list)]
        print('...✅')
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
            app_df = app_df.drop(['customer_id','usage_type'], axis = 1)
    
            # Convert pure whitespace strings to NaN
            app_df = app_df.replace(r'^\s*$', np.nan, regex=True)
    
            # Fill NaNs with 0 (if that's the business rule)
            app_df = app_df.fillna(0)       
        
            # Drop duplicates
            before = len(app_df)
            app_df = app_df.drop_duplicates()
            after = len(app_df)
            #print(f"Dropped {before - after} duplicates. Remaining rows: {after}")
            
            # Update Dictionary 
            self.apps[app_name] = app_df
        print('...✅')
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
        self.cut_off_extra_day_cols = [f'Day_{i}' for i in range(81,self.last_day_col+1)]
        self.cutoff_day = self.last_day_col - 10
        self.cut_off_day_cols = [f'Day_{i}' for i in range(1,self.cutoff_day+1)]
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
            app_df['Duration'] = app_df['last_active_day']
    
            # Churn definition
            app_df['event'] = ((self.cutoff_day - app_df['last_active_day']) >= inactivity_window).astype(int)  
            app_df = app_df.drop('last_active_day', axis = 1)
            self.apps[app_name] = app_df.drop(self.cut_off_extra_day_cols, axis = 1)
        print('...✅')
        return self

    def downsample_subset_df(self):
        '''
         The dataframe is too large for the CPU to handle, thus we must downsample it. 
         For the Model to not pick churner each time, we must match the same amount of churners vs. nonchurners for the model to actually learn
        '''
        print('Downsample each app\'s datasets')
        for app_name, app_df in self.apps.items():
             # Split the Df in Minority / Majority 
            df_majority = app_df[app_df['event'] == 0]
            df_minority = app_df[app_df['event'] == 1]
    
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

            self.apps[app_name] = subset_df_frac
        print('...✅')
        return self

    def agg_features(self):
        print('Creating Aggregated Features')
        ann_shap_values = pd.read_csv(r'outputs/ANN_shap_values.csv')
        self.ANN_shap_dict = ann_shap_values.set_index('App Name')['feature_name'].apply(ast.literal_eval).to_dict()

        for app_name, feature_list in self.ANN_shap_dict.items():                    
            for top_day in feature_list:
                app_df = self.apps[app_name]
                first_7_days_cols = [f'Day_{i}' for i in range(1,8)]
                # Aggregated Activity Behavior
                app_df['avg_activity_first_week'] = app_df[first_7_days_cols].mean(axis=1)
                app_df['avg_activity_total'] = app_df[self.cut_off_day_cols].mean(axis=1)
                #app_df['consistently_active'] = (app_df[self.cut_off_day_cols]>0).sum(axis=1)/len(app_df[self.cut_off_day_cols])
                
                # Early Behavior Binary Activity
                app_df['used_Day_1'] = (app_df['Day_1'] > 0).astype(int)
                app_df['used_Day_7'] = (app_df['Day_7'] > 0).astype(int)

                # Critical Days Behavior Binary Activity
                app_df[f'used_{top_day}'] = (app_df[top_day] > 0).astype(int)
                self.apps[app_name] = app_df
        print('...✅')
        return self

    def XY_split(self):
        '''
        Defining what the x and y datasets are for each apps before train test split. 
        '''
        print("Splitting Dataframe into X and Y...\n")
        
        for app_name, app_df in self.apps.items():
            # Drop Day Columns
            df = app_df.drop(self.cut_off_day_cols, axis = 1)
            days_df = app_df[self.cut_off_day_cols]
            
            x = app_df.drop(columns = ['event'])
            y = app_df['event']
            self.agg_col = [col for col in x.columns]
            self.XY_apps[app_name] = {'x': x, 'y': y,'df':df, 'days_df':days_df}
        print('...✅')
        return self 

    def XY_traintest_split(self):
        print("Train Test Split...\n")  
        for app_name, xy in self.XY_apps.items():
            x = xy['x']
            y = xy['y']
            df = xy['df']
            days_df = xy['days_df']

            # split the train test sets
            x_train, x_test,y_train,y_test = tts(x,y, test_size = 0.3, random_state = 42)

            self.train_test_apps_split[app_name] = {
                'x_train': x_train,
                'x_test' : x_test,
                'y_train': y_train,
                'y_test' : y_test,
                'x'      : x,
                'y'      : y, 
                'df'     : df,
                'days_df': days_df
                }
        print('...✅')
        return self 

    def run(self):
        self.app_split()
        self.clean_df()
        self.churn_df()
        #self.downsample_subset_df()
        self.agg_features()
        self.XY_split()
        self.XY_traintest_split()
        return self


# %%  --------------------------------------------------------------- HAZARD CURVES  ----------------------------------------------------------------
class survival_curves(Prep_df):
    def __init__(self, train_test_apps_split): 
        self.tt = train_test_apps_split
        self.feature_sig = pd.DataFrame()
        self.feature_summary = pd.DataFrame()
        
            


    def Kaplan_meier(self):
        '''
        Plot a survival curves of the overall data using E = 'event of churn', T = time over the last active day
        First graph displays the survival probability over the time (days). Whereas, the following graphs after display
        the survival probability for onboarding vs critcal day in order to see the shift in probability and which is more impactful. 
        '''
        print('Working on Kaplan Meirer survival probability curve')

        kmf = KaplanMeierFitter()
        for app_name, ttas in self.tt.items():
            df = ttas['df']
            E = df['event']
            T = df['Duration']
            kmf.fit(durations = T, event_observed = E)
            kmf.plot_survival_function()
            plt.title(f'Survival Function for {app_name}')
            plt.ylabel('Survival Probability')
            plt.xlabel('Time (Days)')
            plt.xlim(0,90)
            plt.grid(True)
            plt.show()
            
            
            used_cols = [col for col in df.columns if 'used' in col]
            for col in used_cols: 
                fig, ax = plt.subplots()
                m = (df[col] == 0)
                kmf.fit(durations = T[m], event_observed = E[m], label = 'Inactive')
                kmf.plot_survival_function(ax = ax)
                kmf.fit(durations = T[~m], event_observed = E[~m], label = 'Active')
                kmf.plot_survival_function(ax = ax, at_risk_counts = True)
                plt.title(f'{app_name}: Survival by {col} (Active vs Inactive)')
                ax.grid(True)
                plt.show()
        print('...✅')
        return self

    def Cox_pro_hazards(self):
        '''
        Cox proportional Hazards regression model assumptions: 
            - independnce of survival times between distinct __________ in the sample
            - multiplicative relationship between the predictors and the hazard
            - a constant hazard ratioi over time
        Hazard Ratio: 
            - hazard : slope of the survival curve. measures how rapidly users are churning  
            - HR compares two groups, if the hazard ratio is 2.0 then the rate of churn in one group is twice the rate in the other group. 
        '''
        print('Working on Cox rate of survival')
        feature_sig_rows = []
        cph = CoxPHFitter(penalizer = 0.1) # applying regularization to stablize coefficient estimation with the high correlated features
        for app_name, ttas in self.tt.items():
            df = ttas['df']
            cph.fit(df, duration_col = 'Duration', event_col = 'event')
            summary_df = cph.summary
            p_values = cph._compute_p_values()
            hazard_ratios = cph.hazard_ratios_ 
            features = summary_df.index.tolist()
            for (x,y,z) in zip(features,p_values,hazard_ratios):
                if y < 0.005: 
                    y = '< 0.005'
                else: 
                    y = round(y,2)
                feature_sig_rows.append({
                        'app name': app_name,
                        'Critical Days': x,
                        'p-value': y, 
                        'hazard ratios': round(z,2)
                    })
            
            feature_sig = pd.DataFrame(feature_sig_rows).reset_index(drop = True)
        feature_sig = feature_sig.sort_values(by = ['app name','hazard ratios'], ascending = [True, False])
        feature_sig = feature_sig.groupby('app name').head(3)
        self.feature_sig = feature_sig.style.set_properties(subset = ['app name','Critical Days'], **{'text-align': 'left'})
        print('...✅')
        return self

    def Cox_feature_summary(self):
        feature_sum = self.feature_sig.data
        feature_sum['day num'] = feature_sum['Critical Days'].str.replace('used_Day_','').astype('int64')
        interpretation_row = []
        for (day,ratio) in zip(feature_sum['day num'], feature_sum['hazard ratios']):
            if day<2:
                if ratio>1.5:
                    label = 'High Onboarding Churn Risk'
                else:
                    label = 'Onboarding Behavioral Signal'
            elif 1<day<8:
                if ratio>1.5:
                    label = 'User Disengagment Signal'
                elif ratio>1.00:
                    label = 'Early Churn Risk'
                else: 
                    label = 'Retention Opportunity'
            elif 7<day<15:
                if ratio > 1.00:
                    label = 'Engagement Drop-Off'
                else:
                    label = 'Habit Formation Window'
            elif day>14:
                if ratio>1.00:
                    label = 'Mid-Cycle Churn Risk'
                elif ratio<1.00:
                    label = 'Re-Engagement Window'
            
            interpretation_row.append({
                'day num' : day,
                'hazard ratios': ratio,
                'Interpretation': label
                })   
        interpretation_col = pd.DataFrame(interpretation_row)
        feature_summary = pd.merge(feature_sum,interpretation_col, on = ['day num','hazard ratios'], how = 'outer')
        feature_summary = feature_summary.drop(['p-value','day num'], axis = 1).drop_duplicates().reset_index(drop = True)
        #feature_summary = feature_summary.style.set_properties(subset = ['app name','Critical Days','Interpretation'], **{'text-align': 'left'})
        self.feature_summary = feature_summary
        return self
        

    def run(self): 
        #self.Kaplan_meier()
        self.Cox_pro_hazards()
        self.Cox_feature_summary()
        return self

#  %%  ------------------------------------------------ INITIATING: prepping / transforming / fitting  ---------------------------------------
DF = pd.read_csv(r'files/Real_World_Customer_Churn_dataset RAW.csv')
df = pd.DataFrame(DF)
df = Prep_df(df).run()

ttas = df.train_test_apps_split


# %%  --------------------------------------------------------------  SURVIVAL CURVES  -----------------------------------------------------
# ## Second Class
survival_class = survival_curves(df.train_test_apps_split).run()

feature_sig = survival_class.feature_sig.hide(axis = 'index')
feature_sig

feature_summary = survival_class.feature_summary
feature_summary['hazard ratios'] = (feature_summary['hazard ratios'] - 1)*100
feature_summary['Critical Days'] = feature_summary['Critical Days'].str.replace(r'\D+','',regex=True).astype(int)
feature_summary.to_csv('outputs/COX_hazard_ratios.csv', index = False)
print(feature_summary)
print('\n\n IT IS SAVED!')
# %%
