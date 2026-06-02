import streamlit as st
import altair as alt
import pandas as pd
import numpy as np 
import plotly.express as px
from itertools import islice
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import graphviz
import requests 
import pickle
from io import StringIO
from io import BytesIO
from huggingface_hub import hf_hub_download # cache the files locally after first download

# ---------------------------------------------------- Grabbing External Files  ----------------------------------------------------------------
@st.cache_resource
def load_dict(file_name):
    file_path = hf_hub_download(
                repo_id="lauren-cunningham-stack/churn-analysis-outputs-folder",
                filename=file_name,
                repo_type="dataset"
            )

    with open(file_path, "rb") as f:
        return pickle.load(f)
    
@st.cache_data
def load_csv(file_name):
    file_path = hf_hub_download(
                repo_id="lauren-cunningham-stack/churn-analysis-outputs-folder",
                filename=file_name,
                repo_type="dataset"
            )

    return pd.read_csv(file_path)

# ---------------------------------------------------  DICTS / CSV FILES  ---------------------------------------------------------------------
afi = load_dict('shap_daily_values.pk1')

retention_results = load_dict('retention_results.pk1')
retention_Day1 = retention_results['Day_1_retention']
Retention_TFD = retention_results['Day_CRIT_retention']
ONE_week_window_DIFF = retention_results['1_week_ret_diff']
TWO_week_window_DIFF = retention_results['2_week_ret_diff']


ecps = load_csv('early_churn_prediciton_table.csv')
cox_hazard_ratios = load_csv('COX_hazard_ratios.csv')



# ------------------------------------------------------------------  SIDEBAR  -----------------------------------------------------------------------------
with st.sidebar: 
    st.caption(
        "Executive overview of churn drivers, risk windows, and retention opportunities"
    )
    st.markdown("## 🛣️ Project Roadmap")
    side = st.sidebar.radio('View', ['Business Side', 'Nerd Side'])
    st.divider()

if side == 'Business Side':
    st.sidebar.markdown("### 📈 Business Roadmap")
    st.sidebar.markdown("""
        📍 [Overview](#overview)

        📢 [Behavioral Drivers of Churn Risk](#behavioral-drivers-of-churn-risk)

        🔮 [Real-Time Churn Monitoring](#real-time-churn-monitoring)

        🎯 [Re-Engagement & Churn Risk Patterns](#re-engagement-&-churn-risk-patterns)

        ⚛︎ [Strategic Recommendations and Business Impact](#strategic-recommendations-and-business-impact)
        """)

    # --------------------------------------------------------  BUSINESS Main Page -------------------------------------------------------
    st.title('📱 User Retention & Churn Intelligence Platform')

    #impage_path = 'Churn_Analysis_Project/other/social-media-logo-collection.jpg'
    with st.container(horizontal_alignment='center'):
        st.image('files/social-media-logo-collection.jpg', width=400)

    st.markdown("<div id='overview'></div>", unsafe_allow_html=True)
    st.header("Overview")
    st.write("""
        This platform analyzes behavioral engagement patterns to identify
        when users are most likely to churn and where intervention strategies
        can improve long-term retention.

        The system combines survival analysis, behavioral modeling,
        and explainable machine learning to detect churn risk early
        and identify actionable recovery windows across the customer lifecycle.
        """)
    apps = [
        'Other Apps', 'YouTube', 'Facebook', 'TikTok', 'WhatsApp', 'Helakuru'
    ]
    col1, col2 = st.columns(2)
    col1.metric(
        "Peak Churn Risk",
        "78–84%",
        "Day 1 Onboarding"
    )
    col2.metric(
        "Earliest Reliable Detection",
        "3–4 Days"
    )

    col3, col4 = st.columns(2)
    col3.metric(
        "Primary Recovery Window",
        "Days 14–19"
    )
    col4.metric(
        "Target Precision Threshold",
        "≥75%"
    )
    #  ------------------------------------------------------- Cox Hazard Ratios  --------------------------------------------------------
    st.subheader('Early Lifecycle Churn Concentration')
    st.write("""
        Churn behavior was first analyzed using Cox Hazard modeling to determine
        whether user disengagement was concentrated during onboarding and early lifecycle activity.

        The analysis focused on the first 15 days of activity to measure how rapidly
        churn risk declined after onboarding and identify when users became most vulnerable to disengagement.
        """)
    max_days = cox_hazard_ratios['Critical Days'].max()
    # Graphing the dataframe
    cox_charts = alt.Chart(cox_hazard_ratios).mark_line(point=True).encode(
        alt.X('Critical Days:Q',
            title='Days',
            scale=alt.Scale(domain=[0, max_days + 1])),
        alt.Y('hazard ratios:Q',
            title='Hazard Percentage',
            scale=alt.Scale(domain=[0, 100])),
        color=alt.Color('app name:N', title='App'),
        opacity=alt.value(0.8),
        tooltip=[
            'app name', 'Critical Days', 'hazard ratios', 'Interpretation'
        ]).interactive()
    rule = alt.Chart(pd.DataFrame({'x': [1]
                                    })).mark_rule(color='red',
                                                    strokeDash=[5,
                                                             5]).encode(x='x')

    st.altair_chart(cox_charts + rule, use_container_width=True)
    st.write("""
            #### Insight
            - Churn risk was overwhelmingly concentrated during Day 1 onboarding across all applications
            - Hazard ratios declined sharply after the first week, indicating that early engagement behavior strongly influences long-term retention
            - Several applications continued showing elevated churn sensitivity beyond onboarding, suggesting additional behavioral risk periods existed later in the lifecycle

            #### Action
            - Prioritize onboarding quality and first-session engagement
            - Monitor early behavioral inactivity during the first week
            - Identify additional lifecycle periods where churn risk re-emerges
            """)
    st.write("""
                While onboarding represented the highest-risk churn period,
                hazard analysis alone could not determine which additional behavioral windows
                most strongly influenced future churn probability.

                To identify these recurring churn signals across the customer lifecycle,
                explainable machine learning (SHAP) was used to measure which engagement periods
                consistently contributed most to churn prediction.
             
                ---

        """)
    # --------------------------------------------------------  Feature Importance   -------------------------------------------------------
    st.markdown("<div id='behavioral-drivers-of-churn-risk'></div>",
                unsafe_allow_html=True)
    st.header("Behavioral Drivers of Churn Risk")
    st.write("""
            Following the Cox Hazard analysis, churn was shown to be highly concentrated
            during onboarding and the first week of user activity.

            However, timing alone does not explain *what user behaviors* are driving this risk.
            This indicates churn is not localized to onboarding alone.
            """)
    # STRUCTURE:    project | app name | model name | DF
    app_tab_FI = st.tabs(apps)
    for tab, app in zip(app_tab_FI, apps):
        with tab:
            st.markdown('#### Key Behavioral Risk Periods')
            filtered_ann = afi[app]['ANN']['Churners'].copy()
            filtered_ann['Days'] = filtered_ann['Days'].str.replace('Day_', '', regex=False).astype(int)
            filtered_ann['SHAP Values'] = filtered_ann['SHAP Values'].str[0]
            filtered_ann['SHAP Values'] = filtered_ann['SHAP Values'].astype(float)
            top_threshold = (filtered_ann.sort_values(
                by=['SHAP Values'],
                ascending=[False]).head(5)).reset_index(drop=True)
            afi_barChart = alt.Chart(top_threshold).mark_bar(
                size=30, color='darkolivegreen').encode(
                    x=alt.X('Days:N', axis=alt.Axis(labelAngle=0)),
                    y='SHAP Values').interactive()
            afi_charts = (afi_barChart).properties(
                title='SHAP Values')
            st.altair_chart(afi_charts, use_container_width=True)
    st.write('''
            #### Insight
            - Churn risk is not driven by a single lifecycle stage, but by recurring behavioral windows across the user journey
            - Several engagement periods beyond onboarding consistently contribute to churn prediction
            - Similar behavioral patterns are observed across multiple applications, indicating a shared churn structure

            #### Action
            - Monitor recurring high-risk engagement periods identified by SHAP
            - Trigger interventions before disengagement patterns become permanent
            - Extend retention focus beyond onboarding into early and mid-lifecycle stages
            ''')
    st.write("""
                    While SHAP identifies the key behavioral periods that contribute most to churn risk,
                    it does not explain how users behave within those periods over time.

                    To understand whether these high-risk windows represent opportunities for recovery
                    or irreversible disengagement, retention curve analysis was applied to examine
                    user behavior specifically around these critical lifecycle stages.
             
                    ---

                    """)
    # --------------------------------------------------------  Retention Curves -------------------------------------------------------
    st.markdown("<div id='re-engagement-&-churn-risk-patterns'></div>",
                unsafe_allow_html=True)
    st.header("Retention Curves & Re-Engagement Patterns")
    st.write("""
            Building on the SHAP analysis, which identified recurring high-risk behavioral periods,
            retention curve analysis was used to examine how users actually behave within those critical lifecycle windows.

            This allows us to determine whether these high-risk periods represent:
            - temporary disengagement that can be recovered
            - or irreversible churn trajectories
            """)

    # STRUCTURE:    project : Day_1_retention: app name : [high, medium, low] : DF
    # STRUCTURE:    project : Day_CRIT_retention : app name : [Day_13, Day_6, Day_15] :[high, medium, low] : DF
    # STRUCTURE:    project : A_Table : app name : [Day_13, Day_6, Day_15] : DF
    # STRUCTURE:    project : 1_week_ret_diff : DF
    # STRUCTURE:    project : 2_week_ret_diff : DF _______________________________________________

    app_tab_RC = st.tabs(apps)
    for tab, app in zip(app_tab_RC, apps):
        with tab:
            # -------------------------------------------------------Onboarding vs. re-engagement patterns for high-activity users
            #  --- Day 1 ---
            day1_df = retention_Day1[app]['high']
            day1_df['Retention'] = day1_df['high']
            day1_df = day1_df.drop('high', axis=1)
            day1_df['Days'] = day1_df.index.str.replace('Day_', '', regex=False).astype(int)
            day1_df = day1_df.reset_index(drop=True)

            line_d1 = alt.Chart(day1_df).mark_line().encode(
                x=alt.X('Days:Q', title='Cohort Period (Days)'),
                y=alt.Y('Retention:Q', title='Retention Rate'))

            points_d1 = alt.Chart(day1_df).mark_circle(size=30).encode(
                x='Days:Q', y='Retention:Q')

            
            st.markdown('#### Retention Decay After Onboarding')
            st.write("""
                    These signals converge on a single constraint: churn becomes predictable early, but actionable only within a narrow window.
                """)
            st.caption(
                'Onboarding vs. re-engagement patterns for high-activity users'
            )
            chart_d1 = (line_d1 + points_d1).interactive()
            st.altair_chart(chart_d1, use_container_width=True)

            # --- Critical Days ---
            charts_cr = []
            app_crit_dict = Retention_TFD[app]

            for critical_day, activity_levels in app_crit_dict.items():
                high_df = activity_levels['high']  #.reset_index()
                high_df['Retention'] = high_df['high']
                high_df['Days'] = high_df.index.str.replace('Day_', '', regex=False).astype(int)
                high_df = high_df.drop(['high'], axis=1)
                high_df['Cohort'] = critical_day
                high_df = high_df.reset_index(drop=True)
                charts_cr.append(high_df)

            # Combine everything
            combined_df = pd.concat(charts_cr)

            # --- Chart ---
            line_cr = alt.Chart(combined_df).mark_line().encode(
                x=alt.X('Days:Q', title='Cohort Period (Days)'),
                y=alt.Y('Retention:Q', title='Retention Rate'),
                color=alt.Color('Cohort:N', title='Cohort'),
                tooltip=['Cohort', 'Days', 'Retention'])

            points_cr = alt.Chart(combined_df).mark_circle(size=30).encode(
                x='Days:Q', y='Retention:Q', color='Cohort:N')
            highlight_bump = alt.Chart(
                pd.DataFrame({
                    'start': [40],
                    'end': [52]
                })).mark_rect(opacity=0.1, color='orange').encode(x='start:Q',
                                                                  x2='end:Q')

            chart_cr = (line_cr + points_cr).interactive()
            st.markdown("#### Re-Engagement Behavior Before Churn")
            st.write("""
                    Several applications showed temporary re-engagement periods before churn,
                    suggesting that disengagement develops gradually rather than occurring immediately.
                """)
            st.altair_chart(chart_cr + highlight_bump)
            st.write("""
                    When mapping retention behavior onto the SHAP-identified high-risk periods,
                    a consistent pattern emerges:

                    - Later-stage churn (Days 14–52) shows multiple re-engagement opportunities
                    - Users often return briefly before fully disengaging
                    - These windows represent recoverable churn behavior rather than immediate loss
                    """)
            app_notes = {
                'Other Apps':
                "- Re-engagement spike around Day 16\n- Final activity spike between Days 36–47",

                'Facebook':
                "- Re-engagement spike around Days 15–16\n- Final activity spike between Days 37–52",

                'TikTok':
                "- Re-engagement spike around Day 17\n- Final activity spike between Days 36–52",

                'YouTube':
                "- No major re-engagement or final activity spike observed",

                'WhatsApp':
                "- Re-engagement spike around Day 15\n- Final activity spike between Days 36–47",

                'Helakuru':
                "- Re-engagement spike around Day 16\n- Final activity spike between Days 36–47"
            }
            st.markdown(app_notes[app])
    
    st.write("""
        These findings indicate that while later-stage churners exhibit clear recovery windows,
        retention strategies alone are not sufficient to address early lifecycle drop-off.

        A significant portion of users disengage too early for re-engagement campaigns to be effective,
        particularly within the first week of activity.
        """)

    st.write("""
        This creates a natural split in the churn problem:

        - Later-stage churners can be recovered through timed re-engagement strategies
        - Early-stage churners require immediate detection and intervention before disengagement stabilizes

        To address this, an Early Churn Detection System was developed to identify at-risk users
        within the first days of activity, enabling proactive intervention during the onboarding phase.
                          
        ---

        """)
    # -------------------------------------------------------------------------------------  Early Prediction System -------------------------------------------------------
    st.markdown("<div id='real-time-churn-monitoring'></div>",
                unsafe_allow_html=True)
    st.header("Real-Time Churn Monitoring")
    st.write("""
            The Cox, SHAP, and retention analyses collectively revealed a consistent pattern:
            churn risk emerges early in the lifecycle, but becomes actionable only within a narrow intervention window.

            To operationalize these insights, an Early Churn Detection System was developed to identify at-risk users
            within the first days of activity and enable intervention before disengagement stabilizes.
              """)
    # Create two selections:
    app_tab_EPS = st.tabs(apps)
    for tab, app in zip(app_tab_EPS, apps):
        with tab:
            tct = ecps[ecps['App_name'] == app]
            tct = tct[tct['Model'] == 'XGB']
            tct = tct[tct['Threshold'] == 0.3]
            tct = tct.sort_values(['Precision', 'False Negatives'],
                                  ascending=[False,
                                             True]).groupby(['Days']).head(1)
            tct = tct.sort_values('Days', ascending=True)
            threshold_85 = tct[(tct['False Negatives'] <= 10)
                               & (tct['Precision'] >= 80)]

            threshold_85 = threshold_85.sort_values('Days').head(1)
            threshold_75 = tct[(tct['False Negatives'] <= 10)
                               & (tct['Precision'] >= 74.99)]
            threshold_75 = threshold_75.sort_values('Days').head(1)

            base = alt.Chart(tct).mark_circle(size=80).encode(
                x='Days:Q',
                y='Precision:Q',
                size='False Negatives:Q',
                color=alt.condition((alt.datum['False Negatives'] <= 10.99) &
                                    (alt.datum['Precision'] >= 74.99),
                                    alt.value('green'),
                                    alt.value('lightgrey')),
                tooltip=['Days', 'Precision', 'False Negatives'])

            highlight = alt.Chart(threshold_85).mark_circle(
                size=250, color='gold', stroke='orange',
                strokeWidth=2).encode(x='Days:Q', y='Precision:Q')

            text = alt.Chart(threshold_85).mark_text(
                dy=-15, fontWeight='bold',
                color='white').encode(x='Days:Q',
                                      y='Precision:Q',
                                      text=alt.value('Earliest Valid'))
            st.altair_chart(base + highlight + text)
    st.write('''
            The system was designed to balance early detection with intervention reliability.
            While earlier predictions improve recovery potential, overly aggressive targeting increases operational cost
            and reduces intervention efficiency.

            As a result, the final model prioritizes a balanced threshold that optimizes:
            - early churn visibility
            - precision of intervention targeting
            - manageable false-positive rates

            Across all applications, the system consistently identifies churn risk within the first week of activity
            while maintaining strong predictive stability.
    ''')

    st.markdown('''
            #### Key Operational Insights
            - Churn risk becomes reliably detectable within the first few days of user activity
            - Early detection enables intervention before behavioral disengagement becomes permanent
            - Reducing missed churners is prioritized over maximizing raw model accuracy due to higher recovery value in early lifecycle stages
            - Intervention effectiveness is highest when aligned with SHAP-identified behavioral windows

            #### Recommended Business Actions
            - Monitor onboarding and early engagement behavior in real time
            - Trigger retention campaigns during SHAP-identified high-risk periods
            - Prioritize improving first-week activation and user habit formation
            - Continuously update risk scores as new behavioral data becomes available
                     
            ---

    ''')
    
    # -------------------------------------------  BUSINESS RECOMMENDATIONS  ----------------------------------------------------------------
    st.markdown(
        "<div id='strategic-recommendations-and-business-impact'></div>",
        unsafe_allow_html=True)
    st.header("Retention Strategy & Business Impact")

    st.markdown("""
            #### From Behavioral Signals to Actionable Retention Strategy

            Across survival analysis, feature attribution (SHAP), and retention curve modeling, a consistent lifecycle pattern emerged:

            - Churn is **not a single-event outcome**, but a **progressive process**
            - Risk is **highest during onboarding**, but remains detectable across multiple lifecycle stages
            - Users exhibit **repeatable re-engagement windows before full churn**, creating intervention opportunities beyond the first session

            These findings establish a unified insight:  
            > **Retention can be materially improved by aligning interventions to lifecycle timing rather than static user states.**

            ---

            #### What This Means for the Business

            Instead of reacting to churn after it occurs, the system enables a shift toward **predictive lifecycle intervention**:

            - Identify users early in their engagement journey
            - Detect risk escalation patterns before disengagement stabilizes
            - Intervene during known recovery windows where behavior is still reversible

            This turns churn management from a reactive reporting metric into a **timed intervention system**.

            ---

            ### Recommended Retention Strategy

            ##### 1. Optimize Onboarding (Highest Impact)
            - Reduce early-session friction and drop-off
            - Ensure users reach “first value” within Day 1–3
            - Strengthen activation signals during initial usage

            ##### 2. Lifecycle-Based Intervention Campaigns
            - Trigger engagement workflows during known risk windows
            - Prioritize users showing early inactivity signals
            - Use behavioral timing rather than static segmentation

            ##### 3. Reinforce Habit Formation
            - Encourage repeated engagement in the first week
            - Incentivize return behavior before churn trajectory stabilizes
            - Strengthen early lifecycle consistency

            ---

            #### System Outcome

            This framework enables a shift from static churn prediction to **continuous churn prevention**, where:

            - Users are scored in real time based on behavioral evolution
            - Intervention timing is aligned with empirical risk windows
            - Retention strategies adapt dynamically across the lifecycle

            ---

            #### Business Impact

            - Earlier identification of at-risk users (within first week of activity)
            - Higher efficiency targeting through time-aware intervention windows
            - Reduced churn loss through proactive engagement strategies
            - Scalable retention system applicable across multiple applications
            """)

    st.subheader("Operational Workflow")

    st.markdown("""
            1. Monitor user behavior in real time across lifecycle stages  
            2. Score churn risk continuously as engagement patterns evolve  
            3. Trigger interventions during validated high-risk windows  
            4. Measure response and re-score users dynamically  
            """)

    st.subheader("Final Summary")

    st.write("""
            This project demonstrates that churn is best understood as a time-dependent behavioral process rather than a binary outcome.

            By combining survival modeling, feature-level attribution, and retention curve analysis, we can move from identifying churn risk to actively shaping user retention outcomes through timed, data-driven interventions.
            """)
    


elif side == 'Nerd Side': 
    st.sidebar.markdown("### 🤓 Nerd Roadmap")
    st.sidebar.markdown("""
        ⚔︎   [Modeling Challenges](#modeling-challenges)
                        
        ∑   [Feature Engineering Pipeline](#feature-engineering-pipeline)
                        
        👩🏻‍🔬   [Model Architecture & Selection](#model-architecture-&-selection)   

        🪖  [Evaluating Temporal Churn Risk with Cox Hazard Modeling](#evaluating-cox-hazard-ratios)                   
                
        📐 [SHAP Explainability](#shap-explainability)
                
        ⚖️ [Real-Time Churn Monitoring](#real-time-churn-monitoring)  
                        
        🧮 [Churners Retention Windows](#churners-retention-windows)
                
        🏗️ [Pipeline Architecture](#pipeline-architecture)
        """)
    apps = ['Other Apps','YouTube', 'Facebook', 'TikTok', 'WhatsApp', 'Helakuru']
    # --------------------------------------------------------  NERDS Main Page -------------------------------------------------------
    st.title('📱 Churn Prediction System')
    st.write(""" 
            User churn is not a single-event outcome, but a time-dependent behavioral process that develops progressively across the customer lifecycle.
            This project was designed to model how disengagement evolves over time, identify the behavioral periods most associated with churn risk, and develop an early intervention framework capable of detecting at-risk users before disengagement stabilizes.

            The system combines:
            - survival analysis to measure churn timing risk
            - explainable machine learning to identify high-signal behavioral periods
            - retention curve analysis to validate recovery opportunities
            - early prediction systems for proactive intervention during onboarding

            The primary objective was not simply to maximize classification accuracy, but to build a reliable and interpretable churn monitoring framework capable of supporting real-time retention strategies under noisy and highly imbalanced behavioral data.
        """)
    st.header("Key Results")
    st.markdown("""
        - Churn risk became reliably detectable within the first 3–4 days of user activity
        - Survival analysis confirmed that onboarding represented the highest-risk lifecycle stage across applications
        - SHAP analysis identified recurring behavioral risk windows beyond onboarding
        - Retention analysis revealed measurable re-engagement periods prior to permanent churn
        - XGBoost performed best for early churn detection, while the ANN captured broader long-term engagement dynamics
        - Removing temporally aggregated features significantly reduced leakage-related performance inflation and improved model reliability
    """)


    st.markdown("<div id='modeling-challenges'></div>", unsafe_allow_html=True)
    st.header("Modeling Challenges")
    st.write("""
                The modeling process faced two main challenges: severe class imbalance and high computational cost.
                Most users did not churn, which caused models to bias heavily toward the majority class and made accuracy an unreliable metric. At the same time, training full-scale models across multiple configurations was computationally expensive and slowed experimentation.
                To address this, stratified sampling and class-balanced subsampling were used to preserve the original churn distribution while reducing training overhead. This made it possible to iterate faster across feature transformations, thresholds, and model architectures.
                Overall, these changes stabilized training and improved focus on early churn detection rather than raw accuracy.
                """)

    st.markdown("<div id='feature-engineering-pipeline'></div>", unsafe_allow_html=True)
    st.header("Feature Engineering Pipeline")
    st.markdown("""
            Behavioral data was highly skewed, sparse, and zero-heavy due to long periods of inactivity across users. A small number of highly active users also created extreme outliers, which distorted raw distributions and reduced model stability.
            To fix this, multiple transformation methods were tested, including log scaling, square root transforms, inverse hyperbolic sine (IHS), Yeo-Johnson, Box-Cox, and robust winsorization using IQR and MAD thresholds. Each method was evaluated to improve distribution stability while preserving meaningful behavioral patterns.
            Because many features contained long stretches of zeros, a custom preprocessing step was added to apply transformations only to non-zero values. This helped prevent distortion while improving convergence for both XGBoost and ANN models.
            To avoid data leakage, any aggregated features that included future information were removed. All features were constrained to only use behavior that occurred before the prediction window, ensuring the model learned from valid historical patterns only.
                """)
    st.markdown("<div id='model-architecture-&-selection'></div>", unsafe_allow_html=True)
    st.header("Model Architecture & Selection")
    st.markdown("""
                No single model architecture captured the full structure of churn behavior effectively.
                XGBoost performed best for early-stage churn detection due to its ability to model sparse behavioral signals and nonlinear feature interactions under limited observation windows.
                However, longer-term churn patterns were often more gradual and behaviorally diffuse. To capture these temporal engagement structures, ANN models were introduced to learn broader nonlinear engagement dynamics across the lifecycle.
                Rather than treating the models as competitors, the final system leveraged both approaches:
                - XGBoost optimized early intervention reliability
                - ANN models captured extended behavioral progression patterns
            """)
    # --------------------------------------------------------  COX - Hazard Ratios -------------------------------------------------------
    # Creating interactive GRAPHS 
    st.markdown("<div id='evaluating-cox-hazard-ratios'></div>", unsafe_allow_html=True)
    st.header("Evaluating Temporal Churn Risk with Cox Hazard Modeling")
    st.write("""
        Before building predictive models, the first objective was to determine whether churn risk
        followed a meaningful temporal structure across the customer lifecycle.

        Since early exploratory analysis suggested substantial onboarding drop-off,
        Cox Proportional Hazard modeling was used to measure how churn risk evolved over time
        and identify whether disengagement was concentrated within specific lifecycle stages.
        """)
    st.latex(r'''
            h(t|X) = h_0(t)\exp(\beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n)
            ''')
    st.caption("""
            The Cox Proportional Hazards model estimates how behavioral activity shifts churn risk
            relative to a baseline hazard function over time.
            """)
    apps = ['Other Apps','YouTube', 'Facebook', 'TikTok', 'WhatsApp', 'Helakuru']
    app_tab = st.tabs(apps)
    for tab, app in zip(app_tab, apps):  
        with tab: 
            st.write(''' 
            Peak hazard intervals were extracted per application to identify the lifecycle stages with the highest churn concentration. 
            ''') 

            filtered_df = cox_hazard_ratios[cox_hazard_ratios['app name']==app] 
            st.write(filtered_df) 

    st.write("""
        ### Key Findings

        Cox Hazard analysis revealed a highly non-uniform churn distribution across the lifecycle:

        - Day 1 onboarding consistently produced the highest hazard ratios across all applications
        - Churn risk declined sharply after the first week of activity
        - Several applications continued showing secondary churn-risk periods beyond onboarding

        This confirmed that churn is heavily time-dependent rather than randomly distributed across users.
        """)

    st.write("""
            However, hazard modeling only explains *when* churn risk becomes elevated.

            It does not explain which behavioral engagement windows contribute most strongly
            to churn prediction or how different model architectures interpret these signals.

            To address this, SHAP explainability analysis was introduced to decompose model behavior
            at the feature level and identify which temporal engagement periods carried the strongest predictive influence.
            """)
    
    # --------------------------------------------------------  Feature Importance   -------------------------------------------------------
    # STRUCTURE:    project | app name | model name | DF
    st.markdown("<div id='shap-explainability'></div>", unsafe_allow_html=True)
    st.header("SHAP Explainability")
    st.write("""
        Following the Cox analysis, the next objective was to determine
        which behavioral time windows were actually driving churn prediction.

        While Cox modeling validated that onboarding carried elevated churn risk,
        it could not explain how different engagement periods contributed to model decisions.

        SHAP (SHapley Additive exPlanations) was therefore used to measure
        feature-level contribution strength across temporal behavioral signals.
        """)
    st.latex(r'''
            f(x) = \phi_0 + \sum_{i=1}^{M}\phi_i
            ''')
    st.caption("""
        SHAP decomposes individual predictions into additive feature contributions,
        allowing temporal behavioral signals to be interpreted directly.
        """)
    st.markdown("""
        #### Why SHAP Was Used

        - Quantify feature-level influence on churn prediction
        - Interpret temporal behavioral windows directly
        - Compare behavioral learning patterns across ANN and XGBoost architectures
        - Identify whether churn signals concentrate early or persist longitudinally
        """)

    st.markdown("""
            #### Model Behavior Differences

            **ANN (Kernel SHAP)**
            - Captured distributed behavioral influence across longer lifecycle horizons
            - Revealed recurring engagement and re-engagement structures over time
            - Better suited for longitudinal behavioral interpretation

            **XGBoost (Tree SHAP)**
            - Concentrated predictive weight heavily within Days 1–7
            - Prioritized immediate engagement decay signals
            - Better suited for early churn detection tasks
            """)

    st.markdown("""
            #### Technical Tradeoffs

            Kernel SHAP for ANN models was computationally expensive due to repeated inference sampling.

            To improve scalability:
            - background distributions were strategically sampled
            - representative behavioral structure was preserved
            - computational overhead was reduced without materially degrading explanation quality
            """)
    st.markdown("""
        #### SHAP Implementation
        """)

    st.code("""
    if 'ann' in combo_key.lower():
        background_data = shap.sample(X_train_SHAP, 50)

        model = self.build_ANN_model()

        model.fit(
            X_train_SHAP,
            Y_train_SHAP,
            epochs=50,
            verbose=0
        )

        explainer = shap.KernelExplainer(
            model.predict,
            background_data
        )

    else:
        model = xgb.XGBClassifier(
            n_estimators=100,
            tree_method='hist',
            random_state=4
        )

        model.fit(X_train_SHAP, Y_train_SHAP)

        explainer = shap.TreeExplainer(model)
    """)

    app_tab_fi = st.tabs(apps)
    for tab, app in zip(app_tab_fi, apps): 
        with tab:
            # ANN CHART
            filtered_ann = afi[app]['ANN']['Churners'].copy()
            filtered_ann['Days'] = filtered_ann['Days'].str.replace('Day_', '', regex=False).astype(int)
            filtered_ann['SHAP Values'] = filtered_ann['SHAP Values'].str[0]
            filtered_ann['SHAP Values'] = filtered_ann['SHAP Values'].astype(float)
            filtered_ann.reset_index(drop=True)
            max_days = filtered_ann['Days'].max()
            min_days = filtered_ann['Days'].min()
            max_shap_values = filtered_ann['SHAP Values'].max()
            ann_charts = alt.Chart(filtered_ann).mark_bar(opacity=0.5).encode(
                            alt.X(
                                'Days:Q',
                                scale = alt.Scale(domain=[min_days-1,max_days+1]),
                                axis=alt.Axis(tickMinStep=1),
                                bin=alt.Bin(maxbins = 100)
                                ),
                            alt.Y(
                                'SHAP Values:Q', 
                                scale=alt.Scale(domain=[0, max_shap_values+0.002]), 
                                stack = None
                                )).properties(
                                            title={
                                            'text':'ANN Model',
                                            'anchor': 'middle'}).interactive()
            # XGB CHART
            filtered_xgb = afi[app]['XGB']['Churners'].copy()
            filtered_xgb['Days'] = filtered_xgb['Days'].str.replace('Day_', '', regex=False).astype(int)
            max_days_xgb = filtered_xgb['Days'].max()
            min_days_xgb = filtered_xgb['Days'].min()
            max_shap_values_xgb = filtered_xgb['SHAP Values'].max()
            xgb_charts = alt.Chart(filtered_xgb).mark_bar(opacity=0.5).encode(
                            alt.X(
                                'Days:Q',
                                scale = alt.Scale(domain=[min_days_xgb-1,max_days_xgb+1]),
                                axis=alt.Axis(tickMinStep=1),
                                bin=alt.Bin(maxbins = 100)
                                ),
                            alt.Y(
                                'SHAP Values:Q', 
                                scale=alt.Scale(domain=[0, max_shap_values_xgb+0.2]), 
                                stack = None
                                )).properties(
                                    title={
                                        'text':'XGBoost Model',
                                        'anchor': 'middle'}).interactive()
            st.altair_chart(ann_charts, use_container_width=True)
            st.altair_chart(xgb_charts, use_container_width=True)

    st.write("""
        ### Key Findings

        SHAP analysis revealed that the two model architectures learned fundamentally different temporal structures:

        - XGBoost concentrated predictive importance heavily within early engagement periods (Days 1–7)
        - ANN models distributed importance across broader behavioral horizons extending toward Day 30
        - Multiple recurring behavioral windows consistently contributed to churn prediction beyond onboarding

        These results reinforced that churn is not driven by a single event,
        but by evolving behavioral patterns across the lifecycle.
        """)

    st.write("""
            This distinction directly informed downstream system design:

            - XGBoost was prioritized for early churn classification
            - ANN-derived patterns were used for longitudinal retention analysis

            However, SHAP still does not explain how users behaviorally transition
            through these high-risk periods over time.

            To investigate whether these behavioral windows represented recoverable disengagement
            or irreversible churn trajectories, retention curve analysis was performed next.
            """)

# --------------------------------------------------------  Retention Curves ------------------------------------------------------- 
    st.markdown("<div id='churners-retention-windows'></div>", unsafe_allow_html=True)
    st.header("Churners Retention Windows")
    st.write('''
                SHAP analysis identified several recurring high-risk behavioral windows across the lifecycle.

                However, feature importance alone could not determine whether these periods represented:
                - irreversible churn trajectories
                - or temporary disengagement states that could still be recovered

                To investigate this, cohort-based retention curve analysis was performed to measure
                how users behaviorally evolved after entering these critical engagement windows.
             ''')
    app_tab_rc = st.tabs(apps)
    for tab, app in zip(app_tab_rc, apps):              
        with tab: 
            # ---------------------------------------------------------------------------------------Cohort Levels for ONBOARDING
            #  --- Day 1 ---
            cohort_charts = []
            for level in retention_Day1[app].keys():
                df_on = retention_Day1[app][level]
                df_on['Activity Level'] = level
                df_on['Retention'] = df_on[level]
                df_on = df_on.drop(level,axis =1)
                df_on['Days'] = df_on.index.str.replace('Day_', '', regex=False).astype(int)
                df_on = df_on.reset_index(drop=True)
                cohort_charts.append(df_on)
            # Combine everything
            combined_df_on = pd.concat(cohort_charts)

            layered_area_chart = alt.Chart(combined_df_on).mark_area(opacity=0.3).encode(
                    x=alt.X('Days:Q', title='Cohort Period (Days)'),
                    y=alt.Y('Retention:Q', title='Retention Rate').stack(None),
                    color=alt.Color('Activity Level:N', title='Cohort'),
                    tooltip=['Activity Level', 'Days', 'Retention']
                ).interactive()

            st.markdown("#### Onboarding Retention Decay")

            st.write("""
                Retention curves were first analyzed across onboarding cohorts
                to evaluate how quickly users disengaged after initial activation.

                This allows early-stage retention decay to be compared across
                different engagement intensity groups.
                """)
            st.altair_chart(layered_area_chart)


            # -------------------------------------------------------Critical retention windows
            charts = []
            #  --- Day 1 ---
            day1_df = retention_Day1[app]['high']
            day1_df['Retention'] = day1_df['high']
            day1_df = day1_df.drop('high', axis=1)
            day1_df['Days'] = day1_df.index.str.replace('Day_', '', regex=False).astype(int)
            day1_df = day1_df.reset_index(drop=True)

            # --- Critical Days ---
            app_crit_dict = Retention_TFD[app]

            for critical_day, activity_levels in app_crit_dict.items():
                high_df = activity_levels['high']  #.reset_index()
                high_df['Retention'] = high_df['high']
                high_df['Days'] = high_df.index.str.replace('Day_', '', regex=False).astype(int)
                high_df = high_df.drop(['high'], axis=1)
                high_df['Cohort'] = critical_day
                high_df = high_df.reset_index(drop=True)
                charts.append(high_df)

            # Combine everything
            combined_df = pd.concat(charts)

            # --- Chart ---
            line_t = alt.Chart(combined_df).mark_line().encode(
                    x=alt.X('Days:Q', title='Cohort Period (Days)'),
                    y=alt.Y('Retention:Q', title='Retention Rate'),
                    color=alt.Color('Cohort:N', title='Cohort'),
                    tooltip=['Cohort', 'Days', 'Retention']
                )

            points_t = alt.Chart(combined_df).mark_circle(size=30).encode(
                    x='Days:Q',
                    y='Retention:Q',
                    color='Cohort:N'
                )
            
            st.markdown("#### Behavioral Re-Engagement Patterns")

            st.write("""
                    Retention trajectories were then aligned to SHAP-identified
                    high-risk behavioral windows to evaluate whether users exhibited
                    recoverable engagement behavior before churn.
                    """)
            chart_t = (line_t + points_t).interactive()
            st.altair_chart(chart_t, width='stretch')
            
            app_notes = {
            'Other Apps':
                "- Re-engagement spike around Day 16\n"
                "- Final engagement spike between Days 36–47",

            'Facebook':
                "- Re-engagement spike around Days 15–16\n"
                "- Final engagement spike between Days 37–52",

            'TikTok':
                "- Re-engagement spike around Day 17\n"
                "- Final engagement spike between Days 36–52",

            'YouTube':
                "- Minimal re-engagement behavior observed",

            'WhatsApp':
                "- Re-engagement spike around Day 15\n"
                "- Final engagement spike between Days 36–47",

            'Helakuru':
                "- Re-engagement spike around Day 16\n"
                "- Final engagement spike between Days 36–47"
        }
            st.subheader('Observed Retention Structure')
            st.markdown(app_notes[app])

            # ------------------------------------------------------- Retention Decay Rates Differences of 1 WEEK - 2 WEEKS


            # Aggregate the Tables   
            B = round((TWO_week_window_DIFF.groupby('App Name')['Average Percent Difference']).mean(),2)
            A = round((ONE_week_window_DIFF.groupby('App Name')['Average Percent Difference']).mean(),2)
            C = pd.concat([A,B], axis =1)
            C['Average Percent Difference (week 1)'] = C.iloc[:,0]
            C['Average Percent Difference (week 2)'] = C.iloc[:,1]
            C = C.drop('Average Percent Difference', axis = 1)
            C['Differences'] = C['Average Percent Difference (week 2)'] - C['Average Percent Difference (week 1)']
            st.write(C)
    st.write("""
        ### Key Findings

        Retention analysis revealed that churn behavior was not a smooth monotonic decline.

        Instead:
        - users frequently displayed temporary re-engagement before permanent churn
        - several applications showed highly repeatable recovery windows around Days 15–19
        - secondary late-stage engagement spikes emerged between Days 36–55

        These findings suggest that churn develops progressively rather than instantaneously.
        """)

    st.write("""
            This distinction became critical for system design.

            Later-stage churners often remained behaviorally recoverable through timed intervention strategies.

            However, onboarding churners behaved differently:
            many disengaged too rapidly for traditional retention workflows to respond effectively.

            This created the need for a dedicated early-stage prediction system capable of identifying
            churn risk within the first days of activity before disengagement stabilized.
            """)
    
    # -------------------------------------------------------------------------------------  Early Prediction System -------------------------------------------------------
    st.markdown("<div id='real-time-churn-monitoring'></div>", unsafe_allow_html=True)
    st.header("Real-Time Churn Monitoring")
    st.write("""
        Retention curve analysis revealed a critical behavioral distinction:

        - later-stage churners often exhibited recoverable re-engagement behavior
        - onboarding churners disengaged too rapidly for traditional retention workflows to respond effectively

        This created the need for a dedicated early-stage detection system capable of identifying
        churn risk within the first days of user activity before disengagement stabilized.
        """)

    st.write("""
            The primary objective was therefore not maximizing overall classification accuracy,
            but maximizing operational intervention value under severe class imbalance conditions.

            This shifted model optimization toward:
            - minimizing missed churners (false negatives)
            - identifying churn risk as early as possible
            - preserving sufficient precision for scalable intervention workflows
            """)
    st.markdown("<div id='threshold-optimization'></div>", unsafe_allow_html=True)
    st.write("""
        Default probability thresholds proved too conservative for early churn interception.

        Since onboarding churners disengage rapidly, delayed detection substantially reduces
        recovery opportunity. Classification thresholds were therefore intentionally shifted
        away from standard decision boundaries to improve early-stage churn sensitivity.
        """)

    st.write("""
            This introduced an intentional tradeoff:

            - lower thresholds increased recall
            - false negatives decreased substantially
            - churn became detectable earlier in the lifecycle
            - false positive rates increased moderately as a cost of earlier intervention

            The final thresholding strategy prioritized operational retention utility
            rather than naive classification accuracy.
            """)

    st.code("""
    threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5]

    for threshold in threshold_list:

        xgb_model = self.XGB_model().fit(X_train, Y_train)

        y_prob = xgb_model.predict_proba(X_test)[:, 1]

        y_pred = (y_prob >= threshold).astype(int)
    """)

    # -------------------------------------------------------------------------------------
    # Evaluation Metrics
    # -------------------------------------------------------------------------------------

    st.subheader("Evaluation Framework Under Class Imbalance")

    st.write("""
            Traditional accuracy metrics became unreliable due to severe churn imbalance.

            Most users did not churn, meaning models could achieve artificially high accuracy
            while still failing to identify at-risk users effectively.

            Evaluation was therefore centered around metrics better aligned
            with intervention-system performance.
            """)

    st.markdown("""
            #### Recall
            - maximize early churn detection sensitivity

            #### False Negatives
            - minimize missed at-risk users

            #### PR AUC
            - evaluate minority-class separability under imbalance

            #### F2 Score
            - overweight recall relative to precision
            """)

    st.latex(r'''
        F_{\beta} =
        (1+\beta^2)
        \frac{Precision \cdot Recall}
        {(\beta^2 \cdot Precision) + Recall}
    ''')

    st.caption("""
            F2 weighting emphasized recall sensitivity to prioritize early churn interception.
            """)

    app_tab_eps = st.tabs(apps)
    for tab, app in zip(app_tab_eps, apps):              
        with tab: 
            st.write(f'### {app} - Early Churn Prediction Tradeoffs')
            filtered_df = ecps[ecps['App_name']==app] 
            threshold_dropdown = st.selectbox('Select Threshold',
                                            options = sorted(filtered_df['Threshold'].unique()),
                                            key = f'{app}_threshold')
            chart_df = filtered_df[filtered_df['Threshold']== threshold_dropdown]
            precision_chart = (
                        alt.Chart(chart_df)
                        .mark_circle(opacity=0.75)
                        .encode(
                                alt.X(
                                    "Days:Q", 
                                    title="Days",
                                    sort = 'ascending',
                                    scale=alt.Scale(domain=[0, 15], nice=False)
                                    ),
                                alt.Y(
                                    "Precision:Q",
                                    sort = 'ascending',
                                    title="Precision Score",
                                    ),
                                color=alt.Color('Model:N', title='Model'
                                                ),
                                tooltip = ['App_name',
                                            'Model',
                                            'Threshold',
                                            'Days',
                                            'Precision',
                                            'False Negatives',
                                            'recall',
                                            'F2 Score',
                                            'PR AUC'],
                                size=alt.Size(
                                            "Precision:Q", 
                                            scale=alt.Scale(range=[20,100]), 
                                            legend = None
                                        )
                            ).properties(
                                        width=700, 
                                        height=350, 
                                        title = f'Precision Across Days (Threshold = {threshold_dropdown})'
                                        )
                        )
                    
            # Bottom panel is a bar chart of weather type
            FN_chart = (
                        alt.Chart(chart_df)
                        .mark_bar()
                        .encode(
                            # Horizontal length of bars
                            x=alt.X(
                                "False Negatives:Q",
                                title='False Negatives (%)',
                                scale=alt.Scale(domain=[0, chart_df['False Negatives'].max() + 5])
                                ),

                            # Vertical grouping
                            y=alt.Y(
                                "Days:O",
                                sort = 'descending',
                                title='Days'
                            ),

                            # Side-by-side bars for models
                            yOffset='Model:N',

                            color=alt.Color(
                                'Model:N',
                                title='Model'
                            ),

                            tooltip=[
                                'App_name',
                                'Model',
                                'Threshold',
                                'Days',
                                'False Negatives',
                                'Precision',
                                'recall',
                                'F2 Score',
                                'PR AUC'
                            ]
                        )
                        .properties(
                            width=700,
                            height=400,
                            title=f'False Negatives Across Days (Threshold = {threshold_dropdown})'
                        )
                    )
            
            # Display
            st.altair_chart(alt.vconcat(precision_chart,FN_chart).resolve_scale(color='independent'), use_container_width=True)      
    st.write("""
        ### Key Findings

        The final detection system consistently identified elevated churn risk
        within approximately 3–4 days of user activity across applications.

        Several important modeling tradeoffs emerged:

        - lower thresholds improved early churn interception
        - minimizing false negatives materially improved intervention opportunity
        - moderate precision loss was acceptable due to the higher business cost of missed churners
        - XGBoost consistently outperformed ANN models for immediate churn classification tasks
        """)

    st.write("""
            These findings reinforced the broader project conclusion:

            churn prediction systems should be optimized around intervention timing and behavioral recoverability
            rather than static classification accuracy alone.

            The resulting framework transforms churn modeling from a retrospective reporting problem
            into a real-time intervention system capable of adapting across the customer lifecycle.
            """)
        


    # ------------------------------------  Project Pipeline Archecture  ---------------------------------------------------------
    st.markdown("<div id='pipeline-architecture'></div>", unsafe_allow_html=True)
    st.header("Pipeline Architecture")
    st.markdown("""
                The pipeline was designed around three principles:

                ##### 1. Temporal Interpretability
                            Predictions must explain *when* churn risk emerges.

                ##### 2. Operational Recall
                            Missing churners was considered more costly than over-flagging users.

                ##### 3. Behavioral Generalization
                            Models needed to identify reusable engagement patterns across multiple applications rather than overfit to platform-specific activity distributions.
            """)

    graph_shap = graphviz.Digraph()
    graph_shap.edge('DataFrame Prep', 'Multi-Transformations', fillcolor = 'black', fontcolor = 'black', arrowhead = 'inv')
    graph_shap.edge('Multi-Transformations', 'Model Fittings', fillcolor = 'black', fontcolor = 'white', arrowhead = 'inv')
    graph_shap.edge('Model Fittings', 'Best Transformer / Model Combination', fillcolor = 'black', fontcolor = 'white', arrowhead = 'inv')
    graph_shap.edge('Best Transformer / Model Combination',
                    'SHAP Feature Analysis', fillcolor = 'black', fontcolor = 'white', arrowhead = 'inv')
    
    st.graphviz_chart(graph_shap)

    st.write('''
            ##### DataFrame Prep
              - Cleaning
              - Sampling
              - Encoding
              - Splitting

            ##### Multi-Transformations
              - Winsorization IQR & MAD
              - Log Squareroot
              - Inverse Hyperbolic Sine
              - Yeo Johnson
              - Power
              - Box Cox

            ##### Code: 
            Transformations were applied only to non-zero behavioral activity
            to preserve inactivity structure while stabilizing skewed distributions.
            
                mask_train = x_train[col] > 0
                train_input = x_train.loc[mask_train, [col]]
                transformed_train = transformer.fit_transform(train_input)
                x_train.loc[mask_train, col] = (
                    transformed_train[col].values.flatten()
                )
            ##### Model Fittings
              - Artificial Neural Network
              - XGBoost
              ''')
    st.markdown('#### Cox-Hazard Ratios')
    graph_cox = graphviz.Digraph()
    graph_cox.edge('DataFrame Prep',
                   'Cox Proportional Hazards Regression Model', arrowhead = 'inv')
    graph_cox.edge('Cox Proportional Hazards Regression Model',
                   'Cox Feature Summary', arrowhead = 'inv')
    st.graphviz_chart(graph_cox)

    st.markdown('#### Early Churn Prediction System')
    graph_early = graphviz.Digraph()
    graph_early.edge('DataFrame Prep', 'Multi-Transformations', arrowhead = 'icurve', fontname = 'Helvetica')
    graph_early.edge('Multi-Transformations', 'Model Fittings', arrowhead = 'icurve', fontname = 'Helvetica')
    graph_early.edge('Model Fittings', 'Best Transformer / Model Combination', arrowhead = 'icurve', fontname = 'Helvetica')
    graph_early.edge('Best Transformer / Model Combination',
                     'SHAP Feature Analysis', arrowhead = 'icurve', fontname = 'Helvetica')
    st.graphviz_chart(graph_early)


    st.markdown('#### Retention Curves')
    graph_retention = graphviz.Digraph()
    graph_retention.edge('DataFrame Prep', 'First Day Activity Levels')
    graph_retention.edge('First Day Activity Levels', 'Onboarding Cohorts')
    graph_retention.edge('Onboarding Cohorts', 'Onboarding Retention Summary')
    #graph_retention_crit = graphviz.Digraph()
    graph_retention.edge('DataFrame Prep', 'Critical Day Shift Activity Levels')
    graph_retention.edge('Critical Day Shift Activity Levels', 'Critical Days Cohorts')
    graph_retention.edge('Critical Days Cohorts', 'Critical Days Retention Summary')
    graph_retention.edge('Critical Days Retention Summary','Retention Shift Differnces')
    st.graphviz_chart(graph_retention , use_container_width=True)
