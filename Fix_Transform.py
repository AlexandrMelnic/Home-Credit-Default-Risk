import pandas as pd
import numpy as np

class No_Out():
    
    def __init__(self, df):
        
        self.df = df
    
    def social_circ(self):
        
        # For OBS_30/60 we replace all values greater than 100 with missin values:

        self.df["OBS_30_CNT_SOCIAL_CIRCLE"] = self.df["OBS_30_CNT_SOCIAL_CIRCLE"].apply(lambda x: np.nan if x >= 100 else x)
        self.df["OBS_60_CNT_SOCIAL_CIRCLE"] = self.df["OBS_60_CNT_SOCIAL_CIRCLE"].apply(lambda x: np.nan if x >= 100 else x)
        
        # For DEF_30/60 we replace all values greater than 10 with missin values:
        
        self.df["DEF_30_CNT_SOCIAL_CIRCLE"] = self.df["DEF_30_CNT_SOCIAL_CIRCLE"].apply(lambda x: np.nan if x >= 10 else x)
        self.df["DEF_60_CNT_SOCIAL_CIRCLE"] = self.df["DEF_60_CNT_SOCIAL_CIRCLE"].apply(lambda x: np.nan if x >= 10 else x)
     
    def enquires_req(self):
        
        self.df["AMT_REQ_CREDIT_BUREAU_QRT"].replace(261, np.nan, inplace = True)
    
    def economics_inc(self):
        
        mm = self.df["AMT_INCOME_TOTAL"].max()
        self.df.loc[self.df["AMT_INCOME_TOTAL"] == mm, "AMT_INCOME_TOTAL"] = np.nan
        
    def cars(self):
        
        self.df.loc[self.df["OWN_CAR_AGE"] > 65, "OWN_CAR_AGE"] = np.nan
    
    def days_emp(self):
        
        out = self.df["DAYS_EMPLOYED"].max()
        self.df.loc[self.df["DAYS_EMPLOYED"] == out, "DAYS_EMPLOYED"] = np.nan
        
    def Drop_Out(self):
        
        pd.set_option('mode.chained_assignment', None)
        
        self.social_circ()
        self.enquires_req()
        self.economics_inc()
        self.cars()
        self.days_emp()

        return self.df

class New_Attributes():
    
    def __init__(self, df):
        
        self.df = df
        self.final_df = df[["SK_ID_CURR"]]
    
    '''
    It generates a column which says whether more thant the 50% of the info are present (YES) or 
    are missing (NO).
    '''
    
    def house(self):
        
        # Generate the DF with only the info about house:
        
        home_test = self.df[["APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG", 
                   "COMMONAREA_AVG", "ELEVATORS_AVG", "ENTRANCES_AVG", "FLOORSMAX_AVG", "FLOORSMIN_AVG", 
                   "LANDAREA_AVG", "LIVINGAPARTMENTS_AVG", "LIVINGAREA_AVG", "NONLIVINGAPARTMENTS_AVG", 
                   "NONLIVINGAREA_AVG", "APARTMENTS_MODE", "BASEMENTAREA_MODE", "YEARS_BEGINEXPLUATATION_MODE",
                   "YEARS_BUILD_MODE", "COMMONAREA_MODE", "ELEVATORS_MODE", "ENTRANCES_MODE", "FLOORSMAX_MODE", 
                   "FLOORSMIN_MODE", "LANDAREA_MODE", "LIVINGAPARTMENTS_MODE", "LIVINGAREA_MODE", 
                   "NONLIVINGAPARTMENTS_MODE", "NONLIVINGAREA_MODE", "APARTMENTS_MEDI", "BASEMENTAREA_MEDI", 
                   "YEARS_BEGINEXPLUATATION_MEDI", "YEARS_BUILD_MEDI", "COMMONAREA_MEDI", "ELEVATORS_MEDI", 
                   "ENTRANCES_MEDI", "FLOORSMAX_MEDI", "FLOORSMIN_MEDI", "LANDAREA_MEDI", "LIVINGAPARTMENTS_MEDI", 
                   "LIVINGAREA_MEDI", "NONLIVINGAPARTMENTS_MEDI", "NONLIVINGAREA_MEDI", "FONDKAPREMONT_MODE", 
                   "HOUSETYPE_MODE", "TOTALAREA_MODE", "WALLSMATERIAL_MODE", "EMERGENCYSTATE_MODE"]]
        
        # New variable:
        
        n_cols = home_test.shape[1]
        self.final_df["HOUSE_INFO"] = (home_test.isnull().sum(axis = 1)/n_cols).apply(lambda x: "No" if x >= 0.5 else "Yes")
        
    '''
    It makes a summary about the most importa doc (document 3) and the other: 2 columns one to say if any document 
    different from the 3rd is present and another whch say if none document is present (the third column is 
    already present in the df, it is DOC_3).
    '''
    
    def flags(self):
        
        # Generate the DF
        
        flags_test = self.df[["FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3", "FLAG_DOCUMENT_4", "FLAG_DOCUMENT_5", "FLAG_DOCUMENT_6", 
               "FLAG_DOCUMENT_7", "FLAG_DOCUMENT_8", "FLAG_DOCUMENT_9", "FLAG_DOCUMENT_10", "FLAG_DOCUMENT_11", 
               "FLAG_DOCUMENT_12", "FLAG_DOCUMENT_13", "FLAG_DOCUMENT_14", "FLAG_DOCUMENT_15", "FLAG_DOCUMENT_16", 
               "FLAG_DOCUMENT_17", "FLAG_DOCUMENT_18", "FLAG_DOCUMENT_19", "FLAG_DOCUMENT_20", "FLAG_DOCUMENT_21"]].copy()
        
        # Create the new variables we need:
        
        flags_test.insert(20, "DOC_6_OR_8", 0)
        flags_test.insert(21, "ANY_OTHER_DOCS", 0)
        flags_test.insert(22, "NO_DOCUMENTS", 0)
        
        flags_test.loc[(flags_test['FLAG_DOCUMENT_6'] == 1) | (flags_test['FLAG_DOCUMENT_8'] == 1), 'DOC_6_OR_8'] = 1

        flags_test.loc[(flags_test.iloc[:, [0, 2, 3, 5, 7, 8, 9 ,10, 11, 12, 13, 14, 15, 16, 17, 18, 19]] == 1).any(axis = 1) &
                 (flags_test.iloc[:, [1,4,6]] != 1).all(axis = 1), 'ANY_OTHER_DOCS']= 1
        
        flags_test.loc[(flags_test.iloc[:, :-3] == 0).all(axis = 1), "NO_DOCUMENTS"] = 1
        
        # Drop no needed columns:
        
        flags_test.drop(columns = ['FLAG_DOCUMENT_2', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
                      'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_10',   'FLAG_DOCUMENT_11', 'FLAG_DOCUMENT_12', 
                      'FLAG_DOCUMENT_13', 'FLAG_DOCUMENT_14', 'FLAG_DOCUMENT_15', 'FLAG_DOCUMENT_16', 'FLAG_DOCUMENT_17', 
                      'FLAG_DOCUMENT_18', 'FLAG_DOCUMENT_19','FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21'],
          inplace = True)
        
        flags_test.columns = ['FLAG_DOC_3', 'FLAG_DOC_6_OR_8', 'FLAG_ANY_OTHER_DOCS', 'FLAG_NO_DOCUMENTS']
        flags_test = flags_test.astype('uint8')
        
        # Column Adjustment:
        
        flags_test.insert(3, "FLAG_ANY_BUT_3", 0)
        flags_test.loc[(flags_test["FLAG_DOC_6_OR_8"] == 1) & (flags_test["FLAG_ANY_OTHER_DOCS"] == 0), "FLAG_ANY_BUT_3"] = 1
        flags_test.loc[(flags_test["FLAG_DOC_6_OR_8"] == 0) & (flags_test["FLAG_ANY_OTHER_DOCS"] == 1), "FLAG_ANY_BUT_3"] = 1
        flags_test.loc[(flags_test["FLAG_DOC_6_OR_8"] == 1) & (flags_test["FLAG_ANY_OTHER_DOCS"] == 1), "FLAG_ANY_BUT_3"] = 1
        
        flags_test = flags_test.drop(columns=["FLAG_DOC_6_OR_8", "FLAG_ANY_OTHER_DOCS"], axis = 1)
        flags_test = flags_test[["FLAG_DOC_3", "FLAG_ANY_BUT_3", "FLAG_NO_DOCUMENTS"]]
        flags_test = flags_test.drop("FLAG_DOC_3", axis = 1)
        
        # Add to final df:
        
        self.final_df = pd.concat([self.final_df, flags_test], axis = 1)
        
    '''
    Calculate the total number of requests yearly.
    '''
    
    def enquires(self):
        
        # Generate the DF:
        
        enquires_test = self.df[["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
                 "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]]
        
        # The new variable creation:
        
        enquires_test["TOT_ENQUIRES_YEARLY"] = enquires_test.iloc[: , range(6)].sum(axis = 1)
        enquires_test.loc[(enquires_test.iloc[: , range(6)].isnull()).any(axis = 1), "TOT_ENQUIRES_YEARLY"] = np.nan
        
        # Add to DF:
        
        self.final_df = pd.concat([self.final_df, enquires_test[["TOT_ENQUIRES_YEARLY"]]], axis = 1)
        
        
    '''
    Get some cool economics info: aproximal term (DURATION) and percent of the incomes which goes away to repay
    the loan ("AMT_ANNUITY_INCOMES_RATIO")
    '''
    
    def socials(self):
        
        social_test = self.df[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
                 'DEF_60_CNT_SOCIAL_CIRCLE']]
        
        # New column:
        
        social_test["SOCIAL_DEFAUL_RATIO_60"]=social_test["DEF_60_CNT_SOCIAL_CIRCLE"] /social_test["OBS_60_CNT_SOCIAL_CIRCLE"]
    
        # Transform 0 in missing values:
        
        social_test.loc[social_test["OBS_60_CNT_SOCIAL_CIRCLE"] == 0, "OBS_60_CNT_SOCIAL_CIRCLE"] = np.nan
        social_test.loc[(social_test["OBS_60_CNT_SOCIAL_CIRCLE"] == np.nan) & 
                        (social_test["DEF_60_CNT_SOCIAL_CIRCLE"] == 0), "DEF_60_CNT_SOCIAL_CIRCLE"] = np.nan
        
        # Add to final DF:
        
        self.final_df = pd.concat([self.final_df, social_test["SOCIAL_DEFAUL_RATIO_60"]], axis = 1)
    
    def economics(self):
        
        # Generate the DF:
        
        economics_test = self.df[["AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE"]]
        
        # New varaibles:
        
        economics_test["DURATION"] = economics_test["AMT_CREDIT"] / economics_test["AMT_ANNUITY"]
        economics_test["AMT_ANNUITY_INCOMES_RATIO"] = economics_test["AMT_ANNUITY"] / economics_test["AMT_INCOME_TOTAL"]
        economics_test["AMT_INCOME_TOTAL_LOG"] = np.log(economics_test["AMT_INCOME_TOTAL"])
        economics_test["AMT_ANNUITY_LOG"] = np.log(economics_test["AMT_ANNUITY"])
        economics_test["AMT_GOODS_PRICE_LOG"] = np.log(economics_test["AMT_GOODS_PRICE"])
        
        economics_test.drop(columns = ["AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT", "AMT_GOODS_PRICE"], 
                            inplace = True)
        
        # Add:
        
        self.final_df = pd.concat([self.final_df, economics_test], axis = 1)
        
    '''
    Call all the Above functions
    '''
    
    def Generate(self):
        
        self.house()
        self.flags()
        self.enquires()
        self.socials()
        self.economics()

    '''
    It transforms in year variables expressed in days (drop the original) + convert some categorical columns 
    wrongly read as numbers.
    '''
    
    def Replace_and_Convert(self):
        
        # Transform in years
        
        self.df["YEARS_BIRTH"] = self.df["DAYS_BIRTH"].apply(lambda x: -int(x/365)) 
        self.df.drop(columns = ["DAYS_BIRTH"],inplace = True)
        
        self.df['YEARS_LAST_PHONE_CHANGE'] = self.df['DAYS_LAST_PHONE_CHANGE'].apply(lambda x: -x/365)
        self.df.drop(columns = ["DAYS_LAST_PHONE_CHANGE"],inplace = True)
        
        self.df["YEARS_ID_PUBLISH"] = self.df["DAYS_ID_PUBLISH"].apply(lambda x: -x/365)
        self.df["YEARS_REGISTRATION"] = self.df["DAYS_REGISTRATION"].apply(lambda x: -x/365)
        self.df.drop(columns = ["DAYS_ID_PUBLISH", "DAYS_REGISTRATION"],inplace = True)
        
        self.df["YEARS_EMPLOYED"] = self.df["DAYS_EMPLOYED"].apply(lambda x: -x/365 + 1)
        self.df["YEARS_EMPLOYED"] = self.df.loc[self.df["YEARS_EMPLOYED"] < 0, "YEARS_EMPLOYED"] = np.nan
        self.df["YEARS_EMPLOYED_LOG"] = np.log(self.df["YEARS_EMPLOYED"])
        self.df.drop(columns = ["DAYS_EMPLOYED", "YEARS_EMPLOYED"],inplace = True)
        
        # Make them Categorical:
        
        self.df.insert(6, "HAS_CHILDREN", "No")
        self.df.loc[self.df["CNT_CHILDREN"] > 0, "HAS_CHILDREN"] = "Yes"
        self.df.drop(columns = ["CNT_CHILDREN"],inplace = True)
        
        cols = ['FLAG_EMP_PHONE','FLAG_PHONE','FLAG_WORK_PHONE', 'FLAG_EMAIL']
        for c in cols:   
            self.df.loc[self.df[c] == 1, c] = "Yes"
            self.df.loc[self.df[c] == 0, c] = "No"
        
        self.df.loc[self.df['LIVE_CITY_NOT_WORK_CITY'] == 1, 'LIVE_CITY_NOT_WORK_CITY'] = "Different"
        self.df.loc[self.df['LIVE_CITY_NOT_WORK_CITY'] == 0, 'LIVE_CITY_NOT_WORK_CITY'] = "Same"
    
        self.df.loc[self.df['LIVE_REGION_NOT_WORK_REGION'] == 1, 'LIVE_REGION_NOT_WORK_REGION'] = "Different"
        self.df.loc[self.df['LIVE_REGION_NOT_WORK_REGION'] == 0, 'LIVE_REGION_NOT_WORK_REGION'] = "Same"
        
        self.df.loc[self.df['REG_CITY_NOT_LIVE_CITY'] == 1, 'REG_CITY_NOT_LIVE_CITY'] = "Different"
        self.df.loc[self.df['REG_CITY_NOT_LIVE_CITY'] == 0, 'REG_CITY_NOT_LIVE_CITY'] = "Same"
        
        self.df.loc[self.df['REG_CITY_NOT_WORK_CITY'] == 1, 'REG_CITY_NOT_WORK_CITY'] = "Different"
        self.df.loc[self.df['REG_CITY_NOT_WORK_CITY'] == 0, 'REG_CITY_NOT_WORK_CITY'] = "Same"
        
        self.df.loc[self.df['REG_REGION_NOT_LIVE_REGION'] == 1, 'REG_REGION_NOT_LIVE_REGION'] = "Different"
        self.df.loc[self.df['REG_REGION_NOT_LIVE_REGION'] == 0, 'REG_REGION_NOT_LIVE_REGION'] = "Same"
        
        self.df.loc[self.df['REG_REGION_NOT_WORK_REGION'] == 1, 'REG_REGION_NOT_WORK_REGION'] = "Different"
        self.df.loc[self.df['REG_REGION_NOT_WORK_REGION'] == 0, 'REG_REGION_NOT_WORK_REGION'] = "Same"
    
    '''
    Merge the original DF (modified and the new columns created)
    '''
    
    def Complete_Data(self):
        
        pd.set_option('mode.chained_assignment', None)
        
        # Apply the mod:
        
        self.Replace_and_Convert()
        self.Generate()
        
        # Create final df:
        
        transformed_df = self.df.merge(self.final_df, on = "SK_ID_CURR")
        
        return transformed_df
    
  