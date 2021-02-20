'''
Librires
'''

import pandas as pd
import numpy as np
from collections import Counter

import warnings

import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.preprocessing import scale
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

'''
Class with functions utilized in both DF
'''

class utils():
    
    def imputer(self, df, col, method = 'median'):
        
        imp = SimpleImputer(missing_values = np.nan, strategy = method)
        df[col] = imp.fit_transform(df[[col]])
    
    
    '''
    Function that checks for correlations and delete one of the two features if the correlation is above a certain threshold t. Takes in input the 
    dataframe and the threshold t and returns the new dataframe.
    '''
    def remove_collinearity (self, df, t):
        df_x = df.iloc[:307511]
        cor_mat = df_x.corr().apply(abs)
        corr_columns = []
        for i in range(len(cor_mat)):
            for j in range(i+1, len(cor_mat)):
                if cor_mat.iloc[i,j]>t and i != j:
                    corr_columns.append((cor_mat.index[i], cor_mat.index[j]))    
        corr_columns = list(set(corr_columns))
        del_columns = [col[0] for col in corr_columns]
        del_columns = list(set(del_columns))
        print(len(del_columns), ' collinear columns')
        for col in del_columns:
            del df[col]
        return(df)
    
    '''
     Preprocessing function: normalizes with MinMaxScaler; categorical encoding with label encoding for categories with 2 values and one hot encoding
     for all the others; delete columns with a certain quantity of NaNs, in out case 75%.
    
    '''
    def preprocess(self, df):
        df_x = df.iloc[:307511]
        
        
        summary = df_x.describe().iloc[0]
        # columns that have NaNs
        # true number of loans
        T_count = len(df_x)
        nancol = {}
        # for each column with missing values save number of NaNs/T_count
        for col in df_x.columns:
            nans=sum(df_x[col].isnull())
            if nans != 0:
                nancol[col]=nans/T_count
        #print("number of columns with nans ",len(nancol))
    
        #nandf = pd.DataFrame.from_dict(nancol, orient="index")
        #columns with nans % greater than 75%
        miss_col = [key for key,value in nancol.items() if value > 0.75]
        #nandf.loc[miss_col,0].head()
    
        for col in miss_col:
            del df[col]
    
            
        scaler = MinMaxScaler()
        scaler.fit(df[[col for col in df.columns if df[col].dtype != 'object']].iloc[:307511]) 
        df[[col for col in df.columns if df[col].dtype != 'object']].iloc[:307511] = scaler.transform(
        df[[col for col in df.columns if df[col].dtype != 'object']].iloc[:307511])
        
        scaler = MinMaxScaler()
        scaler.fit(df[[col for col in df.columns if df[col].dtype != 'object']].iloc[307511:]) 
        df[[col for col in df.columns if df[col].dtype != 'object']].iloc[307511:] = scaler.transform(
        df[[col for col in df.columns if df[col].dtype != 'object']].iloc[307511:])        
            
        # Label Encoder for categorical variables with only 2 values, e.g. "yes", "no"
    
        le = LabelEncoder()
        for col in df.columns:
            if df[col].dtype == "object" and len(df[col].unique())==2:
                le.fit(df[col])
                df[col] = le.transform(df[col])
                #print(col)
    
        # One-hot encoding for categorical variables with more than 2 values. Since this type of encoding introduces
        # multicollinearity between the new formed columns, the solution we can take to prevent this is to drop one of the new columns.
        df = pd.get_dummies(df, drop_first=True)   
        return(df)
    
'''
All features needed to study the train DF
'''

class train_analysis():
    
    '''
    It returns a barplot with x the variable under observation ans as y the 
    mean related to TARGET variable for each category (The higher the mean, 
    the higher the probability to default, for that category)
    '''
    
    def bins_impact(self, df, col_lst = None, nrow = 1, ncol = 1, rot = 0, sort = False, size = (8, 6)):
        
        if nrow == 1 and ncol == 1:
            
            grouped_df = df.groupby(col_lst[0]).mean()
            if sort:
                grouped_df = grouped_df.sort_values(by = ["TARGET"], ascending = False)
            label = grouped_df.index.name
            
            plt.figure(figsize = size)
            sns.barplot(grouped_df.index.astype(str), 100 * grouped_df['TARGET'], zorder = 2)
            plt.grid(color = "gainsboro", linestyle='-.', linewidth = 0.5, zorder = 0)
            plt.title(label + " Vs Percentage to Fail", pad = 15, fontsize = 14)
            plt.ylabel("Failure to Repay (%)")
            plt.show()
        
        else:
            
            fig, axs = plt.subplots(nrow, ncol, figsize = size)
            fig.subplots_adjust(hspace = .3, wspace = .3)
            axs = axs.ravel()
            
            for i, axis in zip(range(nrow * ncol), axs.flat):
                
                c = col_lst[i]
                gdf = df.groupby(c).mean()
                
                if sort:
                    gdf = gdf.sort_values(by = ["TARGET"], ascending = False)
                
                label = gdf.index.name
            
                sns.barplot(gdf.index.astype(str), 100 * gdf['TARGET'], zorder = 2, ax = axis)
                axs[i].set_title(label + " Vs " + "Prob to Repay", pad = 12)
                axs[i].set_xlabel("")
                axs[i].set_xticklabels(labels = gdf.index, rotation = rot)
                
    
    '''
    It showsthe bar plot for categorical variables. 
    With method compare, shows also the count for each class of the target
    '''
    
    def bar_plot(self, df, col_lst, nrow = 1, ncol = 1, rot = 0, size = (8,6), name = ""):

        if nrow == 1 and ncol == 1:   
            
            names = df[col_lst[0]].unique()
            values = Counter(df[col_lst[0]]).values()
            
            plt.figure(figsize = size)
            plt.bar(names, values, color = sns.color_palette("Set2", len(names)), zorder = 2)
            plt.grid(color = "gainsboro", linestyle='-.', linewidth = 0.5, zorder = 0)
            plt.ylabel("Count")
            plt.title(col_lst[0] + " per Category", pad = 15, fontsize = 14)
            plt.show()
        
        else:
            
            fig, axs = plt.subplots(nrow, ncol, figsize = size)
            fig.subplots_adjust(hspace = .5, wspace = .3)
            axs = axs.ravel()
            
            for i in range(nrow * ncol):
                
                names = df[col_lst[i]].unique()
                values = Counter(df[col_lst[i]]).values()
                

                axs[i].bar(names, values, color = sns.color_palette("Set2", len(names)), zorder = 2)
                axs[i].grid(color = "gainsboro", linestyle='-.', linewidth = 0.5, zorder = 0)
                axs[i].set_ylabel("Count")
                axs[i].set_title(col_lst[i] + " per Category", pad = 15, fontsize = 14)
                axs[i].set_xticklabels(labels = names, rotation = rot)
                

    '''
    It returns the correlation among variables in the input DF,
    NOCATEGORICAL VARIABLE ADMITED
    '''
    
    def cor_plot(self, df, size = (10, 8), an = False, name = None):
        
        cat_type = [x for x in df.dtypes.values if x == np.object]
        if cat_type:
            
            warnings.warn("Categorical Variables are NOT Ammited, they are Automaitcally DROPED")
            
            df = df.select_dtypes('float64')
            cor_matrix = df.corr()
    
            plt.figure(figsize = size)
            sns.heatmap(cor_matrix, linewidths=.5, cmap = "RdBu_r", center = 0, annot = an)
            #plt.xticks(range(df.shape[1]), df.columns, fontsize = 14, rotation = 90)
            #plt.yticks(range(df.shape[1]), df.columns, fontsize = 14)
            
            plt.show()
        
        else:
            
            cor_matrix = df.corr()
        
            plt.figure(figsize = size)
            sns.heatmap(cor_matrix, linewidths=.5, cmap = "RdBu_r", center = 0, annot = an)
            #plt.xticks(range(df.shape[1]), df.columns, fontsize = 14, rotation = 90)
            #plt.yticks(range(df.shape[1]), df.columns, fontsize = 14)
            plt.show()
    
    '''
    It shows the distribution of more than one variables in the DF in only one figure.
    You can also choose among normal histogram or kdeplot ("kind" parameter)
    '''
    
    def plot_distributions(self, df, col_lst, nrow = 1 , ncol = 1, kind = "histogram", size = (15, 6), name = ""):
        
        np.warnings.filterwarnings('ignore')
        
        if nrow == 1 and ncol == 1:
            
            if kind == "histogram":
                
                plt.figure(figsize = size)
                plt.hist(df[col_lst[0]], edgecolor='black', linewidth=1.2)
                plt.show()
            
            elif kind == "kdeplot":
                
                plt.figure(figsize = size)
                sns.kdeplot(df.loc[df['TARGET'] == 0, col_lst[0]], label = 'target == 0')
                sns.kdeplot(df.loc[df['TARGET'] == 1, col_lst[0]], label = 'target == 1')
                plt.xlabel(col_lst[0])
                plt.title(col_lst[0] + " Distribution by Target", pad = 14)     
                plt.show()
        
        else:
        
            fig, axs = plt.subplots(nrow, ncol, figsize = size)
            fig.subplots_adjust(hspace = .5, wspace = .3)
            axs = axs.ravel()
            
            if kind == "histogram":
            
                for i in range(nrow * ncol):
                    
                    axs[i].hist(df[col_lst[i]], edgecolor='black', linewidth=1.2)
                    axs[i].set_xlabel(col_lst[i])
                    axs[i].set_ylabel("Count", size = 14)
                    axs[i].set_title(col_lst[i] + " Distribution", pad = 12)
            
            elif kind == "kdeplot":
                
                
                for i, axis in zip(range(nrow * ncol), axs.flat):
    
                    sns.kdeplot(df.loc[df['TARGET'] == 0, col_lst[i]], label = 'target == 0', ax = axis)
                    sns.kdeplot(df.loc[df['TARGET'] == 1, col_lst[i]], label = 'target == 1', ax = axis)
                    axs[i].set_xlabel(col_lst[i])
                    axs[i].set_title(col_lst[i] + " Distribution by Target", pad = 14)
                    
    
    '''
    It shows multiple box plots in the same figure
    '''
    
    def Box_Plots(self, df, col_lst, nrow, ncol, size = (10, 10), name = ""):
        
        fig, axs = plt.subplots(nrow, ncol, figsize = size)
        fig.subplots_adjust(hspace = .3, wspace = .3)
        axs = axs.ravel()
        
        for i, axis in zip(range(nrow * ncol), axs.flat):
        
            sns.boxplot(df[col_lst[i]], orient = "v", ax = axis)
            axs[i].set_ylabel("")
            axs[i].set_xlabel("")
            axs[i].set_title(col_lst[i], pad = 12)
        
        
    '''
    INPUT: The DF Should be insert without target variable
        df = the dataframe musn't contain NA
    It calculate the features importance through random forest,
    It plot feature importance thorugh barplot.
    
    '''
    
    def box_hist(self, df, col, size = (14, 6), name = ""):
        
        plt.figure(figsize = size)
        plt.subplots_adjust(wspace = .5)
        
        plt.subplot(121)
        sns.boxplot(df[col], orient = "v")
        plt.title(col + " Box Plot", pad = 12)
        
        plt.subplot(122)
        plt.hist(df[col], edgecolor='black', linewidth=1.2)
        plt.title(col + " Box Plot", pad = 12)
        
        plt.show()
    
    def ANOVA_Importance(self, df, train, size = (10,10), name = None):
        
        
        are_there_na = df.isnull().sum().sum() + train["TARGET"].isnull().sum()
        
        if are_there_na != 0:
            
            raise TypeError("NaN not ammitted, fill them before going on")
        
        else:
        
            # Variable:
    
            X = df
            y = train["TARGET"]
            
            # Test the importance (ANOVA method)
    
            fs = SelectKBest(score_func = f_classif, k = 'all')
            fs.fit(X, y)
            
            # Generate the DF:
            
            importance = list(zip(df.columns, fs.scores_))
            importance_df = pd.DataFrame(importance, columns = ["FEATURES", "IMPORTANCE"])
            importance_df = importance_df.sort_values(by=['IMPORTANCE'], ascending = False)
            
            importance_avg = importance_df["IMPORTANCE"].mean()
            
            # Plot:
            
            plt.figure(figsize = size)
            sns.barplot(x = "IMPORTANCE", y = "FEATURES", data = importance_df)
            plt.plot([importance_avg, importance_avg], [-1, df.shape[1]], "--r")
            plt.title("Important Features", pad = 15, fontsize = 14)
            plt.xticks(fontsize = 10)
            plt.show()


class generate_test():
    
    def __init__(self, df):
        
        self.df = df
        self.final_df = df[["SK_ID_CURR"]]
        
    def house_info(self):
        
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
        
        # Calculate the column to check if info is present:
        
        n_cols = home_test.shape[1]
        home_test["HOUSE_INFO"] = (home_test.isnull().sum(axis = 1)/n_cols).apply(lambda x: "NO" if x >= 0.5 else "YES")
        
        # Add this column to final DF:
        
        self.final_df = pd.concat([self.final_df, home_test["HOUSE_INFO"]], axis = 1)

        '''
        # Fill NA of needed columns:
        
        tools = utils()
        
        home_test["EMERGENCYSTATE_MODE"].fillna(value = "not specified", inplace = True)
        
        tools.imputer(home_test, "TOTALAREA_MODE", method = "median")
        tools.imputer(home_test, "YEARS_BEGINEXPLUATATION_AVG", method = "mean")
        '''
        
        # Add the columns to the final DF and Encode them:
        
        self.final_df = pd.concat([self.final_df, home_test[["TOTALAREA_MODE", "YEARS_BEGINEXPLUATATION_AVG", "EMERGENCYSTATE_MODE"]]],
                                    axis = 1)
        self.final_df = pd.get_dummies(self.final_df)
        
        # Drop the columns to avoid multicollinearity:
        
        self.final_df = self.final_df.drop(columns = ["HOUSE_INFO_NO"])
        self.final_df = self.final_df[["SK_ID_CURR", "TOTALAREA_MODE", "YEARS_BEGINEXPLUATATION_AVG", 
                                           "HOUSE_INFO_YES", "EMERGENCYSTATE_MODE_Yes", 
                                           "EMERGENCYSTATE_MODE_No"]]
        
    def external_sources(self):
        
        # Get the columns we need:
        
        externals_test = self.df[["EXT_SOURCE_2", "EXT_SOURCE_3"]].copy()
        
        '''
        # Fill NAs:
        
        tools = utils()
        tools.imputer(externals_test, "EXT_SOURCE_2", method = "median")
        tools.imputer(externals_test, "EXT_SOURCE_3", method = "median")
        '''
        
        # Add to final DF:
        
        self.final_df = pd.concat([self.final_df, externals_test], axis = 1)
    
    def flags_doc(self):
        
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
        
        # Add to the final DF new variables:
        
        self.final_df = pd.concat([self.final_df, flags_test], axis = 1)
    
    def social_circle(self):
        
        # Create the DF to manipulate:
        
        social_test = self.df[['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
                 'DEF_60_CNT_SOCIAL_CIRCLE']]
        
        # Remove attributes:
        
        social_test =  social_test.drop(columns = ['OBS_30_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE'], axis = 1)
        
        # Transform 0 in missing values:
        
        social_test.loc[social_test["OBS_60_CNT_SOCIAL_CIRCLE"] == 0, "OBS_60_CNT_SOCIAL_CIRCLE"] = np.nan
        social_test.loc[(social_test["OBS_60_CNT_SOCIAL_CIRCLE"] == np.nan) & 
                        (social_test["DEF_60_CNT_SOCIAL_CIRCLE"] == 0), "DEF_60_CNT_SOCIAL_CIRCLE"] = np.nan
        
        # New column:
        
        social_test["SOCIAL_DEFAUL_RATIO_60"]=social_test["DEF_60_CNT_SOCIAL_CIRCLE"] /social_test["OBS_60_CNT_SOCIAL_CIRCLE"]
        
        
        '''                
        tools = utils()
        tools.imputer(social_test, "OBS_60_CNT_SOCIAL_CIRCLE", method = "median")
        '''

        # Add to the final DF:
               
        self.final_df = pd.concat([self.final_df, social_test["SOCIAL_DEFAUL_RATIO_60"]], axis = 1)
    
    def enquires_yearly(self):
        
        # Generate the DF:
        
        enquires_test = self.df[["AMT_REQ_CREDIT_BUREAU_HOUR", "AMT_REQ_CREDIT_BUREAU_DAY", "AMT_REQ_CREDIT_BUREAU_WEEK",
                 "AMT_REQ_CREDIT_BUREAU_MON", "AMT_REQ_CREDIT_BUREAU_QRT", "AMT_REQ_CREDIT_BUREAU_YEAR"]]
        
        # The new variable creation:
        
        enquires_test["TOT_ENQUIRES_YEARLY"] = enquires_test.iloc[: , range(6)].sum(axis = 1)
        enquires_test.loc[(enquires_test.iloc[: , range(6)].isnull()).any(axis = 1), "TOT_ENQUIRES_YEARLY"] = np.nan
        
        '''
        # Fill nan:
        
        tools = utils()
        tools.imputer(enquires_test, "TOT_ENQUIRES_YEARLY", method = "median")
        '''
        
        # Add to the final DF:
        
        self.final_df = pd.concat([self.final_df, enquires_test["TOT_ENQUIRES_YEARLY"]], axis = 1)
    
    def contract_type(self):
        
        # Generate the dummies:
        
        type_dummy = pd.get_dummies(self.df["NAME_CONTRACT_TYPE"], prefix = "CONTRACT_TYPE")
        type_dummy.drop(columns = ["CONTRACT_TYPE_Revolving loans"], inplace = True)
        
        # Add to the final DF:
        
        self.final_df = pd.concat([self.final_df, type_dummy], axis = 1)
    
    def economics_stat(self):
        
        # Generate the DF:
        
        economics_test = self.df[["AMT_INCOME_TOTAL", "AMT_ANNUITY", "AMT_CREDIT"]]
        
        # New varaibles:
        
        economics_test["DURATION"] = economics_test["AMT_CREDIT"] / economics_test["AMT_ANNUITY"]
        economics_test["AMT_ANNUITY_INCOMES_RATIO"] = economics_test["AMT_ANNUITY"] / economics_test["AMT_INCOME_TOTAL"]

        '''
        # Fill nan:
        
        tools = utils()
        tools.imputer(economics_test, "DURATION", method = "median")
        tools.imputer(economics_test, "AMT_ANNUITY_INCOMES_RATIO", method = "median")
        '''
        
        # Add to final DF:
        
        self.final_df = pd.concat([self.final_df, economics_test[["DURATION", "AMT_ANNUITY_INCOMES_RATIO"]]],
                                    axis = 1)
        
    def personal_info(self):
        
        # Generate the DF:
        
        personalInf_test = self.df[['CODE_GENDER','DAYS_BIRTH', 'NAME_EDUCATION_TYPE']]
        
        # Transform variables:
        
        personalInf_test["YEARS_BIRTH"] = personalInf_test["DAYS_BIRTH"].apply(lambda x: -int(x/365))
        
        
        # Fill NA:
        '''
        gender_mode = personalInf_test["CODE_GENDER"].mode()
        personalInf_test.loc[personalInf_test["CODE_GENDER"] == "XNA", "CODE_GENDER"] = gender_mode
        
        edu_mode = personalInf_test["CODE_GENDER"].mode()
        personalInf_test.loc[personalInf_test["NAME_EDUCATION_TYPE"] == "XNA", "NAME_EDUCATION_TYPE"] = edu_mode
        '''
        
        # Group the edu variable:
        
        personalInf_test.loc[personalInf_test['NAME_EDUCATION_TYPE'] == 'Incomplete higher', 'NAME_EDUCATION_TYPE'] = 'Secondary / secondary special'
        personalInf_test.loc[personalInf_test['NAME_EDUCATION_TYPE'] == 'Academic degree', 'NAME_EDUCATION_TYPE'] = 'Higher education'
        personalInf_test.loc[personalInf_test['NAME_EDUCATION_TYPE'] == 'Secondary / secondary special', 'NAME_EDUCATION_TYPE'] = 'Secondary or Less'
        personalInf_test.loc[personalInf_test['NAME_EDUCATION_TYPE'] == 'Lower secondary', 'NAME_EDUCATION_TYPE'] = 'Secondary or Less'

        # Encode:
        
        personalInf_test = pd.get_dummies(personalInf_test)
        personalInf_test = personalInf_test[["YEARS_BIRTH",	"CODE_GENDER_M",	 "NAME_EDUCATION_TYPE_Secondary or Less"]]
        
        # Add to the final DF:
        
        self.final_df = pd.concat([self.final_df, personalInf_test], axis = 1)
        

    def family_status(self):
        
        family_test = self.df[['NAME_FAMILY_STATUS', 'CNT_FAM_MEMBERS','CNT_CHILDREN']]
        
        # Generate new variables:
        
        family_test.insert(2, "HAS_CHILDREN", "No")
        family_test.loc[family_test["CNT_CHILDREN"] > 0, "HAS_CHILDREN"] = "Yes"
        
        '''
        count_yes = len(family_test[family_test["HAS_CHILDREN"] == "Yes"])
        count_no = len(family_test[family_test["HAS_CHILDREN"] == "No"])
        
        if count_yes > count_no:
            family_test.loc[family_test["CNT_CHILDREN"].isnull(), "HAS_CHILDREN"] = "Yes"
        else:
            family_test.loc[family_test["CNT_CHILDREN"].isnull(), "HAS_CHILDREN"] = "No"
        '''
        
        '''
        # Removr NA:
        
        family_mode = family_test['NAME_FAMILY_STATUS'].mode()[0]
        family_test.loc[family_test['NAME_FAMILY_STATUS'] == "Unknown"] = family_mode
        '''
        
        # Get dummies:
        
        family_test = pd.get_dummies(family_test)
        
        # Drop columns:
        
        family_test = family_test.drop(columns = ['CNT_FAM_MEMBERS', "CNT_CHILDREN", "HAS_CHILDREN_No", 
                                                  "NAME_FAMILY_STATUS_Married"])
        
        # Add to the final DF:
        
        self.final_df = pd.concat([self.final_df, family_test], axis = 1)

    def contact_info(self):
        
        # Create DF:
        
        contact_test = self.df[['DAYS_LAST_PHONE_CHANGE', 'FLAG_EMP_PHONE','FLAG_PHONE','FLAG_WORK_PHONE',
                                   'FLAG_EMAIL']]
        
        # Modify column and fill NA:
        
        contact_test['YEARS_LAST_PHONE_CHANGE'] = contact_test['DAYS_LAST_PHONE_CHANGE'].apply(lambda x: -x/365)
        #tools = utils()
        #tools.imputer(contact_test, "YEARS_LAST_PHONE_CHANGE", method = "median")
        
        # Transform in dummies:
        
        cols = ['FLAG_EMP_PHONE','FLAG_PHONE','FLAG_WORK_PHONE', 'FLAG_EMAIL']
        for c in cols:
        
            contact_test.loc[contact_test[c] == 1, c] = "Yes"
            contact_test.loc[contact_test[c] == 0, c] = "No"
        
        # Encode:
        
        contact_test = pd.get_dummies(contact_test)
        
        
        # Drop Columns:
        
        contact_test = contact_test.drop(columns = ["FLAG_EMP_PHONE_No", "FLAG_PHONE_Yes", "FLAG_WORK_PHONE_No",
                                                    "FLAG_EMAIL_No", 'DAYS_LAST_PHONE_CHANGE'])
        
        # Add to the final DF:
        
        self.final_df = pd.concat([self.final_df, contact_test], axis = 1)
        
    def ownership(self):
        
        # Create DF:
        
        properties_test = self.df[['OWN_CAR_AGE','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_HOUSING_TYPE',
                                      'NAME_INCOME_TYPE']]
        
        # Transform variables:
        
        properties_test.insert(4, "NAME_INCOME_TYPE_GROUPED", "")
        properties_test.loc[(properties_test["NAME_INCOME_TYPE"] == "Maternity leave") | 
                (properties_test["NAME_INCOME_TYPE"] == "Unemployed") |
                (properties_test["NAME_INCOME_TYPE"] == "Student"), 
                "NAME_INCOME_TYPE_GROUPED"] = "Not working income / No income"
        properties_test.loc[(properties_test["NAME_INCOME_TYPE"] != "Maternity leave") & 
                (properties_test["NAME_INCOME_TYPE"] != "Unemployed") &
                (properties_test["NAME_INCOME_TYPE"] != "Student"), 
                "NAME_INCOME_TYPE_GROUPED"] = "Working Income"
        
        properties_test.insert(3, "NAME_HOUSING_TYPE_GROUPED", "")
        properties_test.loc[(properties_test["NAME_HOUSING_TYPE"] == "Rented apartment") | 
               (properties_test["NAME_HOUSING_TYPE"] == "Municipal apartment") |
               (properties_test["NAME_HOUSING_TYPE"] == "Co-op apartment"), 
               "NAME_HOUSING_TYPE_GROUPED"] = "Subjected to a rent"
        properties_test.loc[(properties_test["NAME_HOUSING_TYPE"] == "Office apartment"), 
               "NAME_HOUSING_TYPE_GROUPED"] = "Office"
        properties_test.loc[(properties_test["NAME_HOUSING_TYPE"] == "With parents"), 
               "NAME_HOUSING_TYPE_GROUPED"] = "With parents"
        properties_test.loc[(properties_test["NAME_HOUSING_TYPE"] == "House / apartment"), 
               "NAME_HOUSING_TYPE_GROUPED"] = "Not subjected to a rent"

        # Drop unuse columns
        
        properties_test = properties_test.drop(columns = ['NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE'])

       
        # Get Dummies:
        
        properties_test = pd.get_dummies(properties_test)
        
        # Deal with dummy trap:
        
        properties_test = properties_test.drop(columns = ["FLAG_OWN_CAR_Y", "FLAG_OWN_REALTY_Y", 
                                        "NAME_HOUSING_TYPE_GROUPED_Not subjected to a rent", 
                                       "NAME_INCOME_TYPE_GROUPED_Working Income"])
        
        '''
        # Fill NA:
        
        car_mean = properties_test["OWN_CAR_AGE"].mean()
        properties_test.loc[(properties_test["OWN_CAR_AGE"].isnull()) & 
                            (properties_test["FLAG_OWN_CAR_N"] == 0), "OWN_CAR_AGE"] = car_mean
        properties_test.loc[properties_test["OWN_CAR_AGE"].isnull(), "OWN_CAR_AGE"] = -9999
        '''
        
        # Add to final DF:
        
        self.final_df = pd.concat([self.final_df, properties_test], axis = 1)
        
    def info(self):
        
        # Generate DF:
        
        otherInfo_test = self.df[['DAYS_ID_PUBLISH','DAYS_REGISTRATION','NAME_TYPE_SUITE']]
        
        # Transform columns:
        
        otherInfo_test["YEARS_ID_PUBLISH"] = otherInfo_test["DAYS_ID_PUBLISH"].apply(lambda x: -x/365)
        otherInfo_test["YEARS_REGISTRATION"] = otherInfo_test["DAYS_REGISTRATION"].apply(lambda x: -x/365)
        
        
        # Fill NA:
        
        suite_mode = otherInfo_test["NAME_TYPE_SUITE"].mode()[0]
        otherInfo_test.loc[otherInfo_test["NAME_TYPE_SUITE"].isnull(), "NAME_TYPE_SUITE"] = suite_mode
       
        
        # Generate new column:
        
        otherInfo_test.insert(4, "NAME_TYPE_SUITE_GROUP", "")
        otherInfo_test.loc[(otherInfo_test["NAME_TYPE_SUITE"] == "Other_A") | 
                (otherInfo_test["NAME_TYPE_SUITE"] == "Other_B"), "NAME_TYPE_SUITE_GROUP"] = "Others"
        otherInfo_test.loc[(otherInfo_test["NAME_TYPE_SUITE"] == "Children") |
                (otherInfo_test["NAME_TYPE_SUITE"] == "Family") | 
                (otherInfo_test["NAME_TYPE_SUITE"] == "Spouse, partner"), "NAME_TYPE_SUITE_GROUP"] = "Family"
        otherInfo_test.loc[(otherInfo_test["NAME_TYPE_SUITE"] == "Group of people"), 
                           "NAME_TYPE_SUITE_GROUP"] = "Group of people"
        otherInfo_test.loc[(otherInfo_test["NAME_TYPE_SUITE"] == "Unaccompanied"), 
                           "NAME_TYPE_SUITE_GROUP"] = "Unaccompanied"
        
        # Drop columns:
        
        otherInfo_test = otherInfo_test.drop("NAME_TYPE_SUITE", axis = 1)
                           
        # Get dummies:
        
        otherInfo_test = pd.get_dummies(otherInfo_test)
        
        # Drop Columns:
        
        otherInfo_test = otherInfo_test.drop(columns = ["NAME_TYPE_SUITE_GROUP_Unaccompanied",'DAYS_ID_PUBLISH',
                                                        'DAYS_REGISTRATION'])
        
        # Add to the final DF:
        
        self.final_df = pd.concat([self.final_df, otherInfo_test], axis = 1)
    
    def region(self):
        
        # Generate DF:
        
        regionInfo_test = self.df[['REGION_RATING_CLIENT_W_CITY', 'LIVE_CITY_NOT_WORK_CITY','REG_CITY_NOT_LIVE_CITY']]
        
        # Transform variables:
        
        regionInfo_test.loc[regionInfo_test['LIVE_CITY_NOT_WORK_CITY'] == 1, 'LIVE_CITY_NOT_WORK_CITY'] = "Different"
        regionInfo_test.loc[regionInfo_test['LIVE_CITY_NOT_WORK_CITY'] == 0, 'LIVE_CITY_NOT_WORK_CITY'] = "Same"
        
        regionInfo_test.loc[regionInfo_test['REG_CITY_NOT_LIVE_CITY'] == 1, 'REG_CITY_NOT_LIVE_CITY'] = "Different"
        regionInfo_test.loc[regionInfo_test['REG_CITY_NOT_LIVE_CITY'] == 0, 'REG_CITY_NOT_LIVE_CITY'] = "Same"
        
        # Get Dummies:
        
        regionInfo_test = pd.get_dummies(regionInfo_test)
        
        # Drop columns:
        
        regionInfo_test = regionInfo_test.drop(columns = ["LIVE_CITY_NOT_WORK_CITY_Same", "REG_CITY_NOT_LIVE_CITY_Same"])
        
        # Add to Final DF:
        
        self.final_df = pd.concat([self.final_df, regionInfo_test], axis = 1)
        
    def occupation_test(self):
        
        # Generate DF:
        
        occupation_test = self.df[['DAYS_EMPLOYED', 'OCCUPATION_TYPE','ORGANIZATION_TYPE']]
        
        # Adjust Columns:
        
        occupation_test["YEARS_EMPLOYED"] = occupation_test["DAYS_EMPLOYED"].apply(lambda x: -x/365)
        occupation_test["YEARS_EMPLOYED"] = occupation_test["YEARS_EMPLOYED"].astype('float32')
        occupation_test["YEARS_EMPLOYED_LOG"] = np.log(occupation_test["YEARS_EMPLOYED"] + 1)
        
        '''
        tools = utils()
        tools.imputer(occupation_test, "YEARS_EMPLOYED_LOG", method = "mean")
        '''
        
        # Add columns

        occupation_test.insert(2,"OCCUPATION_TYPE_GROUP", "")
        occupation_test.insert(4, "ORGANIZATION_TYPE_GROUP", "")
        
        # Special GROUPING:
        ###############################################
        
        # Fill NA:
        
        org_type_mode = occupation_test["ORGANIZATION_TYPE"].mode()[0]
        occ_type_mode = occupation_test["OCCUPATION_TYPE"].mode()[0]
        
        occupation_test.loc[occupation_test["ORGANIZATION_TYPE"] == "XNA","ORGANIZATION_TYPE"] = org_type_mode
        occupation_test.loc[occupation_test["OCCUPATION_TYPE"] == "XNA", "OCCUPATION_TYPE"] = occ_type_mode
        
        # ORGANIZATION TYPE:
        
        occupation_test.loc[(occupation_test["ORGANIZATION_TYPE"] == "Business Entity Type 3") , "ORGANIZATION_TYPE_GROUP"] = "Business Entity Type 3"
        occupation_test.loc[(occupation_test["ORGANIZATION_TYPE"] == "Construction") , "ORGANIZATION_TYPE_GROUP"] = "Construction"
        occupation_test.loc[(occupation_test["ORGANIZATION_TYPE"] == "Transport type 3") , "ORGANIZATION_TYPE_GROUP"] = "Construction"
        occupation_test.loc[(occupation_test["ORGANIZATION_TYPE"] == "School") , "ORGANIZATION_TYPE_GROUP"] = "School"
        
        occupation_test.loc[(occupation_test["ORGANIZATION_TYPE"] != "Business Entity Type 3") & (occupation_test["ORGANIZATION_TYPE"] != "Construction") &
                      (occupation_test["ORGANIZATION_TYPE"] != "Transport type 3") & (occupation_test["ORGANIZATION_TYPE"] != "School"),
                      "ORGANIZATION_TYPE_GROUP"] = "Other"
        
        # occupation_test TYPE:
        
        occupation_test.loc[(occupation_test["OCCUPATION_TYPE"] == "Laborers") , "OCCUPATION_TYPE_GROUP"] = "Laborers"
        occupation_test.loc[(occupation_test["OCCUPATION_TYPE"] == "Drivers") , "OCCUPATION_TYPE_GROUP"] = "Drivers"
        occupation_test.loc[(occupation_test["OCCUPATION_TYPE"] == "Low-skill Laborers") , "OCCUPATION_TYPE_GROUP"] = "Low-skill Laborers"
        occupation_test.loc[(occupation_test["OCCUPATION_TYPE"] == "Accountants") , "OCCUPATION_TYPE_GROUP"] = "Accountants"
        occupation_test.loc[(occupation_test["OCCUPATION_TYPE"] == "Core staff") , "OCCUPATION_TYPE_GROUP"] = "Core staff"
        
        occupation_test.loc[(occupation_test["OCCUPATION_TYPE"] != "Laborers") & (occupation_test["OCCUPATION_TYPE"] != "Drivers") &
                       (occupation_test["OCCUPATION_TYPE"] != "Low-skill Laborers") & (occupation_test["OCCUPATION_TYPE"] != "Accountants") &
                       (occupation_test["OCCUPATION_TYPE"] != "Core staff"), "OCCUPATION_TYPE_GROUP"] = "Others"
    
        ############################################
       
        # Get needed columns and Dummies:

        occupation_test = occupation_test[["YEARS_EMPLOYED_LOG", "OCCUPATION_TYPE_GROUP", "ORGANIZATION_TYPE_GROUP"]]
        occupation_test = pd.get_dummies(occupation_test)
        
        # Drop Columns:

        occupation_test = occupation_test.drop(columns = ['ORGANIZATION_TYPE_GROUP_Other', "OCCUPATION_TYPE_GROUP_Others"]) 

        # Add to final DF:

        self.final_df = pd.concat([self.final_df, occupation_test], axis = 1)          
    
    def app_time(self):
        
        self.final_df = pd.concat([self.final_df, self.df[['HOUR_APPR_PROCESS_START']]], axis = 1)
        
    def Test_Output(self):
        
        pd.set_option('mode.chained_assignment', None)
        
        self.house_info()
        self.external_sources()
        self.flags_doc()
        self.social_circle()
        self.enquires_yearly()
        self.contract_type()
        self.economics_stat()
        self.personal_info()
        self.family_status()
        self.contact_info()
        self.ownership()
        self.info()
        self.region()
        self.occupation_test()
        self.app_time()
        
        
        return self.final_df
        
        
class dataframe_group:
    
    '''
    In all the dataframes except train and test for each SK_ID_CURR correspond different rows, that are different loans, credit history,
    monthly payments...For each SK_ID_CURR we grouped all those values and took the min, max, mean/median, and std for numerical values, and
    mode and frequency for categorical ones.
    
    '''
    
    
    def dataframe_chaining(self, df):
        
        numcol = [col for col in df.columns if df[col].dtype != "object"]
        catcol = [col for col in df.columns if col not in numcol]
        
       
        numcol.remove(df.columns[0])
        new = pd.DataFrame()
        new['SK_ID_CURR'] = df.index.unique()
        new = new.set_index('SK_ID_CURR')

        for col in numcol:
            print(col)
            temp = df[col][pd.notna(df[col])]
            temp = temp.groupby(level=0).apply(list).to_dict()
            jnk = {}
            for i in temp:
                jnk[i] = max(temp[i])
            new[col+'_MAX'] = pd.Series(jnk)    
            jnk = {}
            for i in temp:
                jnk[i] = min(temp[i])
            new[col+'_MIN'] = pd.Series(jnk)
            jnk = {}
            for i in temp:
                jnk[i] = sum(temp[i])/(len(temp[i]))
            new[col+'_MEAN'] = pd.Series(jnk)
            jnk = {}
            for i in temp:
                jnk[i] = np.std(temp[i])
            new[col + '_STD'] = pd.Series(jnk)
    
            
        for col in catcol:
            print(col)
            temp = df[col][pd.notna(df[col])]
            temp = temp.groupby(level=0).apply(list).to_dict()
            jnk = {}
            jnk2 = {}
            for i in temp:
                jnk[i] = Counter(temp[i]).most_common()[0][0]
                jnk2[i] = temp[i].count(jnk[i])/len(temp[i])
            new[col+'_MODE'] = pd.Series(jnk)
            new[col+'_FREQ'] = pd.Series(jnk2)
            
            
        return(new)
        
        
