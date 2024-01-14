import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class Data_Preprocessor:
    def __init__(self, 
                 wd_path:str, filename:str, 
                 attr_cat:list, attr_num:list, attr_dep:list, attr_ind:list,
                 aggregator:str, 
                 test_size:float=0.2
                 ):
        self.WD_PATH=wd_path        # working dir path to the dataset 
        self.FILE_NAME=filename     # finename of the dataset 
        self.ATTR_DEP=attr_dep      # dependent attr
        self.ATTR_IND=attr_ind      # independent attr
        self.ATTR_CAT=attr_cat      # catagorical attr
        self.ATTR_NUM=attr_num      # numeric attr 
        self.AGGREGATOR=aggregator  # aggregator strategy for replacing NaNs in numeric attr
        self.TEST_SIZE=test_size    # test size in TT-split 
        
        self.dataset=None           # dataset as np.array
        self.df_dataset=None        # dataset as pd.dataframe
        
        self.df_X=None
        self.df_y=None
        
        self.X=None
        self.y=None
        
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None

    def load_dataset(self)-> pd.DataFrame:
        target_file = self.WD_PATH + self.FILE_NAME   # absolute path of the file
        try:
            self.df_dataset = pd.read_csv(target_file) # load dataset into DataFrame
        except Exception as E:
            print('ERROR: LOADING FAILED !!')
            print(E)
    
    def dep_ind_split(self):
        self.df_X = pd.DataFrame(self.df_dataset, columns=self.ATTR_IND) # extract subset df of independent attr
        self.df_y = pd.DataFrame(self.df_dataset, columns=self.ATTR_DEP) # extract subset df of dependent attr

    def remove_missing_data(self):
        col_with_na = self.df_X.columns[self.df_X.isna().any()].tolist()  # columns (List) with missing value 

        non_catagrical_cols_with_na =list(set(col_with_na) - set(self.ATTR_CAT))
        catagorical_cols_with_na = list(set(col_with_na).intersection(self.ATTR_CAT))

        # Replace missing data by mean for each NON-CATAGORICAL column with missing data.
        for col in non_catagrical_cols_with_na:

            # calculate aggregated_val as per chosen aggregator (mean/median/mode)
            if self.AGGREGATOR == 'mean':
                aggregated_val = self.df_X[col].mean()
            elif self.AGGREGATOR == 'median':
                aggregated_val = self.df_X[col].median()
            elif self.AGGREGATOR == 'mode':
                aggregated_val = self.df_X[col].mode()
            else:
                raise Exception('Unknown aggregator: choose from mean/median/mode')

            self.df_X[col].fillna(aggregated_val,inplace=True)

        # Replace missing data by removing rows with missing data for each CATAGORICAL column with missing data.
        self.df_dataset.dropna(subset=catagorical_cols_with_na, inplace=True)

    def one_hot_encoding(self):
        for col in self.ATTR_CAT:     # for all catagorical col
            dummy_cols = pd.get_dummies(self.df_X[col], prefix=col).astype(np.int8)  # get OHE dummy cols as 0/1
            dummy_cols.drop(columns=list(dummy_cols.columns)[0], inplace=True)       # drop the first dummy attr (avoid multi collinearity)
            self.df_X = self.df_X.join(dummy_cols)         # add dummy cols inplace to df_X
            self.df_X.drop([col],axis=1,inplace=True)      # drop original col inplace

    def label_encoding(self):
        le = LabelEncoder()
        self.df_y=le.fit_transform(self.df_y)

    def df_to_np_arr(self):
        self.X=np.array(self.df_X)
        self.y=np.array(self.df_y)

    def train_test_split(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=self.TEST_SIZE)

    def feature_scaling(self):
        sc_X = StandardScaler()
        self.X_train = sc_X.fit_transform(self.X_train)
        self.X_test = sc_X.transform(self.X_test)
        return {
            'x_train':self.X_train,
            'x_test':self.X_test,
            'y_train':self.y_train,
            'y_test':self.y_test
        }

    def pre_process(self)->dict:
        processed_data = {}                # to be populated and returned 

        try:
            self.load_dataset()            # load dataset
            self.dep_ind_split()           # dependent, indepndent split
            self.remove_missing_data()     # fix missing data (replace with aggregator for numeric, remove for catagorical)
            self.one_hot_encoding()        # OHE on catagorical cols
            self.label_encoding()          # label encode dependent attr
            self.df_to_np_arr()            # convert dataframe to np arr (X and y)
            self.train_test_split()        # train-test Split
            processed_data = self.feature_scaling() # Feature Scalling

        except Exception as E:
            print(E)
        
        finally:
            return processed_data