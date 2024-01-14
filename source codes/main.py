from data_preprocessing import Data_Preprocessor

metadata={
        'wd_path':'C:\\Users\\sapta\\Documents\\GitHub\\MLAAS\\dev_docs\\datasets\\',
        'filename':'data_prep_dataset.csv',
        'attr_ind':['Country', 'Age', 'Salary'],
        'attr_dep':['Purchased'],
        'attr_cat':['Country'],
        'attr_num':['Age', 'Salary'],
        'aggregator':'mean',
        'test_size':0.3
    }

def main():
    data_prep = Data_Preprocessor(**metadata) # create a pre-processor object with dataset metadata
    result = data_prep.pre_process()
    print(result)
    

if __name__ == '__main__':
    main()