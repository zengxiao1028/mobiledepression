import pandas as pd
import numpy as np
from scipy import stats
import os
import pickle
import MyConfig
from sklearn.externals import joblib

def prepare_em_state(file_path, parse_cols):
    labels = pd.read_excel(file_path,parse_cols=parse_cols)
    ems = labels[82:]
    ems = ems.dropna(how='all')
    print 'start',ems.iloc[0]
    print 'end', ems.iloc[-1]
    print len(ems)
    return ems

def main():
    ## subjects:
    ## CL263NI
    ## 1155329
    ## total phq9_score does not match

    clinical_path = MyConfig.clinical_path
    file_path = os.path.join(clinical_path,'CS120Final_Baseline.xlsx')
    baseline = prepare_em_state(file_path,'E,AV')
    baseline_em_dict = baseline[['ID', 'slabels02']].set_index('ID').T.to_dict('list')
    print np.array(baseline_em_dict.values()).min()

    new_baseline_em_dict = {str(key):baseline_em_dict[key][0] for key in baseline_em_dict.keys()}

    joblib.dump(new_baseline_em_dict,'em_dict.pkl')
    print( 'Finish')

if __name__ == '__main__':
    main()



