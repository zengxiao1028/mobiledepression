import pandas as pd
import numpy as np
from scipy import stats
import os
import pickle
import MyConfig
def prepare_label(file_path, parse_cols):
    labels = pd.read_excel(file_path,parse_cols=parse_cols)
    phq9 = labels[82:]
    phq9 = phq9.dropna(how='all')
    # phq9 = phq9.set_index('ID').T.to_dict('list')
    # print phq9
    print len(phq9)
    return phq9

def main():
    ## subjects:
    ## CL263NI
    ## 1155329
    ## total phq9_score does not match
    clinical_path = MyConfig.clinical_path
    file_path = os.path.join(clinical_path,'CS120Final_Screener.xlsx')
    week0_phq9 = prepare_label(file_path,'E,BM:BU')
    week0_phq9['sum_score'] = week0_phq9['phq01'] + week0_phq9['phq02'] + week0_phq9['phq03'] + week0_phq9['phq04']\
                        + week0_phq9['phq05'] + week0_phq9['phq06'] + week0_phq9['phq07'] + week0_phq9['phq08']
    week0_phq9_dict = week0_phq9[['ID','sum_score']].set_index('ID').T.to_dict('list')

    file_path = os.path.join(clinical_path, 'CS120Final_3week.xlsx')
    week3_phq9 = prepare_label(file_path, 'E,BJ:BQ')
    week3_phq9['sum_score'] = week3_phq9['phqPrompt01_3wk Little interest or pleasure in doing things'] + \
                              week3_phq9['phqPrompt01_3wk Feeling down; depressed; or hopeless'] + \
                              week3_phq9['phqPrompt01_3wk Trouble falling or staying asleep; or sleeping too much'] + \
                              week3_phq9['phqPrompt01_3wk Feeling tired or having little energy'] + \
                              week3_phq9['phqPrompt01_3wk Poor appetite or overeating'] + \
                              week3_phq9['phqPrompt01_3wk Feeling bad about yourself - or that you are a failure or have let yourself or your family down'] + \
                              week3_phq9['phqPrompt01_3wk Trouble concentrating on things; such as reading the newspaper or watching television'] + \
                              week3_phq9['phqPrompt01_3wk Moving or speaking so slowly that other people could have noticed. ' \
                                         'Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual']
    week3_phq9_dict = week3_phq9[['ID','sum_score']].set_index('ID').T.to_dict('list')

    file_path = os.path.join(clinical_path,'CS120Final_6week.xlsx')
    week6_phq9 = prepare_label(file_path, 'E,BJ:BQ')
    week6_phq9['sum_score'] = week6_phq9['phqPrompt01_6wk Little interest or pleasure in doing things'] + \
                              week6_phq9['phqPrompt01_6wk Feeling down; depressed; or hopeless'] + \
                              week6_phq9['phqPrompt01_6wk Trouble falling or staying asleep; or sleeping too much'] + \
                              week6_phq9['phqPrompt01_6wk Feeling tired or having little energy'] + \
                              week6_phq9['phqPrompt01_6wk Poor appetite or overeating'] + \
                              week6_phq9[ 'phqPrompt01_6wk Feeling bad about yourself - or that you are a failure or have let yourself or your family down'] + \
                              week6_phq9['phqPrompt01_6wk Trouble concentrating on things; such as reading the newspaper or watching television'] + \
                              week6_phq9['phqPrompt01_6wk Moving or speaking so slowly that other people could have noticed. ' \
                                         'Or the opposite - being so fidgety or restless that you have been moving around a lot more than usual']
    week6_phq9_dict = week6_phq9[['ID','sum_score']].set_index('ID').T.to_dict('list')

    scores_dict = dict()
    for subject in week0_phq9_dict.keys():
        subject_phq9_scores = np.ones((3)) * -1

        assert len(week0_phq9_dict[subject]) == 1
        subject_phq9_scores[0] = week0_phq9_dict[subject][0]

        if subject in week3_phq9_dict.keys():
            assert len(week3_phq9_dict[subject]) == 1
            subject_phq9_scores[1] = week3_phq9_dict[subject][0]

        if subject in week6_phq9_dict.keys():
            assert len(week6_phq9_dict[subject]) == 1
            subject_phq9_scores[2] = week6_phq9_dict[subject][0]

        scores_dict[str(subject)] = subject_phq9_scores

    print scores_dict

    labels_dict = dict()
    for subject in scores_dict.keys():
        subject_phq9_scores = scores_dict[subject].astype(np.int32)
        subject_labels = np.empty_like(subject_phq9_scores)
        for idx,score in enumerate(subject_phq9_scores):
            # score 0-9, nondepression(0)
            if score>=0 and score <= 9:
                subject_labels[idx] = 0
            # score>9 , depression(1)
            elif score>9:
                subject_labels[idx] = 1
            # score< 0
            else:
                subject_labels[idx] = -1

        labels_dict[subject] = subject_labels

    #find consistent samples
    dprsn_counter = 0
    ndprsn_counter = 0
    filtered_labels = dict()
    for subject in labels_dict.keys():
        subject_labels = labels_dict[subject]
        if subject_labels[0] == subject_labels[1] and subject_labels[0] == subject_labels[2]:
            if subject_labels[0] == -1:
                continue
            elif subject_labels[0] == 0:
                filtered_labels[subject] = 0
                ndprsn_counter += 1
            elif subject_labels[0] == 1:
                filtered_labels[subject] = 1
                dprsn_counter += 1
    print 'valid sample num:',dprsn_counter,ndprsn_counter

    with open('target.pkl', 'wb') as f:
        pickle.dump(filtered_labels,f)

    print 'Finish'

if __name__ == '__main__':

    main()



