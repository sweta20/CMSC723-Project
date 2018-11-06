import json
import pandas as pd
import argparse
import random

# categories = { 
#     0: ['History'],
#     1: ['Literature'],
#     2: ['Mythology', 'Philosophy', 'Religion', 'Social Science', 'Geography'],
#     3: ['Science'],
#     4: ['Current Events', 'Trash', 'Fine Arts']
#     }

categories = {
    0: ['History', 'Philosophy', 'Religion'],
    1: ['Literature', 'Mythology'],
    2: ['Science', 'Social Science'],
    3: ['Current Events', 'Trash', 'Fine Arts', 'Geography']
}

def get_data():
    input_file='data/qanta.train.2018.04.18.json'
    with open(input_file, 'r') as f:
        data_dict = json.load(f)
    questions = data_dict['questions']
    question_df = pd.DataFrame.from_dict(questions)
    return question_df

def assign(n_workers=4, order='all', categories=categories):
    questions = get_data()
    assignment = {}
    ids = list(questions['qanta_id'])

    if order=='all':
        for i in range(n_workers):
            assignment[i] = ids

    elif order=='random':
        random.shuffle(ids)
        split = int(len(ids)/n_workers)
        for i in range(n_workers):
            assignment[i] = ids[i*split:(i+1)*split]

        # add leftover questions
        assignment[n_workers-1] = assignment[n_workers-1] + ids[n_workers*split:]

    elif order=='category':
        question_groups = questions.groupby(['category'])
        for i in range(n_workers):
            ids = []
            for category in categories[i]:
                category_ids = list(question_groups.get_group(category)['qanta_id'])
                ids = ids + category_ids
            assignment[i] = ids

    return assignment

# if __name__ == '__main__':
#     a = assign(order='category')
#     print(len(a[0]))
#     print(len(a[1]))
#     print(len(a[2]))
#     print(len(a[3]))
    
    