
import io
from bert_serving.client import BertClient
import numpy as np
from termcolor import colored


topk = 5 #How many similar questions to search
question_list = []
answer_list = []
question_file_path = './COVID Questions .txt'
answer_file_path = './COVID Answers.txt'

question_file = io.open(question_file_path, mode = 'r', encoding = 'utf-8')
answer_file = io.open(answer_file_path, mode = 'r', encoding='utf-8')
question = question_file.readline()

while question:
    question_list.append(question.strip())
    question = question_file.readline()

answers = answer_file.read()
answer_list = answers.split('@')

question_file.close()
answer_file.close()

# test if the two documents match by testing if the last elements in both files are the same
# print(question_list[-1])
# print(answer_list[-1])

# encode all the questions into embeddings with BERT pretrained model
bc = BertClient()
q_vecs = bc.encode(question_list)

while True:
    index = 0
    querry = input('Hi! I am your friendy bot to answer any COVID-19 related questions. Please in put your questions: ')
    querry_vc = bc.encode([querry])[0]
    score = np.sum(querry_vc * q_vecs, axis = 1) / np.linalg.norm(q_vecs, axis = 1)
    topk_idx = np.argsort(score)[::-1][:topk]
    print ('The most relevant question I can find in our database is: ')
    print('> %s\t%s' % (colored('%.1f' % score[topk_idx[index]], 'cyan'), colored(question_list[topk_idx[index]], 'yellow')))
    print(colored(answer_list[topk_idx[index]], 'yellow'))

    while True:
        to_continue = input('Does this answers your question? (y/n)')
        if to_continue == 'n':
            index = index + 1
            print('The next relevant question I can find in our database: ')
            print('> %s\t%s' % (colored('%.1f' % score[topk_idx[index]], 'cyan'), colored(question_list[topk_idx[index]], 'yellow')))
            print(colored(answer_list[topk_idx[index]], 'yellow'))
        elif to_continue == 'y':
            break
