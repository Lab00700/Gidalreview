import gc
import sqlite3
import pandas as pd
import numpy as np
import os
from multiprocessing import Process, Queue

from pykospacing import Spacing
from konlpy.tag import Okt
from soynlp.normalizer import *
from ckonlpy.tag import Twitter
from hanspell import spell_checker
from sklearn.feature_extraction.text import TfidfVectorizer

import warnings
warnings.filterwarnings('ignore')

import joblib


database_list = ['crowling_all',
                 'non_del',
                 'non_pic',
                 'non_del_pic',
                 'total_5',
                 'total_34',
                 'total_12',
                 'taste_5',
                 'taste_34',
                 'taste_12',
                 'quan_5',
                 'quan_34',
                 'quan_12',
                 'deli_5',
                 'deli_34',
                 'deli_12']


def kor_preprocessing(q, q3, df):
    data = df.copy().reset_index(drop=True)
    temp = []

    data = data.str.join('').str.replace(r"\n", "")
    data = data.str.replace(pat=r'[^\w]', repl=r'', regex=True)

    for i in range(len(data)):
        okt = Okt()
        new = okt.normalize(data[i])  # 정규화

        new = only_hangle(new)
        new = emoticon_normalize(new, num_repeats=2)  # ㅋㅋㅋㅋㅋㅋ -> ㅋㅋ, ㅠㅠㅠㅠ -> ㅠㅠ

        data[i] = data[i].replace(" ", '')

        spacing = Spacing()
        new = spacing(data[i])  # Apply space preprocessing
        try:
            new = spell_checker.check(new).checked  # 오타 처리
        except:
            print(new)
        temp.append(new)

    data = pd.Series(temp)

    # 신조어 사전 추가
    token = Twitter()  # 추가
    adding_noun = ['식후감', '존맛', '개존맛', '꿀맛', '짱맛', '요기요', 'ㅈㅁㅌ', 'ㅃㄲ', '소확행', '민초', '치밥', '소맥', '넘사벽', '순삭', '빛삭',
                   '광삭',
                   '반반무', '반반무마니', '솔까말', '스압', '썸남', '썸녀', 'jmt', 'jmtg', 'jmtgr', 'JMT', 'JMTG', 'JMTGR', '배불띠',
                   '돈쭐',
                   '쿨타임', '닥추',
                   '강추', '유튜버', '홧팅', '팟팅', '단짠단짠', '단짠', '맵단', '맵달', '맛도리', '부조캐', '밍밍쓰', '노맛', '존노맛', '최애', '차애',
                   '섭스',
                   '서빗', '프레젠또', '존맛탱', '개존맛탱', '존맛탱구리', '킹맛', '댕맛', '뿌링클', '로제', '오레오', '로투스', '사장님', '싸장님', '사장뉨'
                                                                                                              '소소한',
                   '프라프치노', ' 프라푸치노', '갓성비', '커엽', '굳잡', '굿잡', '굳굳', '이벵트', '이벵']

    for i in adding_noun:
        token.add_dictionary(i, 'Noun')  # 명사 추가

    adding_verb = ['맛나', '마이쩡', '마이쪙', '마시땅', '마시쩡', '마시쪙']

    for i in adding_verb:
        token.add_dictionary(i, 'Noun')  # 동사 추가

    token.add_dictionary('잘', 'Noun')  # 동사 추가

    token = Okt()
    # 불용어 사전
    with open('stop.txt', 'rt', encoding='UTF8') as f:
        stopwords = f.read().replace('\n', ' ')
    stopwords = stopwords.split(' ')

    result = []
    for i in range(len(data)):
        review = data[i]
        temp = (token.morphs(review, norm=True, stem=True))

        stopwords_removed_sentence = [word for word in temp if not word in stopwords]  # 불용어 제거
        sentence = ''

        for s in stopwords_removed_sentence:
            sentence = sentence + ' ' + s
        result.append(sentence)
    q.put(result)
    q3.put(df)

def predict_data(dataf):
    core = 4

    q = Queue()
    q3 = Queue()
    p_list = []
    for c in range(1, core + 1):
        p = Process(target=kor_preprocessing,
                    args=(
                        q, q3, dataf.loc[int((len(dataf) * (c - 1)) / core):int((len(dataf) * c) / core) - 1],))
        p.start()
        p_list.append(p)

    review = []
    t_comment = []
    for _ in range(len(p_list)):
        review.extend(q.get())
        t_comment.extend(q3.get())

    for _ in p_list:
        _.join()

    words = review
    t_comment = pd.DataFrame(t_comment, columns=['comment'])
    ttt = TfidfVectorizer()
    tfidf = ttt.fit_transform(review)
    tfidf_df = pd.DataFrame(tfidf.toarray(), columns=ttt.vocabulary_)
    return words, tfidf_df, t_comment

def predict(df):
    print("Predict DB File Loading...")
    words, tfidf_df, t_comment = predict_data(df)
    total_word = words
    total_review = tfidf_df.copy()
    comment = t_comment.copy()
    print("Predict DB File Loading is Done.\n")
    sql = sqlite3.connect('./learn_data.db')
    data = pd.read_sql("SELECT * FROM Learn_Columns", sql, index_col=None)

    data = list(data['word'])

    print("Learn Columns Loading.\n")
    temp_review = pd.DataFrame(columns=data)
    total_review = pd.concat([temp_review, total_review], axis=0)
    total_review = total_review[data]
    total_review = total_review.fillna(0)
    print(total_review.columns)
    label = total_review['total'].astype('int8')
    total_review = total_review.drop(['total'], axis=1)
    print(label.value_counts())

    print("\nStart Predict...\n")

    model = joblib.load('./best_model.pkl')
    comment['predict'] = model.predict(total_review)
    comment['predict'] = comment['predict'].astype('int32')
    comment=pd.concat([comment, pd.DataFrame(total_word, columns=['words'])], axis=1)
    comment['review']=comment['comment']
    return pd.merge(df,comment,how='outer',on='review')