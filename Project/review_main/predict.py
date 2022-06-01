
import time
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

from review_main import learning
from review_main import text_embed

import warnings
warnings.filterwarnings('ignore')

import itertools


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

def typo_spacing(data):
    temp = np.array([])
    for i in data.to_numpy():
        okt = Okt()
        # new = okt.normalize(data[i]) # 정규화

        new = only_hangle_number(i)
        new = emoticon_normalize(new, num_repeats=2)  # ㅋㅋㅋㅋㅋㅋ -> ㅋㅋ, ㅠㅠㅠㅠ -> ㅠㅠ

        i = new.replace(" ", '')

        spacing = Spacing()
        new = spacing(i)  # Apply space preprocessing
        try:
            new = spell_checker.check(new).checked  # 오타 처리
        except:
            print("Spell check Error occurred")
            while True:
                try:
                    time.sleep(1)
                    new = spell_checker.check(new).checked
                    print("Spell check Error resolved")
                    break
                except:
                    pass

        temp=np.append(temp,new)
    return temp

def kor_preprocessing(df):
    data=df['comment'].copy().reset_index(drop=True)

    data = typo_spacing(data)
    gc.collect()
    # 신조어 사전 추가
    adding_noun = ['식후감', '존맛', '개존맛', '꿀맛', '짱맛', '요기요', 'ㅈㅁㅌ', 'ㅃㄲ', '소확행', '민초', '치밥', '소맥', '넘사벽', '순삭', '빛삭', '광삭',
                '반반무', '반반무마니', '솔까말', '스압', '썸남', '썸녀', 'jmt', 'jmtg', 'jmtgr', 'JMT', 'JMTG', 'JMTGR', '배불띠', '돈쭐', '쿨타임', '닥추',
                '강추', '유튜버', '홧팅', '팟팅', '단짠단짠', '단짠', '맵단', '맵달', '맛도리', '부조캐', '밍밍쓰', '노맛', '존노맛', '최애', '차애', '섭스',
                '서빗', '프레젠또', '존맛탱', '개존맛탱', '존맛탱구리', '킹맛', '댕맛', '뿌링클', '로제', '오레오', '로투스', '사장님', '싸장님', '사장뉨' 
                '소소한', '프라프치노',' 프라푸치노',  '갓성비', '커엽', '굳잡', '굿잡', '굳굳', '이벵트', '이벵']

    token = Twitter()  # 추가
    for i in adding_noun:
        token.add_dictionary(i, 'Noun') # 명사 추가

    adding_verb = ['맛나', '마이쩡', '마이쪙', '마시땅', '마시쩡', '마시쪙']

    for i in adding_verb:
        token.add_dictionary(i, 'Noun') # 동사 추가

    token.add_dictionary('잘', 'Noun') # 동사 추가

    token = Okt()
    # 불용어 사전
    with open('stop.txt', 'rt', encoding='UTF8') as f:
        stopwords = f.read().replace('\n', ' ')
    stopwords = stopwords.split(' ')
    sentences=[]
    result=[]
    count=0
    for i in data:
        review = i
        temp = (token.morphs(review, norm=True, stem=True))

        stopwords_removed_sentence = [word for word in temp if not word in stopwords]  # 불용어 제거
        sentence = np.array([])
        for s in stopwords_removed_sentence:
            sentence=np.append(sentence,s)
            if s in result:
                pass
            else:
                result.append(s)

        sentences.append([sentence, df['total'].iloc[count]])
        count=count+1

    print(str(os.getpid())+" is Done")

    return result, sentences

def new_or_old_rev(dataf): #학습되지 않은 새로운 리뷰인지 판별
    sql_ = sqlite3.connect('database/Learned_review.db')
    try:
        learned_rev = pd.read_sql('SELECT * FROM Learned_review', sql_, index_col=None)
    except:
        learned_rev = pd.DataFrame()
    sql_.close()

    dataf = dataf[dataf.duplicated(keep='last') == False]

    not_inner=pd.concat([dataf,learned_rev])
    not_inner=pd.concat([dataf,not_inner[not_inner.duplicated(keep='last')]]) #중복확인을 위해 concat -> 중복인 경우 이미 학습
    not_inner = not_inner[not_inner.duplicated(keep=False)==False] #중복되지 않는 리뷰들만 필터링

    print("{} reviews exist, {} reviews are new\n".format(len(dataf)-len(not_inner),len(not_inner)))
    return not_inner

def predict_data(dataf,word_lib):
    learning_data = []
    total_word = []
    word,sent = kor_preprocessing(dataf)

    diff = list(set(word) - set(total_word))  # 중복되지 않는 단어 확인
    for i in diff:
        total_word.append(i)

    learning_data = learning_data + sent

    dtm_total = pd.DataFrame()
    dtm_total['word'] = total_word

    ori_in = [i for i in word_lib.index
              if word_lib['word'].loc[i] in dtm_total['word'].values]  # 기존 단어장에서 중복단어들의 인덱스값 필터링
    dtm_in = dtm_total.loc[list(itertools.chain(
        *[dtm_total[dtm_total['word'] == word_lib['word'].loc[i]].index.tolist() for i in
          ori_in])), dtm_total.columns != 'word']  # 학습 대상 리뷰들의 단어들 중 기존 단어장에 존재하는 단어들 필터링

    word_lib.loc[ori_in, word_lib.columns != 'word'] = word_lib.loc[ori_in, word_lib.columns != 'word'].reset_index(
        drop=True) + dtm_in.reset_index(drop=True)  # 기존 단어장에 학습되었던 단어의 평점, 빈도 갱신

    dtm_diff = [i for i in dtm_total.index
                if dtm_total['word'].loc[i] not in word_lib['word'].values]  # 학습되지 않은 새로운 단어들 필터링

    if dtm_diff != 0:
        word_lib = pd.concat([word_lib, dtm_total.loc[dtm_diff]]).reset_index(drop=True)  # 새로운 단어가 있다면 기존 단어장에 추가

    print("{} words exist, {} words are new\n".format(len(ori_in), len(dtm_diff)))

    word2 = text_embed.word2index(word_lib)

    y_label = []
    x_value = []
    for i in range(len(learning_data)):
        y_label.append(int(learning_data[i][1]))
        x_value.append(text_embed.sen2index(learning_data[i][0], word2))

    x_value = pd.DataFrame(x_value).fillna(0).astype('float')  # ,columns=list(range(100)))
    y_label = pd.DataFrame(y_label, columns=['label']).astype('int')

    for i in range(len(x_value.columns), 100):
        x_value[i] = 0
    x_value = x_value.values.tolist()
    y_label = y_label.values.tolist()

    pred = learning.start_learn(x_value, y_label, len(word_lib), 100,"val")

    sql_ = sqlite3.connect('database/DTM_Words.db')
    word_lib.to_sql('DTM_Words', sql_, if_exists='replace', index=False)
    sql_.close()
    print(pred)
    return pred

def learn_data(dataf,word_lib,rest):
    dataf['rest'] = rest
    dataf = new_or_old_rev(dataf)  # 학습되지 않은 새로운 리뷰인지 판별

    total_word = []
    if len(dataf) == 0:  # 0인 경우 학습 대상인 리뷰 없음
        print("Already new data\n")
        return

    learning_data = []
    _, sent = kor_preprocessing(dataf)
    learning_data = learning_data + sent
    word2 = text_embed.word2index(word_lib)

    y_label = []
    x_value = []
    for i in range(len(learning_data)):
        y_label.append(int(learning_data[i][1]))
        x_value.append(text_embed.sen2index(learning_data[i][0], word2))

    x_value = pd.DataFrame(x_value).fillna(0).astype('float')  # ,columns=list(range(100)))
    y_label = pd.DataFrame(y_label, columns=['label']).astype('int')

    for i in range(len(x_value.columns), 100):
        x_value[i] = 0

    print("Dataset loaded")
    print("Start Learning...")

    x_value = x_value.values.tolist()
    y_label = y_label.values.tolist()
    learning.start_learn(x_value, y_label, len(word_lib), 100)

    sql_ = sqlite3.connect('database/Learned_review.db')
    dataf.to_sql('Learned_review', sql_, if_exists='append', index=False)
    sql_.close()


def predict(df,star,rest=None):
    sql_ = sqlite3.connect('database/DTM_Words.db')
    try:
        origin = pd.read_sql("SELECT * FROM DTM_Words", sql_, index_col=None)
    except:
        origin = pd.DataFrame(columns=['word'])
    sql_.close()

    dataf = pd.DataFrame()
    dataf["comment"]=df
    dataf["total"]=star
    if rest==None:
        return predict_data(dataf,origin)
    else:
        learn_data(dataf,origin,rest)