import gc
import sqlite3
import pandas as pd
import os
import torch
import itertools
import kor_preprocessing
import learning
import text_embed

import warnings
warnings.filterwarnings('ignore')

category = ['chicken',
            'chicken_franchise',
            'Pizza_WesternFood',
            'Pizza_WesternFood_franchise',
            'ChineseFood',
            'ChineseFood_franchise',
            'KoreanFood',
            'KoreanFood_franchise',
            'JapaneseFood',
            'JapaneseFood_franchise',
            'Jokbal_Bossam',
            'Jokbal_Bossam_franchise',
            'SchoolFood',
            'SchoolFood_franchise',
            'Cafe',
            'Cafe_franchise']

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

def dtm(dataf, word_lib, rest):
    dataf['rest'] = rest
    dataf = new_or_old_rev(dataf) #학습되지 않은 새로운 리뷰인지 판별

    total_word = []
    if len(dataf) == 0:  # 0인 경우 학습 대상인 리뷰 없음
        print("Already new data\n")
        return

    learning_data = []
    print("Start Preprocessing...")
    word,sent=kor_preprocessing.kor_preprocessing(dataf)

    diff = list(set(word) - set(total_word))  # 중복되지 않는 단어 확인
    for i in diff:
        total_word.append(i)

    learning_data = learning_data + sent  # 리뷰 토큰화 저장

    print("Preprocessing is done.")
    dtm_total = pd.DataFrame()
    dtm_total['word'] = total_word

    ori_in = [i for i in word_lib.index
              if word_lib['word'].loc[i] in dtm_total['word'].values] #기존 단어장에서 중복단어들의 인덱스값 필터링
    dtm_in = dtm_total.loc[list(itertools.chain(
        *[dtm_total[dtm_total['word'] == word_lib['word'].loc[i]].index.tolist() for i in
          ori_in])), dtm_total.columns != 'word'] #학습 대상 리뷰들의 단어들 중 기존 단어장에 존재하는 단어들 필터링

    word_lib.loc[ori_in, word_lib.columns != 'word'] = word_lib.loc[ori_in, word_lib.columns != 'word'].reset_index(
        drop=True) + dtm_in.reset_index(drop=True) #기존 단어장에 학습되었던 단어의 평점, 빈도 갱신

    dtm_diff = [i for i in dtm_total.index
                if dtm_total['word'].loc[i] not in word_lib['word'].values] #학습되지 않은 새로운 단어들 필터링

    if dtm_diff != 0:
        word_lib = pd.concat([word_lib, dtm_total.loc[dtm_diff]]).reset_index(drop=True) #새로운 단어가 있다면 기존 단어장에 추가

    print("{} words exist, {} words are new\n".format(len(ori_in),len(dtm_diff)))

    word2 = text_embed.word2index(word_lib)

    y_label=[]
    x_value=[]
    for i in range(len(learning_data)):
        y_label.append(int(learning_data[i][1]))
        x_value.append(text_embed.sen2index(learning_data[i][0],word2))

    x_value=pd.DataFrame(x_value).fillna(0).astype('float')#,columns=list(range(100)))
    y_label=pd.DataFrame(y_label,columns=['label']).astype('int')

    for i in range(len(x_value.columns),100):
        x_value[i]=0

    print("Dataset loaded")
    print("Start Learning...")

    x_value = x_value.values.tolist()
    y_label = y_label.values.tolist()
    learning.start_learn(x_value,y_label,len(word_lib), 100)

    sql_ = sqlite3.connect('database/DTM_Words.db')
    word_lib.to_sql('DTM_Words', sql_, if_exists='replace', index=False)
    sql_.close()

    sql_ = sqlite3.connect('database/Learned_review.db')
    dataf.to_sql('Learned_review', sql_, if_exists='append', index=False)
    sql_.close()

if __name__ == '__main__':
    for u in category:  # 카테고리별
        path = 'Crowling-Data/' + u + "/"
        db_list = os.listdir(path)
        print(u)
        print(db_list)
        cate_word_pf = {}

        for db in db_list:  # 음식점별

            sql_ = sqlite3.connect('database/DTM_Words.db')
            try:
                origin = pd.read_sql("SELECT * FROM DTM_Words",sql_, index_col=None)
            except:
                origin = pd.DataFrame(columns=['word'])
            sql_.close()
            print(db[:len(db)-3])

            db_file = sqlite3.connect('Crowling-Data/' + u + "/" + db)
            dataf = pd.DataFrame()
            for t in database_list[:4]:  # db목록 맨앞 4개까지만 불러오기
                sql_to_df = pd.read_sql("SELECT * FROM " + str(t), db_file, index_col=None)
                dataf = pd.concat([dataf, sql_to_df]).reset_index(drop=True)
            print(dataf.columns)
            dataf['comment'] = dataf['comment'].str.join('')    .str.replace(r"\n", "")
            dataf['comment'] = dataf['comment'].str.replace(pat=r'[^\w]', repl=r'', regex=True)

            dtm(dataf[['total','comment']], origin, db[:len(db)-3])