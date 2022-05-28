from pykospacing import Spacing
from konlpy.tag import Okt
from soynlp.normalizer import *
from ckonlpy.tag import Twitter
from hanspell import spell_checker
import pandas as pd
import os
import time
import gc
def typo_spacing(data):
    temp = []
    for i in range(len(data)):
        okt = Okt()
        # new = okt.normalize(data[i]) # 정규화

        new = only_hangle_number(data[i])
        new = emoticon_normalize(new, num_repeats=2)  # ㅋㅋㅋㅋㅋㅋ -> ㅋㅋ, ㅠㅠㅠㅠ -> ㅠㅠ

        data[i] = new.replace(" ", '')

        spacing = Spacing()
        new = spacing(data[i])  # Apply space preprocessing
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

        temp.append(new)
    return temp

def kor_preprocessing(df):
    data=df['comment'].copy().reset_index(drop=True)

    data = pd.Series(typo_spacing(data))
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
    with open('source/stop.txt', 'rt', encoding='UTF8') as f:
        stopwords = f.read().replace('\n', ' ')
    stopwords = stopwords.split(' ')

    sentences=[]
    result={}
    score={}
    for i in range(len(data)):
        review = data[i]
        temp = (token.morphs(review, norm=True, stem=True))

        stopwords_removed_sentence = [word for word in temp if not word in stopwords]  # 불용어 제거
        sentence = []
        for s in stopwords_removed_sentence:
            sentence.append(s)
            if s in result.keys():
                result[s]=result[s]+1
                score[s]=score[s]+int(df['total'].iloc[i])
            else:
                result[s]=1
                score[s]=int(df['total'].iloc[i])

        sentences.append([sentence, df['total'].iloc[i]])

    print(str(os.getpid())+" is Done")

    return result, score, sentences