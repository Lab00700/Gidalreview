from django.db.models import Q
from django.shortcuts import render
from django.views.generic import TemplateView, ListView
from .models import eventModel, SimilarComment, Rest_Info, sorting
import re
from selenium import webdriver
import time
from bs4 import BeautifulSoup
import datetime
import os.path
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline, pipeline
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from ckonlpy.tag import Twitter
from konlpy.tag import Okt
import ast
import matplotlib.font_manager as fm
from PIL import Image
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from wordcloud import WordCloud
import nltk

from review_main import predict



# Create your views here.
def home(request):
    if request.method == 'GET':
        review = eventModel.objects.all()
        return render(request, 'home.html')


def url_search(request):
    if request.method == 'GET':
        sort = request.GET.get('sort', '')
        f = request.GET.getlist('f')

        total = pd.DataFrame()
        total = pd.concat([total, sorting.label], axis=1)
        total.columns = ['label']
        total = pd.concat([total, sorting.review], axis=1)
        total = pd.concat([total, sorting.date], axis=1)
        total = pd.concat([total, sorting.expectedPoint], axis=1)
        total = total.dropna(subset=['date'], axis=0)



        print(sort)
        if sort == 'predict_high':
            total = total.sort_values('predict').reset_index(drop=True)
            total = total[::-1].reset_index(drop=True)
        elif sort == 'predict_low':
            total = total.sort_values('predict').reset_index(drop=True)

        print(total['predict'])
        print(total['date'])
        print(total.columns)

        Rest_Info_model = Rest_Info()
        Rest_Info_model.star_point = Rest_Info.star_point
        Rest_Info_model.star = Rest_Info.star
        Rest_Info_model.expected_star_point = Rest_Info.expected_star_point
        Rest_Info_model.expected_star = Rest_Info.expected_star
        Rest_Info_model.rest_name = Rest_Info.rest_name
        Rest_Info_model.review_notice = Rest_Info.review_notice

        total_filter = total.copy()
        # 이벤트 리뷰 필터링
        if len(f)==int(1):
            condition = (total_filter.label == f[0])
            total_filter = total_filter[condition]


        similar_list = zip(list(sorting.star), list(sorting.comment), list(sorting.similarComment1),
                           list(sorting.similarComment2))
        event_list = zip(total_filter['predict'].tolist(), total_filter['review'].tolist(), total_filter['label'].tolist())
        # context = {'review_list': review_model.review, 'star_list': review_model.star, 'pred_review_list':review_model.pred_review, 'expectedPoint_list':review_model.expectedPoint, 'label_list':review_model.label,
        # context = {'review_list': review_model.review, 'star_list': review_model.star, 'label_list':review_model.label, 'expectedPoint_list':review_model.expectedPoint,
        #             'Rest_Info':Rest_Info_model,
        #             'representing_comment_list':similar_model.comment, 'representing_comment_star_list':similar_model.star, 'similar_list_1':similar_model.similarComment1, 'similar_list_2':similar_model.similarComment2,'schedule_list':schedule_list}
        context = {'Rest_Info': Rest_Info_model, 'similar_list': similar_list, 'event_list': event_list}
        # ################################################################# 수정

        return render(request, 'search_results.html', context)

    if request.method == 'POST':
        url_src = request.POST.get("url_src", '')
        if 'yogiyo' in url_src:
            # 크롤링 코드, result.html로 보내기
            review_data = lets_do_crawling(url_src)
            prepro = review_data.copy()
            review_model = eventModel.objects.all()
            review_model.review_id = review_data['review_id']
            review_model.review = review_data['review']
            review_model.date = review_data['date']
            review_model.star = review_data['star']
            print(review_model.review)

            review_model.label = pd.Series(eventReview(prepro))
            print(review_model.label)

            # 예측 평점 모델 적용
            pred = predict.predict(review_model.review)

            rest_info = rest_info_crowling(url_src, pred['predict'])
            print(rest_info)
            print(review_model.review)
            print(review_model.label)
            print('**********************pred_predict**********************')
            print(pred['predict'])
            eventModel.expectedPoint = pred['predict']
            # review_model.review = pd.Series((v[0] for v in eventReview(example)))
            # print(review_model.review)

            # 가게 정보
            Rest_Info.star_point = rest_info['star_point'].loc[0]
            Rest_Info.star = rest_info['star'].loc[0]
            Rest_Info.expected_star_point = rest_info['expected_star_point'].loc[0]
            Rest_Info.expected_star = rest_info['expected_star'].loc[0]
            Rest_Info.rest_name = rest_info['rest_name'].loc[0]
            Rest_Info.review_notice = rest_info['review_notice'].loc[0]

            Rest_Info_model = Rest_Info()
            Rest_Info_model.star_point = Rest_Info.star_point
            Rest_Info_model.star = Rest_Info.star
            Rest_Info_model.expected_star_point = Rest_Info.expected_star_point
            Rest_Info_model.expected_star = Rest_Info.expected_star
            Rest_Info_model.rest_name = Rest_Info.rest_name
            Rest_Info_model.review_notice = Rest_Info.review_notice

            # 유사 리뷰
            sim_review_data = findSimilarReview(review_data)
            representing_review_data = findRepresentingReview(sim_review_data)

            similar_model = SimilarComment()
            similar_model.star = representing_review_data['star']
            similar_model.comment = representing_review_data['comment']
            similar_model.preprocessingComment = representing_review_data['preprocessing_comment']
            similar_model.similarComment1 = representing_review_data['similar_comment_1']
            similar_model.similarity1 = representing_review_data['similarity_1']
            similar_model.similarComment2 = representing_review_data['similar_comment_2']
            similar_model.similarity2 = representing_review_data['similarity_2']

            sorting.review = review_model.review
            sorting.similarComment1 = similar_model.similarComment1
            sorting.similarComment2 = similar_model.similarComment2
            sorting.expectedPoint = eventModel.expectedPoint
            sorting.date = review_data['date']
            sorting.label = review_model.label
            sorting.star = similar_model.star
            sorting.comment = similar_model.comment

            similar_list = zip(list(sorting.star), list(sorting.comment), list(sorting.similarComment1),
                               list(sorting.similarComment2))
            event_list = zip(list(sorting.expectedPoint), list(sorting.review), list(sorting.label))
            f = ['리뷰 이벤트 참여 가능성 높음', '리뷰 이벤트 참여 가능성 낮음']
            # context = {'review_list': review_model.review, 'star_list': review_model.star, 'pred_review_list':review_model.pred_review, 'expectedPoint_list':review_model.expectedPoint, 'label_list':review_model.label,
            # context = {'review_list': review_model.review, 'star_list': review_model.star, 'label_list':review_model.label, 'expectedPoint_list':review_model.expectedPoint,
            #             'Rest_Info':Rest_Info_model,
            #             'representing_comment_list':similar_model.comment, 'representing_comment_star_list':similar_model.star, 'similar_list_1':similar_model.similarComment1, 'similar_list_2':similar_model.similarComment2,'schedule_list':schedule_list}
            context = {'Rest_Info': Rest_Info_model, 'similar_list': similar_list, 'event_list': event_list,
                       'filter': f}
            # ################################################################# 수정

            return render(request, 'search_results.html', context)
        elif 'yogiyo' not in url_src:
            return render(request, 'home.html',
                          {'error01': '요기요 URL을 입력해주세요!'})
        else:
            return render(request, 'home.html',
                          {'error01': 'URL을 입력해주세요!'})


def rest_info_crowling(url_src, pred):
    options = webdriver.ChromeOptions()
    # 창 숨기는 옵션 추가
    options.add_argument("headless")
    # driver 실행
    # driver = webdriver.Chrome(r"C:\chromedriver.exe", options=options)
    driver = webdriver.Chrome(r"C:\chromedriver.exe", options=options)
    driver.get(str(url_src))

    er_t = 0
    error = True
    while error:
        try:
            error = False
            rev = driver.find_element_by_xpath("//*[@id='content']/div[2]/div[1]/ul/li[3]/a")
            rev.click()
        except:
            print("정보 Click Error. Try again...")
            error = True
            time.sleep(5)
            er_t = er_t + 1
            if er_t == 5:
                return "error"

    time.sleep(0.2)

    er_t = 0
    error = True
    while error:
        try:
            error = False
            soup = BeautifulSoup(driver.page_source, "html.parser")
        except:
            print("Restaurant info Crowling Error. Try again...")
            error = True
            time.sleep(5)
            er_t = er_t + 1
            if er_t == 5:
                return "error"

    star_point = float(soup.find("strong", attrs={"class": "ng-binding"}).get_text())
    star = int(star_point)
    if star == 0:
        star = ''
    elif star == 1:
        star = '★'
    elif star == 2:
        star = '★★'
    elif star == 3:
        star = '★★★'
    elif star == 4:
        star = '★★★★'
    elif star == 5:
        star = '★★★★★'

    expected_star_point = round(pred.mean(), 2)  # 리뷰들 예측평점의 평균 구하기
    expected_star = int(expected_star_point)

    if expected_star == 0:
        expected_star = ''
    elif expected_star == 1:
        expected_star = '★'
    elif expected_star == 2:
        expected_star = '★★'
    elif expected_star == 3:
        expected_star = '★★★'
    elif expected_star == 4:
        expected_star = '★★★★'
    elif expected_star == 5:
        expected_star = '★★★★★'

    rest_name = soup.find("span", attrs={"class": "restaurant-name ng-binding"}).get_text()
    review_notice = soup.find("div", attrs={"class": "info-text ng-binding"}).get_text()

    rest_info = pd.DataFrame(
        columns=['star_point', 'star', 'expected_star_point', 'expected_star', 'rest_name', 'review_notice'])
    rest_info.loc[0] = [star_point, star, expected_star_point, expected_star, rest_name, review_notice]
    print(rest_info)

    return rest_info


def lets_do_crawling(url_src):
    options = webdriver.ChromeOptions()
    # 창 숨기는 옵션 추가
    options.add_argument("headless")
    # driver 실행
    # driver = webdriver.Chrome(r"C:\chromedriver.exe", options=options)
    driver = webdriver.Chrome(r"C:\chromedriver.exe", options=options)
    page = 5
    driver.get(str(url_src))

    i = 0
    k = 0
    time.sleep(0.3)

    er_t = 0
    error = True
    while error:
        try:
            error = False
            rev = driver.find_element_by_xpath("//*[@id='content']/div[2]/div[1]/ul/li[2]/a")
            rev.click()
        except:
            print("클린리뷰 Click Error. Try again...")
            error = True
            time.sleep(5)
            er_t = er_t + 1
            if er_t == 5:
                return "error"

    time.sleep(0.3)

    er_t = 0
    error = True
    while error:
        try:
            error = False
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
        except:
            print("Scroll down Error. Try again...")
            error = True
            time.sleep(5)
            er_t = er_t + 1
            if er_t == 5:
                return "error"

    time.sleep(0.3)

    run = True

    review_data = pd.DataFrame(columns=['review_id', 'review', 'date', 'star'])

    while i < page:
        com_time = ''
        comment = ''

        if i != 0:
            try:  # 더보기 버튼이 존재할 때 리뷰 불러오기
                driver.find_element_by_xpath("//*[@id='review']/li[" + str(i) + "2]")
                rev = driver.find_element_by_xpath("//*[@id='review']/li[" + str(i) + "2]")
                time.sleep(0.4)
                rev.click()
            except:  # 리뷰 전체를 불러왔기에 더보기 버튼이 없을 때
                run = False

            time.sleep(0.4)

            er_t = 0
            error = True
            while error:
                try:
                    error = False
                    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
                except:
                    print("Scroll down Error. Try again...")
                    error = True
                    time.sleep(5)
                    er_t = er_t + 1
                    if er_t == 5:
                        return "error"
        i = i + 1

        er_t = 0
        error = True
        while error:
            try:
                error = False
                soup = BeautifulSoup(driver.page_source, "html.parser")
            except:
                print("Reviews Crowling Error. Try again...")
                error = True
                time.sleep(5)
                er_t = er_t + 1
                if er_t == 5:
                    return "error"

        if run == False:  # 크롤링 중지
            break

    time.sleep(0.2)
    reviews = soup.find_all("li", attrs={"class": "list-group-item star-point ng-scope"})

    current_time = datetime.datetime.now()  # 현재 시간
    for review in reviews:
        try:
            temp_time = review.find("span",
                                    attrs={"class": "review-time ng-binding",
                                           "ng-bind": "review.time|since"}).get_text()  # 시간
            if temp_time == "일주일 전":
                t_time = current_time - datetime.timedelta(days=7)
                temp_time = str(t_time.year) + "년 " + str(t_time.month) + "월 " + str(t_time.day) + "일"
            elif temp_time == "어제":
                t_time = current_time - datetime.timedelta(days=1)
                temp_time = str(t_time.year) + "년 " + str(t_time.month) + "월 " + str(t_time.day) + "일"
            elif "일 전" in temp_time:
                before = re.findall("\d+", temp_time)
                t_time = current_time - datetime.timedelta(days=int(before[0]))
                temp_time = str(t_time.year) + "년 " + str(t_time.month) + "월 " + str(t_time.day) + "일"
            elif "시간 전" in temp_time:
                before = re.findall("\d+", temp_time)
                t_time = current_time - datetime.timedelta(hours=int(before[0]))
                temp_time = str(t_time.year) + "년 " + str(t_time.month) + "월 " + str(t_time.day) + "일"
            elif "분 전" in temp_time:
                before = re.findall("\d+", temp_time)
                t_time = current_time - datetime.timedelta(minutes=int(before[0]))
                temp_time = str(t_time.year) + "년 " + str(t_time.month) + "월 " + str(t_time.day) + "일"
            elif "지금 등록" in temp_time:
                t_time = current_time
                temp_time = str(t_time.year) + "년 " + str(t_time.month) + "월 " + str(t_time.day) + "일"
            com_time = temp_time
            total = 0
            for cnt in range(5):
                if str(review.select("div")[2].select("span")[
                           cnt + 1]) == "<span class=\"full ng-scope\" ng-repeat=\"i in review.rating|number_to_array track by $index\">★</span>":
                    total += 1
            star = str(total)
            comment = review.find("p", attrs={"ng-show": "review.comment"}).get_text()  # 리뷰 코멘트
            review_data.loc[len(review_data)] = [k, comment, com_time, star]  # 크롤링된 데이터 1행씩 이어쓰기
            k = k + 1
        except:  # 리뷰 수집과정에서 리뷰 인식 못했을 경우 어떤 리뷰인지 출력하고 pass(드문 경우)
            print("Error review Pass : " + str(review) + " " + str(com_time))
            k = k + 1

    review_data = review_data[::-1].reset_index(drop=True, inplace=False)  # 역순정렬 -> 갱신된 리뷰 데이터들 이어붙이기 편하게

    return review_data


def use_split_join(data):
    string = ' '.join(data.split())
    return string


def eventReview(review_data):
    # 전처리
    review_data['review'] = review_data['review'].str.join('').str.replace(r"\n", "")
    review_data['review'] = review_data['review'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)
    review_data['review'] = review_data['review'].apply(use_split_join)

    example = review_data['review'].tolist()

    ## sentiment analysis
    koelectra_finetuned_model_dir = os.path.join('', "koelectra-review-finetune.bin")
    tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model = AutoModelForSequenceClassification.from_pretrained("monologg/koelectra-base-v3-discriminator")
    model.load_state_dict(torch.load(koelectra_finetuned_model_dir, map_location=torch.device('cpu')))
    sentiment_classifier = pipeline('sentiment-analysis', tokenizer=tokenizer, model=model)

    y_pred = []
    total_len = len(example)
    for cnt, review in enumerate(example):

        pred = sentiment_classifier(review)
        # print(f"{cnt} / {total_len} : {pred[0]}")

        if pred[0]['label'] == 'LABEL_1':
            y_pred.append("리뷰 이벤트 참여 가능성 높음")
        else:
            y_pred.append("리뷰 이벤트 참여 가능성 낮음")

    return y_pred


def make_token(data):
    # 신조어 사전 추가
    token = Twitter()  # 추가
    adding_noun = ['식후감', '존맛', '개존맛', '꿀맛', '짱맛', '요기요', 'ㅈㅁㅌ', 'ㅃㄲ', '소확행', '민초', '치밥', '소맥', '넘사벽', '순삭', '빛삭', '광삭',
                   '반반무', '반반무마니', '솔까말', '스압', '썸남', '썸녀', 'jmt', 'jmtg', 'jmtgr', 'JMT', 'JMTG', 'JMTGR', '배불띠', '돈쭐',
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

    # 불용어 사전
    with open('stop.txt', 'rt', encoding='UTF8') as f:
        stopwords = f.read().replace('\n', ' ')
    stopwords = stopwords.split(' ')

    temp = (token.morphs(data, norm=True, stem=True))

    stopwords_removed_sentence = [word for word in temp if not word in stopwords]  # 불용어 제거

    return stopwords_removed_sentence


def make_doc_embed_list(model, test_tokenized_data):
    document_embedding_list = []  # 단어 벡터의 평균 구하기
    # 리뷰에 존재하는 단어들의 벡터값의 평균을 구하여 해당 리뷰의 벡터값을 연산

    # 각 리뷰에 대해서
    for idx, review in enumerate(test_tokenized_data):
        doc2vec = None
        cnt = 0
        for word in review:
            if word in list(model.wv.index_to_key):
                cnt += 1
                # 해당 리뷰에 있는 모든 단어들의 벡터값을 더한다.
                if doc2vec is None:
                    doc2vec = model.wv.__getitem__(word)
                else:
                    doc2vec = doc2vec + model.wv.__getitem__(word)

        # 단어 벡터를 모두 더한 벡터의 값을 문서 길이로 나눠준다.
        if doc2vec is not None:
            doc2vec = doc2vec / cnt
        else:
            doc2vec = np.zeros(100, )

        document_embedding_list.append(doc2vec)

    # 각 리뷰에 대한 리뷰 벡터 리스트를 리턴
    return document_embedding_list


def use_split_join(data):
    string = ' '.join(data.split())
    return string


def findSimilarReview(data):
    data['review'] = data['review'].str.join('').str.replace(r"\n", "")  # 줄바꿈 제거
    data['temp'] = data['review'].copy()

    # 전처리
    data['temp'] = data['temp'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)  # 특수문자를 공백으로
    data['temp'] = data['temp'].apply(use_split_join)  # 여러개 공백을 하나로 바꿈

    # 전처리 후 키워드 시각화(graph, wordcloud)
    visualize(data)

    # 토큰화
    data['token'] = data.temp.map(lambda x: make_token(str(x)))

    tokenized_data = data.token.values.tolist()
    data.token = data.token.map(lambda x: ast.literal_eval(str(x)))
    print('토큰화후', data['review'])

    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=2, min_count=5, workers=4, sg=1)
    # vector_size = 워드 벡터의 특징 값. 즉, 임베딩 된 벡터의 차원.
    # window = 컨텍스트 윈도우 크기
    # min_count = 단어 최소 빈도 수 제한 (빈도가 적은 단어들은 학습하지 않는다.)
    # workers = 학습을 위한 프로세스 수
    # sg = 0은 CBOW, 1은 Skip-gram.

    model.save('word2vec.model')

    document_embedding_list = make_doc_embed_list(model, tokenized_data)  # 단어 벡터의 평균 구하기
    cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)  # 코사인 유사도 구하기
    print(cosine_similarities)

    # 새 테이블로 저장
    new_table = pd.DataFrame(
        columns=['comment', 'star', 'preprocessing_comment', 'similar_comment_1', 'similarity_1', 'similar_comment_2',
                 'similarity_2'])
    new_table['comment'] = data['review'].copy()
    new_table['star'] = data['star'].copy()
    new_table['preprocessing_comment'] = data['temp'].copy()

    # 가장 유사한 리뷰 찾기
    for index in range(0, len(data)):
        sim_score = list(enumerate(cosine_similarities[index]))  # 모든 리뷰와 해당 리뷰의 유사도 구하기
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)  # 유사도 높은 순서로 정렬
        sim_reivew_idx = [i for i in sim_score[1:3]]  # 유사 리뷰 2개 출력 (index 0은 해당리뷰와 해당리뷰의 유사도를 구하므로 제외, 유사도 값은 1)
        cnt = 1
        print(sim_reivew_idx)

        # print(f'현재 리뷰 : ', data.review[index])
        for i in sim_reivew_idx:
            # print(f'유사리뷰 {i} : ', data.review[i[0]])
            new_table.iloc[index]['similar_comment_' + str(cnt)] = data.review[i[0]]  # 유사 리뷰 저장
            new_table.iloc[index]['similarity_' + str(cnt)] = i[1]  # 유사도 저장
            cnt += 1

    return new_table


def visualize(data):
    okt = Okt()
    data['Noun'] = data['temp'].apply(okt.nouns)  # only 명사

    noun_text = [take2 for take1 in data['Noun'] for take2 in take1]
    text = nltk.Text(noun_text, name='NMSC')
    wordInfo = dict()
    for tags, counts in text.vocab().most_common(10):
        if (len(str(tags)) > 1):
            wordInfo[tags] = counts

    bar_save(wordInfo)
    pie_save(data)

    for tags, counts in text.vocab().most_common(100):
        if len(str(tags)) > 1:
            wordInfo[tags] = counts

    word_cloud_save(wordInfo)


def bar_save(wordInfo):
    font = fm.FontProperties(fname='210Black.ttf')
    fig, ax = plt.subplots(figsize=(15, 8), facecolor="#FFFFFF")
    ax.patch.set_facecolor('#FFFFFF')
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for position in ['bottom', 'top', 'left', 'right']:
        ax.spines[position].set_color('#FFFFFF')
    fig.patch.set_alpha(0.2)

    plt.title('주요 키워드', font=font, fontsize=60, color="#66B9CC")

    Sorted_Dict_Values = sorted(wordInfo.values(), reverse=True)
    Sorted_Dict_Keys = sorted(wordInfo, key=wordInfo.get, reverse=True)
    plt.bar(range(len(wordInfo)), Sorted_Dict_Values, align='center', color='#F8DF6F')
    plt.xticks(range(len(wordInfo)), list(Sorted_Dict_Keys), font=font, fontsize=30, color="#D96FA0")
    plt.yticks(font=font, fontsize=20, color="#D96FA0")
    plt.tick_params(labelsize=30)
    plt.savefig("./static/image/graph.png")


# word cloud
def word_cloud_save(wordInfo):
    sw = ['Name', 'Noun', 'object', 'dtype']
    mask = Image.open('cloud.png')
    mask = np.array(mask)
    fig, ax = plt.subplots(facecolor="#181818")
    ax.patch.set_facecolor('#181818')
    wordcloud = WordCloud(
        font_path='210Black.ttf',
        width=1000,
        height=600,
        background_color='white',
        colormap='Set2',
        random_state=0,
        stopwords=sw,
        mask=mask,
    ).generate_from_frequencies(wordInfo)

    plt.figure(figsize=(20, 10), facecolor='#FFFFFF')
    for position in ['bottom', 'top', 'left', 'right']:
        ax.spines[position].set_color('#FFFFFF')
    wordcloud.to_file('./static/image/wordcloud.png')


def pie_save(data):
    font = fm.FontProperties(fname='210Black.ttf')
    fig = plt.figure(figsize=(10, 10))
    star = sorted(list(data['star'].unique()))
    result = data.groupby(data['star']).size()
    print(result)
    colors = ['#9DA3E2', '#C2E3EB', '#CBDFB7', '#F5DC6E', '#F0C5D9', ]
    plt.pie(list(np.array(result.tolist())), explode=None, labels=star, colors=colors, startangle=30,
            textprops={'fontsize': 30, 'font': font, 'color': '#5C65CF'})
    centre_circle = plt.Circle((0, 0), 0.30, fc='white')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.savefig('./static/image/pie.png')


def sorting_slicing(data):
    data = data.sort_values(by=['len'])
    if len(data) > 10:
        data = data[10:11]
    if len(data) > 5:
        data = data[5:6]
    elif len(data) > 4:
        data = data[4:5]
    elif len(data) > 3:
        data = data[3:2]
    elif len(data) > 2:
        data = data[2:3]
    elif len(data) > 1:
        data = data[1:2]
    elif len(data) > 0:
        data = data[0:1]
    return data


def findRepresentingReview(data):
    for i in range(len(data)):
        data.loc[i, 'len'] = len(data.loc[i, 'comment'])

    data_1 = data[data.star == '1']
    data_1 = sorting_slicing(data_1)

    data_2 = data[data.star == '2']
    data_2 = sorting_slicing(data_2)

    data_3 = data[data.star == '3']
    data_3 = sorting_slicing(data_3)

    data_4 = data[data.star == '4']
    data_4 = sorting_slicing(data_4)

    data_5 = data[data.star == '5']
    data_5 = sorting_slicing(data_5)

    RepresentingReviewData = pd.concat([data_1, data_2, data_3, data_4, data_5])

    return RepresentingReviewData


# def filter(request):
#     evnet_list = eventModel.objects.all()
#     similar_list = SimilarComment.objects.all()
#     rest_info = Rest_Info.objects.all()
#
#     f = request.GET.getlist('f')
#
#     print(f)
#
#     similar_list = zip(list(similar_list.star), list(similar_list.comment), list(similar_list.similarComment1),
#                        list(similar_list.similarComment2))
#     event_list = zip(list(evnet_list.expectedPoint), list(evnet_list.review), list(evnet_list.label))
#
#     context = {'Rest_Info': rest_info, 'similar_list': similar_list, 'event_list': event_list, 'filter': f}
#     # ################################################################# 수정
#
#     return render(request, 'search_results.html', context)
