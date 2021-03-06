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
        # ????????? ?????? ?????????
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
        # ################################################################# ??????

        return render(request, 'search_results.html', context)

    if request.method == 'POST':
        url_src = request.POST.get("url_src", '')
        if 'yogiyo' in url_src:
            # ????????? ??????, result.html??? ?????????
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

            # ?????? ?????? ?????? ??????
            pred = predict.predict(review_model.review,review_model.star)
            pred = pd.DataFrame(pred,columns=["predict"])
            rest_info = rest_info_crowling(url_src, pred['predict'])
            print(rest_info)
            print(review_model.review)
            print(review_model.label)
            print('**********************pred_predict**********************')
            print(pred['predict'])
            eventModel.expectedPoint = pred['predict']
            # review_model.review = pd.Series((v[0] for v in eventReview(example)))
            # print(review_model.review)

            # ?????? ??????
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

            # ?????? ??????
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
            f = ['?????? ????????? ?????? ????????? ??????', '?????? ????????? ?????? ????????? ??????']
            # context = {'review_list': review_model.review, 'star_list': review_model.star, 'pred_review_list':review_model.pred_review, 'expectedPoint_list':review_model.expectedPoint, 'label_list':review_model.label,
            # context = {'review_list': review_model.review, 'star_list': review_model.star, 'label_list':review_model.label, 'expectedPoint_list':review_model.expectedPoint,
            #             'Rest_Info':Rest_Info_model,
            #             'representing_comment_list':similar_model.comment, 'representing_comment_star_list':similar_model.star, 'similar_list_1':similar_model.similarComment1, 'similar_list_2':similar_model.similarComment2,'schedule_list':schedule_list}
            context = {'Rest_Info': Rest_Info_model, 'similar_list': similar_list, 'event_list': event_list,
                       'filter': f}
            # ################################################################# ??????
            pred = predict.predict(review_model.review, review_model.star,Rest_Info_model.rest_name)
            return render(request, 'search_results.html', context)
        elif 'yogiyo' not in url_src:
            return render(request, 'home.html',
                          {'error01': '????????? URL??? ??????????????????!'})
        else:
            return render(request, 'home.html',
                          {'error01': 'URL??? ??????????????????!'})


def rest_info_crowling(url_src, pred):
    options = webdriver.ChromeOptions()
    # ??? ????????? ?????? ??????
    options.add_argument("headless")
    # driver ??????
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
            print("?????? Click Error. Try again...")
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
        star = '???'
    elif star == 2:
        star = '??????'
    elif star == 3:
        star = '?????????'
    elif star == 4:
        star = '????????????'
    elif star == 5:
        star = '???????????????'

    expected_star_point = round(pred.mean(), 2)  # ????????? ??????????????? ?????? ?????????
    expected_star = int(expected_star_point)

    if expected_star == 0:
        expected_star = ''
    elif expected_star == 1:
        expected_star = '???'
    elif expected_star == 2:
        expected_star = '??????'
    elif expected_star == 3:
        expected_star = '?????????'
    elif expected_star == 4:
        expected_star = '????????????'
    elif expected_star == 5:
        expected_star = '???????????????'

    rest_name = soup.find("span", attrs={"class": "restaurant-name ng-binding"}).get_text()
    review_notice = soup.find("div", attrs={"class": "info-text ng-binding"}).get_text()

    rest_info = pd.DataFrame(
        columns=['star_point', 'star', 'expected_star_point', 'expected_star', 'rest_name', 'review_notice'])
    rest_info.loc[0] = [star_point, star, expected_star_point, expected_star, rest_name, review_notice]
    print(rest_info)

    return rest_info


def lets_do_crawling(url_src):
    options = webdriver.ChromeOptions()
    # ??? ????????? ?????? ??????
    options.add_argument("headless")
    # driver ??????
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
            print("???????????? Click Error. Try again...")
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
            try:  # ????????? ????????? ????????? ??? ?????? ????????????
                driver.find_element_by_xpath("//*[@id='review']/li[" + str(i) + "2]")
                rev = driver.find_element_by_xpath("//*[@id='review']/li[" + str(i) + "2]")
                time.sleep(0.4)
                rev.click()
            except:  # ?????? ????????? ??????????????? ????????? ????????? ?????? ???
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

        if run == False:  # ????????? ??????
            break

    time.sleep(0.2)
    reviews = soup.find_all("li", attrs={"class": "list-group-item star-point ng-scope"})

    current_time = datetime.datetime.now()  # ?????? ??????
    for review in reviews:
        try:
            temp_time = review.find("span",
                                    attrs={"class": "review-time ng-binding",
                                           "ng-bind": "review.time|since"}).get_text()  # ??????
            if temp_time == "????????? ???":
                t_time = current_time - datetime.timedelta(days=7)
                temp_time = str(t_time.year) + "??? " + str(t_time.month) + "??? " + str(t_time.day) + "???"
            elif temp_time == "??????":
                t_time = current_time - datetime.timedelta(days=1)
                temp_time = str(t_time.year) + "??? " + str(t_time.month) + "??? " + str(t_time.day) + "???"
            elif "??? ???" in temp_time:
                before = re.findall("\d+", temp_time)
                t_time = current_time - datetime.timedelta(days=int(before[0]))
                temp_time = str(t_time.year) + "??? " + str(t_time.month) + "??? " + str(t_time.day) + "???"
            elif "?????? ???" in temp_time:
                before = re.findall("\d+", temp_time)
                t_time = current_time - datetime.timedelta(hours=int(before[0]))
                temp_time = str(t_time.year) + "??? " + str(t_time.month) + "??? " + str(t_time.day) + "???"
            elif "??? ???" in temp_time:
                before = re.findall("\d+", temp_time)
                t_time = current_time - datetime.timedelta(minutes=int(before[0]))
                temp_time = str(t_time.year) + "??? " + str(t_time.month) + "??? " + str(t_time.day) + "???"
            elif "?????? ??????" in temp_time:
                t_time = current_time
                temp_time = str(t_time.year) + "??? " + str(t_time.month) + "??? " + str(t_time.day) + "???"
            com_time = temp_time
            total = 0
            for cnt in range(5):
                if str(review.select("div")[2].select("span")[
                           cnt + 1]) == "<span class=\"full ng-scope\" ng-repeat=\"i in review.rating|number_to_array track by $index\">???</span>":
                    total += 1
            star = str(total)
            comment = review.find("p", attrs={"ng-show": "review.comment"}).get_text()  # ?????? ?????????
            review_data.loc[len(review_data)] = [k, comment, com_time, star]  # ???????????? ????????? 1?????? ????????????
            k = k + 1
        except:  # ?????? ?????????????????? ?????? ?????? ????????? ?????? ?????? ???????????? ???????????? pass(?????? ??????)
            print("Error review Pass : " + str(review) + " " + str(com_time))
            k = k + 1

    review_data = review_data[::-1].reset_index(drop=True, inplace=False)  # ???????????? -> ????????? ?????? ???????????? ??????????????? ?????????

    return review_data


def use_split_join(data):
    string = ' '.join(data.split())
    return string


def eventReview(review_data):
    # ?????????
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
            y_pred.append("?????? ????????? ?????? ????????? ??????")
        else:
            y_pred.append("?????? ????????? ?????? ????????? ??????")

    return y_pred


def make_token(data):
    # ????????? ?????? ??????
    token = Twitter()  # ??????
    adding_noun = ['?????????', '??????', '?????????', '??????', '??????', '?????????', '?????????', '??????', '?????????', '??????', '??????', '??????', '?????????', '??????', '??????', '??????',
                   '?????????', '???????????????', '?????????', '??????', '??????', '??????', 'jmt', 'jmtg', 'jmtgr', 'JMT', 'JMTG', 'JMTGR', '?????????', '??????',
                   '?????????', '??????',
                   '??????', '?????????', '??????', '??????', '????????????', '??????', '??????', '??????', '?????????', '?????????', '?????????', '??????', '?????????', '??????', '??????',
                   '??????',
                   '??????', '????????????', '?????????', '????????????', '???????????????', '??????', '??????', '?????????', '??????', '?????????', '?????????', '?????????', '?????????', '?????????'
                                                                                                              '?????????',
                   '???????????????', ' ???????????????', '?????????', '??????', '??????', '??????', '??????', '?????????', '??????']

    for i in adding_noun:
        token.add_dictionary(i, 'Noun')  # ?????? ??????

    adding_verb = ['??????', '?????????', '?????????', '?????????', '?????????', '?????????']

    for i in adding_verb:
        token.add_dictionary(i, 'Noun')  # ?????? ??????

    token.add_dictionary('???', 'Noun')  # ?????? ??????

    # ????????? ??????
    with open('stop.txt', 'rt', encoding='UTF8') as f:
        stopwords = f.read().replace('\n', ' ')
    stopwords = stopwords.split(' ')

    temp = (token.morphs(data, norm=True, stem=True))

    stopwords_removed_sentence = [word for word in temp if not word in stopwords]  # ????????? ??????

    return stopwords_removed_sentence


def make_doc_embed_list(model, test_tokenized_data):
    document_embedding_list = []  # ?????? ????????? ?????? ?????????
    # ????????? ???????????? ???????????? ???????????? ????????? ????????? ?????? ????????? ???????????? ??????

    # ??? ????????? ?????????
    for idx, review in enumerate(test_tokenized_data):
        doc2vec = None
        cnt = 0
        for word in review:
            if word in list(model.wv.index_to_key):
                cnt += 1
                # ?????? ????????? ?????? ?????? ???????????? ???????????? ?????????.
                if doc2vec is None:
                    doc2vec = model.wv.__getitem__(word)
                else:
                    doc2vec = doc2vec + model.wv.__getitem__(word)

        # ?????? ????????? ?????? ?????? ????????? ?????? ?????? ????????? ????????????.
        if doc2vec is not None:
            doc2vec = doc2vec / cnt
        else:
            doc2vec = np.zeros(100, )

        document_embedding_list.append(doc2vec)

    # ??? ????????? ?????? ?????? ?????? ???????????? ??????
    return document_embedding_list


def use_split_join(data):
    string = ' '.join(data.split())
    return string


def findSimilarReview(data):
    data['review'] = data['review'].str.join('').str.replace(r"\n", "")  # ????????? ??????
    data['temp'] = data['review'].copy()

    # ?????????
    data['temp'] = data['temp'].str.replace(pat=r'[^\w]', repl=r' ', regex=True)  # ??????????????? ????????????
    data['temp'] = data['temp'].apply(use_split_join)  # ????????? ????????? ????????? ??????

    # ????????? ??? ????????? ?????????(graph, wordcloud)
    visualize(data)

    # ?????????
    data['token'] = data.temp.map(lambda x: make_token(str(x)))

    tokenized_data = data.token.values.tolist()
    data.token = data.token.map(lambda x: ast.literal_eval(str(x)))
    print('????????????', data['review'])

    model = Word2Vec(sentences=tokenized_data, vector_size=100, window=2, min_count=5, workers=4, sg=1)
    # vector_size = ?????? ????????? ?????? ???. ???, ????????? ??? ????????? ??????.
    # window = ???????????? ????????? ??????
    # min_count = ?????? ?????? ?????? ??? ?????? (????????? ?????? ???????????? ???????????? ?????????.)
    # workers = ????????? ?????? ???????????? ???
    # sg = 0??? CBOW, 1??? Skip-gram.

    model.save('word2vec.model')

    document_embedding_list = make_doc_embed_list(model, tokenized_data)  # ?????? ????????? ?????? ?????????
    cosine_similarities = cosine_similarity(document_embedding_list, document_embedding_list)  # ????????? ????????? ?????????
    print(cosine_similarities)

    # ??? ???????????? ??????
    new_table = pd.DataFrame(
        columns=['comment', 'star', 'preprocessing_comment', 'similar_comment_1', 'similarity_1', 'similar_comment_2',
                 'similarity_2'])
    new_table['comment'] = data['review'].copy()
    new_table['star'] = data['star'].copy()
    new_table['preprocessing_comment'] = data['temp'].copy()

    # ?????? ????????? ?????? ??????
    for index in range(0, len(data)):
        sim_score = list(enumerate(cosine_similarities[index]))  # ?????? ????????? ?????? ????????? ????????? ?????????
        sim_score = sorted(sim_score, key=lambda x: x[1], reverse=True)  # ????????? ?????? ????????? ??????
        sim_reivew_idx = [i for i in sim_score[1:3]]  # ?????? ?????? 2??? ?????? (index 0??? ??????????????? ??????????????? ???????????? ???????????? ??????, ????????? ?????? 1)
        cnt = 1
        print(sim_reivew_idx)

        # print(f'?????? ?????? : ', data.review[index])
        for i in sim_reivew_idx:
            # print(f'???????????? {i} : ', data.review[i[0]])
            new_table['similar_comment_' + str(cnt)].iloc[index] = data.review[i[0]]  # ?????? ?????? ??????
            new_table['similarity_' + str(cnt)].iloc[index] = i[1]  # ????????? ??????
            print(new_table.iloc[index]['similar_comment_' + str(cnt)])
            print(new_table.iloc[index]['similarity_' + str(cnt)])
            cnt += 1
    return new_table


def visualize(data):
    okt = Okt()
    data['Noun'] = data['temp'].apply(okt.nouns)  # only ??????

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

    plt.title('?????? ?????????', font=font, fontsize=60, color="#66B9CC")

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
#     # ################################################################# ??????
#
#     return render(request, 'search_results.html', context)
