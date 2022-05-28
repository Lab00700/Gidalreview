
import re
from selenium import webdriver
import time
from bs4 import BeautifulSoup
import datetime
import os.path
import pandas as pd
import sqlite3

NoneType = type(None)
f = False
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

if not os.path.exists('Crowling-Data'):  # 데이터 폴더 없으면 폴더 생성
    os.makedirs('Crowling-Data')
if not os.path.exists('Crowling-URL'):  # 주소 폴더 없으면 폴더 생성
    os.makedirs('Crowling-URL')
for i in category:
    if not os.path.exists('Crowling-Data/' + i):  # 데이터 파일 없으면 파일 생성
        os.makedirs('Crowling-Data/' + i)
    if not os.path.isfile("Crowling-URL/" + i + "_url.txt"):  # 크롤링 주소 파일 없으면 파일 생성
        f = open("Crowling-URL/" + i + "_url.txt", 'w')
        f.close()

def Read_url(u):  # 크롤링 주소 읽기
    f = open("Crowling-URL/" + u + "_url.txt", 'r')
    url = f.readline()
    if url=='': # 입력된게 없으면 None 리턴
        f.close()
        return "None"
    url_array = []
    print("*********** Url ***********")
    while url: # 한줄씩 받아서 배열로 바꾼다.
        url_array.append(url)
        url = f.readline()
        url=url.replace("\n","")
        print(url)
    f.close()
    return url_array

def Auto_crowling(data='None',data_non_del='None',data_non_pic='None',data_non_pic_del='None',
                data_total_5="None",data_total_3_4="None",data_total_1_2="None",
                data_taste_5="None",data_taste_3_4="None",data_taste_1_2="None",
                data_quantity_5="None",data_quantity_3_4="None",data_quantity_1_2="None",
                data_delivery_5="None",data_delivery_3_4="None",data_delivery_1_2="None"): 

    i = 0
    k = 0
    time.sleep(0.3)

    er_t=0
    error=True
    while error:
        try:
            error=False
            rev = driver.find_element_by_xpath("//*[@id='content']/div[2]/div[1]/ul/li[2]/a")
            rev.click()
        except:
            print("클린리뷰 Click Error. Try again...")
            error=True
            time.sleep(5)
            er_t=er_t+1
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

    review_data = pd.DataFrame(columns=['total', 'taste', 'quantity', 'delivery', 'com_name', 'com_time', 'order_menu', 'comment', 'com_picture1', 'com_picture2', 'com_picture3'])
    while i < page:
        total = ''
        taste = ''
        quantity = ''
        delivery = ''
        com_name = ''
        com_time = ''
        order_menu = ''
        comment = ''
        com_picture = ["None","None","None"]

        if i != 0:
            try: # 더보기 버튼이 존재할 때 리뷰 불러오기
                driver.find_element_by_xpath("//*[@id='review']/li[" + str(i) + "2]")
                rev = driver.find_element_by_xpath("//*[@id='review']/li[" + str(i) + "2]")
                time.sleep(0.4)
                rev.click()
            except: # 리뷰 전체를 불러왔기에 더보기 버튼이 없을 때
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

        reviews = soup.find_all("li", attrs={"class": "list-group-item star-point ng-scope"})

        current_time = datetime.datetime.now() #현재 시간
        for review in reviews[k:]:

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
            #print(t_time.year)
            if len(data) != 0:  # 기존 크롤링된 파일 존재여부 확인
                t_time = datetime.datetime.strptime(str(data.tail(1)['com_time'].values), "[\'%Y년 %m월 %d일\']")
                temp_time = str(t_time.year) + "년 " + str(t_time.month) + "월 " + str(t_time.day) + "일"
                if com_time <= temp_time:  # 기록된 가장 최근 날짜까지 크롤링 Ex)17일까지 크롤링됬으면, 17일부터 크롤링 진행(시간을 알수 없기때문에 기록된 날짜까지 크롤링)
                    run = False  # 크롤링 중지
                    break;
            if "2017" in str(temp_time):  # 2017년 리뷰들은 총평점만 존재
                run = False
                break;

            total = 0
            for cnt in range(5):
                 if str(review.select("div")[2].select("span")[cnt+1])=="<span class=\"full ng-scope\" ng-repeat=\"i in review.rating|number_to_array track by $index\">★</span>":
                     total += 1
            total = str(total)
            taste = review.find("span",
                            attrs={"class": "points ng-binding", "ng-show": "review.rating_taste > 0"}).get_text() #맛
            quantity = review.find("span", attrs={"class": "points ng-binding",
                                                       "ng-show": "review.rating_quantity > 0"}).get_text() #양
            try:
                delivery = review.find("span", attrs={"class": "points ng-binding",
                                                      "ng-show": "review.rating_delivery > 0"}).get_text()  # 배달
            except AttributeError as e:
                delivery = ''

            com_name = review.find("span", attrs={"class": "review-id ng-binding", "ng-show": "review.phone"}).get_text() #이름

            order_menu = review.find("div", attrs={"class": "order-items default ng-binding"}).get_text() #주문메뉴
            comment = review.find("p", attrs={"ng-show": "review.comment"}).get_text() #리뷰 코멘트
            if review.find("table", attrs={"class": "info-images ng-scope"}) != None: #리뷰 사진 1개
                com_picture[0] = review.find("img")["src"]
            elif review.find("table", attrs={"class": "info-images half ng-scope"}) != None: #리뷰 사진 2개
                com_picture[0] = review.find("img", attrs={"data-index": "0"})["src"]
                com_picture[1] = review.find("img", attrs={"data-index": "1"})["src"]
            elif review.find("table", attrs={"class": "info-images three ng-scope"}) != None: #리뷰 사진 3개
                com_picture[0] = review.find("img", attrs={"data-index": "0"})["src"]
                com_picture[1] = review.find("img", attrs={"data-index": "1"})["src"]
                com_picture[2] = review.find("img", attrs={"data-index": "1"})["src"]

            review_data.loc[k] = [total, taste, quantity, delivery, com_name, com_time, order_menu, comment, com_picture[0], com_picture[1], com_picture[2]] #크롤링된 데이터 1행씩 이어쓰기
            com_picture = ["None", "None", "None"]
            k = k + 1

        if run == False: #크롤링 중지
            break

    review_data = review_data[::-1].reset_index(drop=True, inplace=False) #역순정렬 -> 갱신된 리뷰 데이터들 이어붙이기 편하게

    # 상위100, 하위 100 총 200개의 리뷰 추출
    # if len(review_data.index) > 20000:
    #     review_data = review_data.sort_values(by='total') #평점순 오름차순 정렬
    #     review_data = pd.concat([review_data[0:100], review_data[len(review_data.index)-100:len(review_data.index)]]) #오름차순 100개, 내림차순 100개 추출하여 저장

    review_data_non_del = review_data[review_data['delivery'] == ""]
    review_data_non_del = review_data_non_del.drop(['delivery'], axis=1)
    review_data_non_del = review_data_non_del.loc[review_data_non_del['com_picture1'] != "None"]

    review_data_non_pic = review_data[review_data['com_picture1'] == "None"]
    review_data_non_pic = review_data_non_pic.drop(['com_picture1', 'com_picture2', 'com_picture3'], axis=1)

    review_data_non_pic_non_del = review_data_non_pic[review_data_non_pic['delivery'] == ""]
    review_data_non_pic_non_del = review_data_non_pic_non_del.drop(['delivery'], axis=1)

    review_data_non_pic = review_data_non_pic.loc[review_data_non_pic['delivery'] != ""]

    review_data = review_data.loc[review_data['delivery'] != ""]
    review_data = review_data.loc[review_data['com_picture1']!="None"]

    tot_data = pd.concat([data, review_data], axis=0)
    tot_data = tot_data.drop_duplicates(['com_name', 'com_time', 'order_menu', 'comment'], keep='first')  # 중복데이터 처리

    tot_data_non_del = pd.concat([data_non_del, review_data_non_del], axis=0)
    tot_data_non_del = tot_data_non_del.drop_duplicates(['com_name', 'com_time', 'order_menu', 'comment'], keep='first')  # 중복데이터 처리

    tot_data_non_pic = pd.concat([data_non_pic, review_data_non_pic], axis=0) #이미지여부 추가
    tot_data_non_pic = tot_data_non_pic.drop_duplicates(['com_name', 'com_time', 'order_menu', 'comment'], keep='first') #이미지여부 추가

    tot_data_non_pic_non_del = pd.concat([data_non_pic_del, review_data_non_pic_non_del], axis=0)  # 이미지여부 추가
    tot_data_non_pic_non_del = tot_data_non_pic_non_del.drop_duplicates(['com_name', 'com_time', 'order_menu', 'comment'],keep='first')  # 이미지여부 추가
    
    # 총 별점 분류
    tot_data_total_5 = tot_data[tot_data['total']=="5"]
    tot_data_total_3_4 = tot_data[(tot_data['total']=="3") | (tot_data['total']=="4")]
    tot_data_total_1_2 = tot_data[(tot_data['total']=="1") | (tot_data['total']=="2")]

    # 맛 별점 분류
    tot_data_taste_5 = tot_data[tot_data['taste']=="5"]
    tot_data_taste_3_4 = tot_data[(tot_data['taste']=="3") | (tot_data['taste']=="4")]
    tot_data_taste_1_2 = tot_data[(tot_data['taste']=="1") | (tot_data['taste']=="2")]

    # 양 별점 분류
    tot_data_quantity_5 = tot_data[tot_data['quantity']=="5"]
    tot_data_quantity_3_4 = tot_data[(tot_data['quantity']=="3") | (tot_data['quantity']=="4")]
    tot_data_quantity_1_2 = tot_data[(tot_data['quantity']=="1") | (tot_data['quantity']=="2")]

    # 배달 별점 분류
    tot_data_delivery_5 = tot_data[tot_data['delivery']=="5"]
    tot_data_delivery_3_4 = tot_data[(tot_data['delivery']=="3") | (tot_data['delivery']=="4")]
    tot_data_delivery_1_2 = tot_data[(tot_data['delivery']=="1") | (tot_data['delivery']=="2")]

    return tot_data,tot_data_non_del,tot_data_non_pic,tot_data_non_pic_non_del,tot_data_total_5,tot_data_total_3_4,tot_data_total_1_2,tot_data_taste_5,tot_data_taste_3_4,tot_data_taste_1_2,tot_data_quantity_5,tot_data_quantity_3_4,tot_data_quantity_1_2,tot_data_delivery_5,tot_data_delivery_3_4,tot_data_delivery_1_2 #이미지 + 별점분류

       
def rest_info_crowling():
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

    rest_name = soup.find("span", attrs={"class": "restaurant-name ng-binding"}).get_text()
    review_notice = soup.find("div", attrs={"class": "info-text ng-binding"}).get_text()

    if soup.find("table", attrs={"ng-if": "restaurant_info.introduction_by_owner.images.length == 1"}) != None:
        t_soup = soup.find("table", attrs={"ng-if": "restaurant_info.introduction_by_owner.images.length == 1"})
        r_notice_pic = t_soup.find("img")["src"]
    else:
        r_notice_pic = "None"
    if f == True: # 프랜차이즈 여부 확인
        franchise = "franchise"
    else:
        franchise = "not franchise"

    rest_info = pd.DataFrame(columns=['franchise','rest_name', 'review_notice', 'r_notice_pic'])
    rest_info.loc[0] = [franchise, rest_name, review_notice, r_notice_pic]

    return rest_info

page=1000 #page당 10개의 리뷰데이터가 존재하기에 리뷰 탐색 갯수는 page*10이다.

driver = webdriver.Chrome(r"C:\Users\82102\Desktop\chromedriver.exe")
# driver = webdriver.Chrome()
time.sleep(0.2)
while True:
    for u in category:
        if "franchise" in u:
            f = True
        else:
            f = False
        url = Read_url(u)
        i = 0

        if url == "None":  # Url이 없을 때
            er_t = 0
           # while url == "None" and er_t < 5:  # Url이 입력될 때까지 반복
            while url == "None" and er_t < 1:  # Url이 입력될 때까지 반복
                print(str(u) + " Url is not exist")
                time.sleep(10)  # 10초마다 Url이 입력되었는지 확인
                url = Read_url(u)
                er_t = er_t + 1
            #if er_t == 5:
            if er_t == 1:
                continue
        while i < len(url):
            time.sleep(0.2)

            er_t = 0
            error = True
            while error:
                try:
                    error = False
                    driver.get(str(url[i]))  # 가게 주소 불러오기
                except:
                    print("Restaurant Url load Error. Try again...")
                    error = True
                    time.sleep(10)
                    # driver = webdriver.Chrome("C:\chromedriver.exe")
                    driver = webdriver.Chrome()
                    time.sleep(0.2)
                    er_t = er_t + 1
                    if er_t == 5:
                        break
            if er_t == 5:
                print("***Occurred Error and Pass***")
                i = i + 1
                continue

            # driver.maximize_window()
            time.sleep(0.4)

            rest_info = rest_info_crowling()

            if str(rest_info) == "error":
                print("***Occurred Error and Pass***")
                i = i + 1
                continue

            if os.path.isfile(
                    'Crowling-Data/' + u + "/" + str(rest_info['rest_name'].loc[0]) + '.db'):  # 크롤링 파일 존재여부 확인
                exist_file = True
                print(str(rest_info['rest_name'].loc[0]) + " Crowling Data exists.")
            else:
                exist_file = False
                print(str(rest_info['rest_name'].loc[0]) + " Crowling Data does not exist.")

            df = sqlite3.connect('Crowling-Data/' + u + "/" + str(rest_info['rest_name'].loc[0]) + '.db')  # 있으면 파일 불러오기
            df_cur = df.cursor()
            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table("
                           "total integer, "
                           "taste integer, "
                           "quantity integer, "
                           "delivery integer, "
                           "com_name text, "
                           "com_time text, "
                           "order_menu text, "
                           "comment text, "
                           "com_picture1 text, "
                           "com_picture2 text, "
                           "com_picture3 text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_Non_delivery("
                           "total integer, "
                           "taste integer, "
                           "quantity integer, "
                           "com_name text, "
                           "com_time text, "
                           "order_menu text, "
                           "comment text, "
                           "com_picture1 text, "
                           "com_picture2 text, "
                           "com_picture3 text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_Non_Picture("
                           "total integer, "
                           "taste integer, "
                           "quantity integer, "
                           "delivery integer, "
                           "com_name text, "
                           "com_time text, "
                           "order_menu text, "
                           "comment text)")  # 이미지여부 추가

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_Non_Picture_Non_delivery("
                           "total integer, "
                           "taste integer, "
                           "quantity integer, "
                           "com_name text, "
                           "com_time text, "
                           "order_menu text, "
                           "comment text)")  # 이미지여부 추가

            df_cur.execute("CREATE TABLE IF NOT EXISTS Rest_Info("
                           "franchise, "
                           "rest_name text, "
                           "review_notice text, "
                           "r_notice_pic text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_total_5("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_total_3_4("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_total_1_2("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")    

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_taste_5("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_taste_3_4("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_taste_1_2("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")  

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_quantity_5("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_quantity_3_4("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_quantity_1_2("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")  

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_delivery_5("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_delivery_3_4("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")

            df_cur.execute("CREATE TABLE IF NOT EXISTS Crowling_Table_delivery_1_2("
                            "total integer, "
                            "taste integer, "
                            "quantity integer, "
                            "delivery integer, "
                            "com_name text, "
                            "com_time text, "
                            "order_menu text, "
                            "comment text, "
                            "com_picture1 text, "
                            "com_picture2 text, "
                            "com_picture3 text)")  

            print(str(rest_info['rest_name'].loc[0]) + " Crowling is running.")

            sql_to_df = pd.read_sql("SELECT * FROM Crowling_Table", df, index_col=None)
            sql_to_df_non_delivery = pd.read_sql("SELECT * FROM Crowling_Table_Non_delivery", df, index_col=None)  # 이미지여부 추가
            sql_to_df_non_pic = pd.read_sql("SELECT * FROM Crowling_Table_Non_Picture", df, index_col=None)  # 이미지여부 추가
            sql_to_df_non_pic_non_delivery = pd.read_sql("SELECT * FROM Crowling_Table_Non_Picture_Non_delivery", df, index_col=None)  # 이미지여부 추가
                       
            #총 별점 분류
            sql_to_df_total_5 = pd.read_sql("SELECT * FROM Crowling_Table_total_5", df, index_col=None);
            sql_to_df_total_3_4 = pd.read_sql("SELECT * FROM Crowling_Table_total_3_4", df, index_col=None);
            sql_to_df_total_1_2 = pd.read_sql("SELECT * FROM Crowling_Table_total_1_2", df, index_col=None);

            #맛 별점 분류
            sql_to_df_taste_5 = pd.read_sql("SELECT * FROM Crowling_Table_taste_5", df, index_col=None);
            sql_to_df_taste_3_4 = pd.read_sql("SELECT * FROM Crowling_Table_taste_3_4", df, index_col=None);
            sql_to_df_taste_1_2 = pd.read_sql("SELECT * FROM Crowling_Table_taste_1_2", df, index_col=None);

            #양 별점 분류
            sql_to_df_quantity_5 = pd.read_sql("SELECT * FROM Crowling_Table_quantity_5", df, index_col=None);
            sql_to_df_quantity_3_4 = pd.read_sql("SELECT * FROM Crowling_Table_quantity_3_4", df, index_col=None);
            sql_to_df_quantity_1_2 = pd.read_sql("SELECT * FROM Crowling_Table_quantity_1_2", df, index_col=None);

            #배달 별점 분류
            sql_to_df_delivery_5 = pd.read_sql("SELECT * FROM Crowling_Table_delivery_5", df, index_col=None);
            sql_to_df_delivery_3_4 = pd.read_sql("SELECT * FROM Crowling_Table_delivery_3_4", df, index_col=None);
            sql_to_df_delivery_1_2 = pd.read_sql("SELECT * FROM Crowling_Table_delivery_1_2", df, index_col=None);

            tot_data, tot_data_non_del, tot_data_non_pic, tot_data_non_pic_non_del, tot_data_total_5, tot_data_total_3_4, tot_data_total_1_2, tot_data_taste_5, tot_data_taste_3_4, tot_data_taste_1_2, tot_data_quantity_5, tot_data_quantity_3_4, tot_data_quantity_1_2, tot_data_delivery_5, tot_data_delivery_3_4, tot_data_delivery_1_2  = Auto_crowling(sql_to_df, 
            sql_to_df_non_delivery, sql_to_df_non_pic, sql_to_df_non_pic_non_delivery, #이미지여부 추가
            sql_to_df_total_5, sql_to_df_total_3_4, sql_to_df_total_1_2,  #총 별점 분류
            sql_to_df_taste_5, sql_to_df_taste_3_4, sql_to_df_taste_1_2, #맛 별점 분류
            sql_to_df_quantity_5, sql_to_df_quantity_3_4, sql_to_df_quantity_1_2, #양 별점 분류
            sql_to_df_delivery_5, sql_to_df_delivery_3_4, sql_to_df_delivery_1_2) #배달 별점 분류

            rest_info = rest_info_crowling()

            if str(tot_data) == "error":
                print("***Occurred Error and Pass***")
                i = i + 1
                continue
            elif str(tot_data_non_pic) == "error":  # 이미지여부 추가
                print("***Occurred Error and Pass***")  # 이미지여부 추가
                i = i + 1  # 이미지여부 추가
                continue  # 이미지여부 추가
            er_t = 0
            error = True
            while error:
                try:
                    error = False
                    tot_data.to_sql('Crowling_Table', df, if_exists='replace', index=False, chunksize=10000)
                    tot_data_non_del.to_sql('Crowling_Table_Non_delivery', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 이미지여부 추가
                    tot_data_non_pic.to_sql('Crowling_Table_Non_Picture', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 이미지여부 추가
                    tot_data_non_pic_non_del.to_sql('Crowling_Table_Non_Picture_Non_delivery', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 이미지여부 추가

                    tot_data_total_5.to_sql('Crowling_Table_total_5', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 총 별점 분류 5
                    tot_data_total_3_4.to_sql('Crowling_Table_total_3_4', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 총 별점 분류 3-4
                    tot_data_total_1_2.to_sql('Crowling_Table_total_1_2', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 총 별점 분류 1-2

                    tot_data_taste_5.to_sql('Crowling_Table_taste_5', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 맛 별점 분류 5
                    tot_data_taste_3_4.to_sql('Crowling_Table_taste_3_4', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 맛 별점 분류 3-4
                    tot_data_taste_1_2.to_sql('Crowling_Table_taste_1_2', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 맛 별점 분류 1-2

                    tot_data_quantity_5.to_sql('Crowling_Table_quantity_5', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 양 별점 분류 5
                    tot_data_quantity_3_4.to_sql('Crowling_Table_quantity_3_4', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 양 별점 분류 3-4
                    tot_data_quantity_1_2.to_sql('Crowling_Table_quantity_1_2', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 양 별점 분류 1-2

                    tot_data_delivery_5.to_sql('Crowling_Table_delivery_5', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 배달 별점 분류 5
                    tot_data_delivery_3_4.to_sql('Crowling_Table_delivery_3_4', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 배달 별점 분류 3-4
                    tot_data_delivery_1_2.to_sql('Crowling_Table_delivery_1_2', df, if_exists='replace', index=False,
                                            chunksize=10000)  # 배달 별점 분류 1-2
                    
                    rest_info.to_sql('Rest_Info', df, if_exists='replace', index=False, chunksize=10000)

                except:
                    print("Crowling data Export Error. Try again...")
                    error = True
                    time.sleep(10)
                    er_t = er_t + 1
                    if er_t == 5:
                        break
            if er_t == 5:
                print("***Occurred Error and Pass***")
                i = i + 1
                df.close()
                continue
            df.close()
            print(str(rest_info['rest_name'].loc[0]) + " Crowling done.")

            i = i + 1

    driver.refresh()  # 페이지 새로고침
    print("------Waiting next Crowling.------")
    time.sleep(60) # 시, 분이 같을 경우 계속 반복되는 것을 방지(크롤링 완료 이후 60초 지연)
    while True:
        time.sleep(10) # 10초마다 시간 확인
        current_time = datetime.datetime.now()
        if str(current_time.hour) == '17':
            if str(current_time.minute) == '32':
                break