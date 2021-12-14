***

# 기달리뷰 Project  

<img src="/git_img/main.PNG">

**'기달리뷰' implies that we will improve the review, so wait!**


***

### Member information 
 
* 박윤재 zerglisk123@naver.com  
* 임채윤 lcu1027@gmail.com  
* 장지아 ghdwndi013@gmail.com

***

## Team meeting schedule 
* During the semester 1: Every Wednesday at 7pm, every Sunday at 9pm
* During the semester 2: Every Monday at 6pm, every Friday at 5pm


***


## Overview  

### This project is a graduation project from the Department of Software at Gachon University.  
  
As the untact culture increased in the aftermath of COVID-19, the delivery service naturally expanded significantly.  
The number of companies registered in the delivery app has increased, and the number of customers who receive food by ordering has also increased.
However, this is causing problems such as black consumers and review agencies.  
All delivery app users, including self-employed people, are confused.  
The opinion that the review system of delivery app platforms needs to be fundamentally improved is popping up everywhere.    

In a delivery app that has grown with the advantage of being able to trust and order after seeing 'reviews', the fact that reviews cause problems is a big blow to the growth of delivery apps.    

Therefore, we propose a review improvement system to solve the problems of companies and consumers who use delivery apps.


***

## Objective
Based on the ideas presented above, the goal of our project is to build a review improvement system that only provides credible reviews.


***

## Expected effect
Due to the system we will build in the future, both the psychological burden and the cost burden of review events and rating terrorism experienced by companies will be reduced,
Consumers expect to use delivery apps more quickly and accurately through credible reviews and ratings.


***

## System structure

<img src="/git_img/system_structure.PNG">

It's the overall structure of the "기달리뷰" system. 

The system consists of a crawling automation system, a web server, a database, a review analyzer, and a web page. 

The web server is a Linux-based web server built with Raspberry Pi, and we have built an online public development environment for developers using Real VNC.

FTP servers built on web servers help you analyze reviews and access data to provide users.

The user enters the URL of the company that wants to analyze the review through the web browser. 

The web page data of the URL is crawled by the crawling automation system, and the review is analyzed through a review analyzer, and the user can check it through the website produced.

***


## Key features

### Crawling automation system

The crawling automation system is a system created by us to collect large amounts of data, and collects information on restaurants, reviews, and stars on the Yogiyo website entered through the Chrome Driver, and stores them in the database according to predetermined classification criteria such as image status, franchise status, and star rating.

- yogiyo review crawling using selenium and beautifulSoup

- Total 50000 review about 20 population stores.

<br>
**Crawling Target Data**

[Store Information]
- Name
- Total Horoscope(Recommended, taste, quantity, delivery)
- Review Event Notice(CEO’s Notification)

[Review]
- Star Rating for each Review (taste,quantity, delivery)
- Review Content
- Order menu
- Date


***
### Review analyzer
The review analyzer brings restaurant information and reviews stored in the database to process natural language using a pre-produced word dictionary, and gives predictive ratings, possibility of participating in review events, and keyword analysis results through implemented models and algorithms.

<br>

* [Finding Similar reviews]

<img src="/git_img/similar_review.PNG">

The process of obtaining similar reviews is as follows.

Vectorize the tokenized word through the Word2Vec technique.

Calculate the average of word vectors through the Document embedding technique.

Find cosine similarity between reviews.

Sort in order of high similarity.

<br>


* [Star ratings prediction model]

<img src="/git_img/star_prediction.PNG">

We checked which words were detected in each review to derive the prediction star ratings of the review, and how often each word was used. 

Based on restaurants, based on categories, franchises were classified and extracted into DB files in consideration of a total of three cases based on categories, and 13,337 words could be identified in a total of 50,000 reviews.

The word dictionary was created using TF-IDF techniques.

<br>

* [Review Event Prediction Model]

<img src="/git_img/review_event_prediction.png">

By predicting the participation in the review event as high and low, a filtering function and alignment function were implemented that only the selected prediction results can be viewed using the prediction results.

Collect 1,600 pieces of data each that participated in and did not participate in the review event.

It collects data by directly determining whether to participate in the review event.

Train the KoELECTRA model with the collected data.

<br>


* [Review visualization]

<img src="/git_img/bar_graph.png" width="40%" height="40%"> 
<img src="/git_img/wordcloud.png" width="40%" height="40%">

Using the Matplotlib library provided by Python.

We implemented a bar graph representing key keywords through the frequency of keywords. 
In addition, the frequency of keywords was analyzed from the review data collected using the WordCloud library and visualized in cloud form. 

Review data only extract nouns using the Open-Korea-Text Library, a preprocessing morpheme analyzer. After that, the frequency of each keyword was measured, stored in the form of a dictionary, and visualized.


***

## Progress plan

* Winter vacation <br>
Before the start of the next semester, we will further review the collected data to improve the performance of deep learning models, and provide users with improved review analysis results by adding aesthetic sense and readability of the homepage through CSS, a style sheet language.

<br>

* 2022 semester 1 <br>
We will do documentation work to distribute to users, make marketing plans, and continue to supplement the system through feedback after actual distribution.

***
