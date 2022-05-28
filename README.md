***

# 📑 기달리뷰 Project  

<img src="/git_img/main.PNG">

## **'기달리뷰' implies that we will analyze improve the review, so wait!**
### Watch the Demo Video here 💁‍♀️ 
### View the Document here 💁‍♂️ 
### Click here to use 기달리뷰💁 
***

## 🔍 Member information 
 
###### 👨‍💻 박윤재 zerglisk123@naver.com  
###### 👩‍💻 임채윤 lcu1027@gmail.com  
###### 👩‍💻 장지아 ghdwndi013@gmail.com

***

## 💻 Team meeting schedule 

###### 🟢 During the semester 1 of 2021 : Every Wednesday at 7pm, every Sunday at 9pm
###### 🟢 During the semester 2 of 2021: Every Monday at 6pm, every Friday at 5pm
###### 🟢 During the semester 1 of 2022: Every Wednesday at 4pm

***


## 🗓️ Progress plan


## 📝 2021 Semester 1

### 💡 Week 3-4
**🔸 Prior Studies** <br>
Comparison and analysis of delivery apps	(배달의민족, 요기요, 쿠팡이츠, 배달통, etc) <br>
Analysis Reviews

**🔸 Review numerous reviews, select review criteria, reconfirm reviews based on criteria**
- Low star rating without explanation
- Low star rating with positive reviews
- Low star rating for unfounded reasons
- Store slander, insult review
- Low star rating for food from other stores
- Use a review agency

**🔸 Related papers, Model & Library**<br>
Research papers related to NLP(Natural Language Processing), emotional analysis, and review data <br>
Selenium, BeautifulSoup, PyKoSpacing, KoNLPy, Tesseract-OCR, KorBERT, etc

<br>

### 💡 Week 5-7
**Inspect numerous reviews and select review criteria**
- Yogiyo has 4 star rating criteria: taste, quantity, delivery, and recommended
- Thinking about points to become a special study, not just a review analysis

<br>

### 💡 Week 9-11
Crawling small data for the Yogiyo website. Use Beautiful Soup 4.

### **Crawling Target Data** <br>
#### **[About the store]** <br>
Name, total star rating (whether recommended, taste, quantity, delivery), review event notice (CEO notification) <br>
#### **[Review]** <br>
Star ratings for each review (recommended, taste, quantity, delivery), review content, review creation date, order menu <br>

<br>

### 💡 Week 12
**Spacing preprocessing**
- Use PyKoSpacing to preprocess spaces
- Converting non-spaced Korean sentences into spacing sentences

**Spelling preprocessing**
1.  Pre-processing the entire sentence
- Analyze the grammar of sentences using KcBERT
- Correct grammar and typos throughout the sentence if there are errors in grammar
2.  Calculate the frequency of keyword appearance and preprocess if it is less than a certain frequency
- Use KoNLPy to divide sentences by morpheme and calculate frequency by keyword
- Divide into upper and lower levels according to frequency and preprocess for lower keywords

<br>

### 💡 Week 13-15
**🔸Implement for event participation review classification** <br>
1. Classify as text <br>
Categorize sentences by morpheme using KoNLPy <br>
Explore if keywords that match the review event item exist <br>
Categorize reviews that are suspected of being eventful, such as services, events, etc <br>
 
2. Categorize as an image
Use Tesseract to perform OCR processing that replaces letters in images with text <br>
Analyze replaced text to classify eventuality reviews <br>

**🔸Re-evaluate event review star rating**
1. High star rating, negative reviews
Comparison of review sentiment analysis results using KorBERT <br>
Identify the negative and positive characteristics of the categorized words and reconstruct the star rating <br>

2. High star rating, positive reviews
Comparison of review sentiment analysis results using KorBERT <br>
Review Reconfiguration Verification Procedure <br>

<br>

### 💡 Summer vacation
We investigated the database construction and implemented a crawling automation system.

<br>
<br>

## 📝 2021 Semester 2

### 💡Week 1-2
**Research the databases and servers you want to use and plan for a semester** <br>
**Database candidates: Mongo DB, SQLite** <br>
**Discuss how to build a server. The plan is as follows** <br>
1. Build your own server
2. Rent a server

<br>

### 💡Week 3-4
**Data collection and classification criteria selection**
- Collect data by categorizing them into categories that Yogiyo side categorizes
- 620-2 Gachon University, Bokjeong-dong, Sujeong-gu, Seongnam-si, Gyeonggi-do
- Selected as a restaurant for review events

**Categorization of review data by category**
- Categorize reviews into 8 categories
- Create folders, files for each category
- Repeat crawl by category type
- Separate data frames are divided into several tables and created as a single DB file

<br>

### 💡Week 5-6
**Building a Server with Raspberry Pi**
- The goal is to automate crawling on the server itself.
- Specify the crawling folder as the ftp server folder to allow users to receive files through the ftp server for easier retrieval of crawled data from the server
- Enable concurrent operations with RealVNC for smooth operation

<br>

### 💡Week 9-10
Improvement and maintenance of crawling automation system after studying web crawling and data preprocessing techniques

<br>

### 💡Week 11
Finalize data crawling on Yogiyo website and end data preprocessing. <br>
Project ideas such as participation in review events and classification of non-participation reviews are starting to be realized in earnest.

<br>

### 💡Week 12
After determining the detailed function of the review analyzer, each member is responsible for implementing it. <br>
1.  Review Event Prediction Model - KoELECTRA (장지아)
2.  Predict Star ratings - TF-IDF (박윤재)
3.  Similar Review - Word2Vec, Cosine Similarity (임채윤)

<br>

### 💡Week 13
🔸Evaluate performance after model training. <br>
🔸Implement review event participation sorting and filtering. <br>
🔸Visualize functions according to their respective roles. <br>

<br>

### 💡Week 14-15
The results of the review analysis through the model will be displayed on a web page using the long-range framework.<br>
After implementing a web page through HTML and css languages, connect it to the model.

<br>
<br>

## 📝 2022 Semester 1
##### We did documentation work to distribute to users, make marketing plans, and continue to supplement the system through feedback after actual distribution.

##### [Maintenance]
🔸Reviews are constantly updated, so we implemented adding words to dictionaries on a regular basis and automating learning.
🔸Add Top menu bar and Paging function.
- Top menu bar that does not move even when scrolling and is always fixed at the top of the screen.
- Paging function to the AI Review section so that we could see 5 reviews on each page.



