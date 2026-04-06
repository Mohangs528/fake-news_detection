# Fake News Detection Using NLP and Machine Learning

## 📌 Project Overview
This project is a **Fake News Detection System** developed using **Natural Language Processing (NLP)** and **Machine Learning (ML)**.

The system takes a **news article or news text** as input from the user and predicts whether the news is **REAL** or **FAKE**.

This project is useful for understanding how machine learning can be applied to **text classification problems**.

---

## 🚀 Features
- Detects whether a news article is **REAL** or **FAKE**
- Uses **NLP techniques** for text preprocessing
- Uses **TF-IDF Vectorization** for feature extraction
- Uses **Machine Learning** for classification
- Simple **Flask web interface**
- Easy to use and beginner-friendly

---

## 🛠️ Technologies Used
- **Python**
- **Flask**
- **Scikit-learn**
- **Pandas**
- **NumPy**
- **HTML**
- **CSS**
- **NLP (Text Preprocessing)**

---

## ⚙️ How the Project Works
1. User enters or pastes a **news article** into the web page.
2. The system **cleans and preprocesses** the text.
3. The text is converted into numerical format using **TF-IDF Vectorizer**.
4. The trained **Machine Learning model** predicts whether the news is:
   - **REAL**
   - **FAKE**
5. The result is displayed on the screen.

### 🔄 Workflow
Input News → Text Cleaning → TF-IDF → ML Model → Prediction

---

## 📂 Project Structure
```text
fake-news-detection/
│
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── fake_or_real_news.csv
├── model.pkl
├── vectorizer.pkl
│
├── templates/
│   └── index.html
│
└── static/
    └── style.css
```text

---

📊 Dataset

The project uses a custom dataset containing:

News Text
Label (REAL or FAKE)
Example:
| Text                                     | Label |
| ---------------------------------------- | ----- |
| Government launches new education policy | REAL  |
| Aliens landed in India yesterday         | FAKE  |

---

🧠 Machine Learning Model

The model used in this project is:

Passive Aggressive Classifier
Why this model?
Suitable for text classification
Fast and efficient
Works well for fake news detection

---

🔤 NLP Techniques Used
Lowercasing text
Removing punctuation
Removing unnecessary symbols
Text cleaning

---

📥 Input and Output
Input
News article / headline / news content entered by user
Output
REAL
FAKE

---

▶️ How to Run the Project
1. Clone or Download the Project
git clone https://github.com/your-username/fake-news-detection.git
cd fake-news-detection
2. Install Required Libraries
pip install -r requirements.txt
3. Train the Model
python train_model.py
4. Run the Flask App
python app.py
5. Open in Browser
http://127.0.0.1:5000

---

📌 Example Prediction
Input:

Aliens found near Indian railway station

Output:

FAKE

---

⚠️ Limitations
Accuracy depends on dataset quality
Small dataset may reduce performance
This project does not use real-time news APIs
It is based only on the trained dataset

---

🔮 Future Enhancements
Use a larger dataset
Improve prediction accuracy
Add multilingual support
Use advanced models like BERT / LSTM
Improve UI design

---

🎯 Conclusion

This project demonstrates how NLP and Machine Learning can be used to classify news articles as REAL or FAKE.

It is a simple and useful project for learning text classification and fake news detection.

---

👨‍💻 Author / Team

Project Title: Fake News Detection Using NLP and Machine Learning

Developed By:

Mohan G S
Mohammad Zubair T
Swapna G
Geethanjali K

Guide Name:

Hanumanth Gowda

---

