# 📧 Spam Email Detection using Logistic Regression and Gmail API

A modular **Spam Email Detection System** built using:

- A pre-trained **Logistic Regression** model (trained on 6 Kaggle datasets)
- A **TF-IDF Vectorizer**
- The official **Gmail API** to fetch real-time emails from Gmail

---

## 📝 Project Description

This project demonstrates a complete end-to-end pipeline for Spam Email Detection:

- A Logistic Regression model was trained offline using **6 Kaggle datasets** from the [Phishing Email Dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset), combining the following corpora:
  - `SpamAssasin.csv`
  - `CEAS_08.csv`
  - `Enron.csv`
  - `Ling.csv`
  - `Nazario.csv`
  - `Nigerian_Fraud.csv`
- The model and TF-IDF vectorizer were saved as `.pkl` files for reusable inference.
- Emails are fetched from the user's Gmail account via the **Gmail API**.
- The **email subject and body are combined**, processed, and classified as **Spam** or **Not Spam**.

---

Then place them in the following folder:

```
model_training/
    SpamAssasin.csv
    CEAS_08.csv
    Enron.csv
    Ling.csv
    Nazario.csv
    Nigerian_Fraud.csv

```
After placing the CSV files, run the training script:

```bash
python model_training/logisticregression.py
```

This will generate:

-Spam_Detection_Model.pkl

-features.pkl


## 🚀 Features

✅ Model trained on **6 Kaggle datasets**  
✅ Real-time email fetching from Gmail inbox  
✅ Automatic classification of emails  
✅ Modular and reusable ML pipeline  
✅ Easy to extend and customize  

---

## 📂 Project Structure

```
.
SpamDetectionML/
├── model_training/                 
│   └── logisticregression.py        # Model training code (clean Python script)
│   ├── [user-provided datasets].csv # Place 6 Kaggle CSV datasets here
├── EmailFetchApi.py                 # Inference script using Gmail API + model
├── Spam_Detection_Model.pkl         # Pre-trained Logistic Regression model
├── features.pkl                     # Saved TF-IDF vectorizer
├── requirements.txt                 # Required Python packages
├── README.md                        # Project documentation
├── LICENSE                          # License file (MIT License suggested)
└── .gitignore                       # Ignore sensitive files and unwanted files

```

---

## 🛠️ Setup & Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/yourusername/SpamDetectionML.git
cd SpamDetectionML
```

### 2️⃣ Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3️⃣ Setup Gmail API Credentials

- Go to [Google Cloud Console](https://console.cloud.google.com/)
- Enable the **Gmail API** and create **OAuth2 credentials**
- Download `credentials.json` and place it in your project folder.
- ⚠️ `credentials.json` is included in `.gitignore` → do NOT push this file to GitHub!
- The first time you run the script, it will generate `token.json` (also ignored) after completing the OAuth flow.

---

## 🚀 Running the Project

```bash
python EmailFetchApi.py
```

---

## 📊 Results

- **Accuracy on testing data**: **98.61%**

- **Confusion Matrix**:

|                | Predicted Not Spam | Predicted Spam |
|----------------|-------------------|----------------|
| Actual Not Spam| 11737              | 216            |
| Actual Spam    | 126                | 12667          |

- **Precision** ≈ **98.32%**  
- **Recall** ≈ **99.01%**  
- **F1-score** ≈ **98.66%**

- The model demonstrates strong generalization and balance between **catching spam** and **avoiding false positives**.

---

## 📜 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙌 Acknowledgements

- [Google Gmail API Documentation](https://developers.google.com/gmail/api)
- [scikit-learn](https://scikit-learn.org/)
- [pandas](https://pandas.pydata.org/)
- [numpy](https://numpy.org/)
- [Phishing Email Dataset on Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset)

---

## ⭐️ How to Contribute

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## Author

**Shuvrajyoti Nath Mohajohn**  
[GitHub](https://github.com/ShuvrajyotiN)

---
