import joblib

model = joblib.load('Spam_Detection_Model.pkl')
feature_vector = joblib.load('features.pkl')

from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

flow = InstalledAppFlow.from_client_secrets_file(
    'credentials.json',
    scopes=SCOPES,
    redirect_uri='urn:ietf:wg:oauth:2.0:oob'  # forces manual code copy/paste
)

auth_url, _ = flow.authorization_url(prompt='consent')
print(f"🔗 Visit this URL to authorize:\n{auth_url}")


code = input("🔑 Paste the code here: ")
flow.fetch_token(code=code)

creds = flow.credentials
service = build('gmail', 'v1', credentials=creds)
print("✅ Gmail API connected")


import base64
from bs4 import BeautifulSoup

def get_email_text(payload):
    if payload.get("parts"):
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"]["data"]
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            elif part["mimeType"] == "text/html":
                data = part["body"]["data"]
                html = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                return BeautifulSoup(html, "html.parser").get_text()
    elif payload.get("body", {}).get("data"):
        data = payload["body"]["data"]
        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
    return ""

def fetch_unlabeled_emails(service, max_results=10):
    emails = []
    response = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=max_results).execute()
    messages = response.get('messages', [])

    for msg in messages:
        msg_data = service.users().messages().get(userId='me', id=msg['id']).execute()
        payload = msg_data['payload']
        body = get_email_text(payload)
        subject = next((h['value'] for h in payload.get('headers', []) if h['name'] == 'Subject'), '')
        full_text = subject + " " + body
        emails.append(full_text)

    return emails

new_emails = fetch_unlabeled_emails(service, max_results=10)


X_new = feature_vector.transform(new_emails)
predictions = model.predict(X_new)

for i, pred in enumerate(predictions):
    label = "SPAM" if pred == 1 else "NOT SPAM"
    print(f"📧 Email {i+1} predicted as: {label}")
    print(f"📝 Preview:\n{new_emails[i][:300]}\n{'-'*60}")