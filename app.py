import streamlit as st
import pandas as pd
# import time
import imaplib
import email
from transformers import AutoTokenizer
import transformers
# import torch
from datetime import datetime, date
from email.header import decode_header
# import webbrowser
# import os
import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string
nltk.download('stopwords')
nltk.download('punkt')


# Replace this with your actual Python code that generates the DataFrame
def generate_dataframe():
  # Your logic to create the DataFrame

    # account credentials
  username = "sai_bavisetti@outlook.com"
  password = "Outlook@1998"
  # use your email provider's IMAP server, you can look for your provider's IMAP server on Google
  imap_server = "imap-mail.outlook.com"
  imap = imaplib.IMAP4_SSL(imap_server)
  # authenticate
  imap.login(username, password)
  status, messages = imap.select("INBOX")
  # total number of emails
  messages = int(messages[0])
  # messages
  # original_date = datetime.strptime("2024-03-13",'%Y-%m-%d')
  original_date = date.today()
  # Format the date as "DD-Mon-YYYY"
  formatted_date = original_date.strftime("%d-%b-%Y").upper()

  # print(formatted_date)
  search_criteria = "(ON "+'"'+formatted_date+'"'+")"
  # search_criteria

  # s1,s2=imap.search(None, '(ON "19-MAR-2024")')
  s1,s2=imap.search(None, search_criteria)
  N = int(len(s2[0].split()))
  # N
  Email_df = pd.DataFrame(columns = ['Subject', 'From', 'Body'], index=range(N))
  count = 0
  #
  for i in range(messages, messages-N, -1):
      # fetch the email message by ID
      res, msg = imap.fetch(str(i), "(RFC822)")
      for response in msg:
          if isinstance(response, tuple):
              # parse a bytes email into a message object
              msg = email.message_from_bytes(response[1])
              # decode the email subject
              subject, encoding = decode_header(msg["Subject"])[0]
              if isinstance(subject, bytes):
                  # if it's a bytes, decode to str
                  subject = subject.decode(encoding)

              # decode email sender
              From, encoding = decode_header(msg.get("From"))[0]
              if isinstance(From, bytes):
                  From = From.decode(encoding)

              # print("Subject:", subject)
              # print("From:", From)
              Email_df.loc[count,'Subject'] = subject
              Email_df.loc[count,'From'] = From
              if msg.is_multipart():
                  # iterate over email parts
                  for part in msg.walk():
                      # extract content type of email
                      content_type = part.get_content_type()
                      content_disposition = str(part.get("Content-Disposition"))
                      try:
                          # get the email body
                          body = part.get_payload(decode=True).decode()
                      except:
                          pass
                      if content_type == "text/plain" and "attachment" not in content_disposition:
                          # print text/plain emails and skip attachments
                          Email_df.loc[count,'Body'] = '"""'+body+'"""'
                          # print(body)
              else:
                  # extract content type of email
                  content_type = msg.get_content_type()
                  # get the email body
                  body = msg.get_payload(decode=True).decode()
                  if content_type == "text/plain":
                      # print only text email parts
                      Email_df.loc[count,'Body'] = '"""'+body+'"""'
                      # print(body)
      count=count+1

  email_subjects_sample=pd.read_csv("Email_subject_sample_data.csv")
  # email_subjects_sample
  email_body_sample=pd.read_csv("Email_body_sample_data.csv")
  # email_body_sample

  def calculate_similarity(text1, text2):
      vectorizer = TfidfVectorizer()
      tfidf_matrix = vectorizer.fit_transform([text1, text2])
      similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
      return similarity_matrix[0, 1]

  # Function to clean and preprocess text
  def clean_text(text):
      # Convert to lowercase
      text = text.lower()

      # Remove punctuation
      text = text.translate(str.maketrans('', '', string.punctuation))

      # Remove stopwords
      stop_words = set(stopwords.words('english'))
      words = word_tokenize(text)
      filtered_text = [word for word in words if word.lower() not in stop_words]

      # Join the words back into a string
      cleaned_text = ' '.join(filtered_text)

      return cleaned_text
  model = "facebook/bart-large-cnn"

  tokenizer = AutoTokenizer.from_pretrained(model)
  summarizer = transformers.pipeline("summarization", model=model)
  # summarizer = pipeline("summarization", model=model)


  for i in range(len(Email_df)):
      Email_df.loc[i,"Cleaned_subject"]=clean_text(Email_df.loc[i,'Subject'])
      max_length = 1024 #len(Email_df.loc[i,'Body'])
      Email_df.loc[i,"Body_summary"]=summarizer(Email_df.loc[i,'Body'], max_length=max_length, min_length=20, do_sample=False, truncation=True)[0].get('summary_text')
      sim_scores = []
      for j in range(len(email_subjects_sample)):
          sim_scores.append(calculate_similarity(email_subjects_sample.loc[j,'Cleaned_Subject_sample'],Email_df.loc[i,"Cleaned_subject"]))
      # print(sim_scores)
      Email_df.loc[i,"Prob_subject_class"]=email_subjects_sample.loc[sim_scores.index(max(sim_scores)), "Class"]
      Email_df.loc[i,"Similarity_subject"] = max(sim_scores)

  Email_df.loc[Email_df["Similarity_subject"]<=0.2, "Prob_subject_class"] = "None"

  for i in range(len(Email_df)):
      Email_df.loc[i,"Cleaned_body"]=clean_text(Email_df.loc[i,'Body_summary'])
      # max_length = len(Email_df.loc[i,'Body'])
      # Email_df.loc[i,"Body_summary"]=summarizer(Email_df.loc[i,'Body'], max_length=max_length, min_length=20, do_sample=False, truncation=True)[0].get('summary_text')
      sim_scores = []
      for j in range(len(email_body_sample)):
          sim_scores.append(calculate_similarity(email_body_sample.loc[j,'Cleaned_body_sample'],Email_df.loc[i,"Cleaned_body"]))
      # print(sim_scores)
      Email_df.loc[i,"Prob_body_class"]=email_body_sample.loc[sim_scores.index(max(sim_scores)), "Class"]
      Email_df.loc[i,"Similarity_body"] = max(sim_scores)

  Email_df.loc[Email_df["Similarity_body"]<=0.2, "Prob_body_class"] = "None"

  Email_df["Final_Class"] = np.where(Email_df["Similarity_body"]>=Email_df["Similarity_subject"],
                                    Email_df["Prob_body_class"], Email_df["Prob_subject_class"] )


  df = pd.DataFrame(Email_df)
  return df

def main():
  """Streamlit app to display output on button click."""
  
#   st.title(":blue[Email Sensei]")
  st.markdown("<h1 style='text-align: center; color: blue;'>Email Sensei</h1>", unsafe_allow_html=True)
  st.caption(":green[Please send an email to sai_bavisetti@outlook.com]")
#   email_id = st.text_input('Enter Outlook Email id:', 'sai_bavisetti@outlook.com')
#   password = st.text_input('Enter password:', type = "password")

  if st.button("Process Emails"):
    try:
      with st.container(height = 700):
        with st.spinner('Wait for it...'):
            #time.sleep(5)
            df = generate_dataframe()
        st.success("Check Your Result!")
        
        st.text_area("From", df.loc[0, 'From'])
        st.text_area("Subject", df.loc[0, 'Subject'])
        st.text_area("Body Summary", df.loc[0, 'Body_summary'])
        st.text_area("Category", df.loc[0, 'Final_Class'])
        st.dataframe(df[['Subject','From', 'Body_summary','Final_Class']])
        st.download_button(label="Download data as CSV",data=df[['Subject','From', 'Body_summary','Final_Class']].to_csv(),
                           file_name='Email_summary.csv',mime='text/csv')
    except Exception as e:
      st.error(f"Error executing: {e}")

if __name__ == "__main__":
  main()
