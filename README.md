# End-to-End-ChatBot-Development
A streamlit application for new car purchase chatbot in the UK in 2024
### End to End ChatBot Development using Streamlit
  An END to END ChatBOt is simply the creation of a computer programme to handle conversation(based on a subject) from start to Finish. To achieve one needs to choose a programming language or NLP tools, collect the data(which includes the knowledge based), use the data to train and refine the chatbot, then deploy the chatbot on medium like whatsapp, telegram etc for user accessibility.This chatbot efficiency is limited to the data used to train it and can only perform like the LLM such as chatGPT using RAG(Retrieval Augmented Generation).

  --
* os module acts as a bridge between your Python code and the underlying operating system, allowing you to perform various tasks related to files, directories, processes, and environment variables
* NLTK - natural language tool kit is a python module for natural language processing. Just like Spacy,textBlob,polygot etc and  NLTK has features for text tokenization,stemming, named-entity recognition, part of speech tagging, sentiment analysis. However, nltk cannot be used for production purposes while spacy could
* SSL Secure Sockets Layer (SSL) is crucial in chatbot development, particularly in ensuring the security and privacy of data transmission in terms of fostering data confidentiality, authenticatication, data integrity, trust and compliance.
* Streamlit : for creation of interactive web applications Its reactive nature automatically updates the interface based on data changes or user interactions3.

After importing the above mentioned libraries, we initiate the nltk and secure socket layer functions for our chatbot as below:
ssl._create_default_https_context=ssl._create_unverified_context:

This is like telling your computer not to check the ID of a website when it connects.
Normally, your computer checks the website’s ID to make sure it’s really the site it claims to be.
But this line tells your computer to skip that check.

nltk.data.path.append(os.path.abspath("nltk_data")):

This is like telling a robot to also look in a specific folder for its instruction manuals.
The robot is a program called NLTK, and the instruction manuals are data it uses to understand language.


nltk.download("punkt"):

This is like downloading a new instruction manual for the robot.
The manual is called “punkt”, and it helps the robot understand where sentences start and end.




### Steps in building a typical chatbot
- Defining your intent
- Create Training data/Knowledge base
- Training the ChatBot
- Building the ChatBot
- Testing/Refining the ChatBot
- Deploying the ChatBot

===Create Training data/Knowledge base & Training the ChatBot
This is where we prepare the intents and train an ML for the chatbot



===Deploy the Car ChatBot Using STREAMLIT

Streamlit as mentioned above is used for the end to end chatbot deployment and it involves creating an
interface for a machine learning application where users can interact with the chatbot as an application
on the web without the backend code.


# Streamlit for real time Sentiment Analysis
= Streamlit is an open source python module or framework for creating and designing custom based application
for machine leaning and data science.

#To run streamlit

1. streamlit run your_script.py
