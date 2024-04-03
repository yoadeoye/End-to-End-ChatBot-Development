# import the neccessary libraries and dependencies
import os
import nltk
import ssl #secret socket layer
import streamlit as st
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

ssl._create_default_https_context=ssl._create_unverified_context
nltk.data.path.append(os.path.abspath("nltk_data"))
nltk.download("punkt")

# Defining some of the chatbot intent

intents = [
    {
        "tag": "greeting",
        "patterns": ["Hi", "Hello", "Hey", "How are you", "What's up", "How are you", "How are you today",
                     "How are you doing"],
        "responses": ["Hi there", "Hello", "Hey", "I'm fine", "Thank you", "Nothing much"]
    },
    {

        "tag": "goodbye",
        "patterns": ["Bye", "See you later", "Goodbye", "Take care"],
        "responses": ["Goodbye", "See you later", "Take care"]
    },
    {
        "tag": "thanks",
        "patterns": ["Thank you", "Thanks", "Thanks a lot", "I appreciate it"],
        "responses": ["You're welcome", "No problem", "Glad I could help"]
    },
    {
        "tag": "about",
        "patterns": ["What can you do", "Who are you", "What are you", "What is your purpose",
                     "Tell me about yourself"],
        "responses": ["I am a chatbot", "My purpose is to assist you", "I can answer questions and provide assistance"]
    },
    {

        "tag": "help",
        "patterns": ["Help", "I need help", "Can you help me", "What should I do"],
        "responses": ["Sure, what do you need help with?", "I'm here to help. What's the problem?",
                      "How can I assist you?", "How can I assist you"]
    },
    {
        "tag": "age",
        "patterns": ["How old are you", "What's your age", "what is your date of birth", "what is your release date"],
        "responses": ["I don't have an age. I'm a chatbot.", "I was just born in the digital world.",
                      "Age is just a number for me."]

    },
    {
        "tag": "weather",
        "patterns": ["What's the weather like", "How's the weather today", "Tell me about today's weather"],
        "responses": ["I'm sorry, I cannot provide real-time weather information.",
                      "You can check the weather on a weather app or website."]
    },
    {
        "tag": "budget",
        "patterns": ["How can I make a budget", "What's a good budgeting strategy", "How do I create a budget"],
        "responses": [
            "To make a budget, start by tracking your income and expenses. Then, allocate your income towards essential expenses like rent,food, and bills. Next, allocate some of your income towards savings and debt repayment. Finally, allocate the remainder of your income towards discretionary expenses like entertainment and hobbies.",
            "A good budgeting strategy is to use the 50/30/20 rule. This means allocating 50% of your income towards essential expenses, 30% towards discretionary expenses, and 20% towards savings and debt repayment.",
            "To create a budget, start by setting financial goals for yourself. Then, track your income and expenses for a few months to get a sense of where your money is going. Next, create a budget by allocating your income towards essential expenses, savings and debt repayment, and discretionary expenses."]
    },
    {
        "tag": "credit_score",
        "patterns": ["What is a credit score", "How do I check my credit score", "How can I improve my credit score"],
        "responses": [
            "A credit score is a number that represents your creditworthiness. It is based on your credit history and is used by lenders to determine whether or not to lend you money. The higher your credit score, the more likely you are to be approved for credit.",
            "You can check your credit score for free on several websites such as Credit Karma and Credit Sesame,Clear Score, Experian."]
    }, {
        "tag": "buying_car_uk_2024",
        "patterns": ["What should I consider when buying a new car in the UK in 2024?",
                     "What are the steps to buy a new car in the UK?",
                     "Can you guide me on buying a new car in the UK in 2024?"],
        "responses": [
            "When buying a new car in the UK in 2024, consider the following steps: 1. Determine your budget and the type of car you need. 2. Research about the car's fuel efficiency, insurance costs, and road tax. 3. Check for any government incentives for electric or hybrid cars. 4. Test drive the car before purchasing. 5. Review the warranty and after-sales service.",
            "Sure, here are some tips: 1. Decide on the type of car (SUV, sedan, hatchback, etc.) and the fuel type (petrol, diesel, electric, etc.). 2. Set a budget considering the total cost of ownership, not just the purchase price. 3. Research about the car's performance, safety features, and customer reviews. 4. Consider the resale value of the car. 5. Check for financing options. 6. Always take a test drive.",
            "Of course, when buying a new car in the UK in 2024, you should: 1. Decide what you need the car for (commuting, family, off-road, etc.) and choose the type accordingly. 2. Set a budget and stick to it. Remember to factor in running costs such as fuel, insurance, and maintenance. 3. Research the market for the best deals and discounts. 4. If you're considering an electric vehicle, check the range and charging options. 5. Finally, make sure to read the terms and conditions of the sale carefully."]
    },
    {
        "tag": "negotiating_car_deal",
        "patterns": ["How do I negotiate a good deal when buying a new car?",
                     "What are some negotiation strategies for buying a car?",
                     "Can you give me tips on negotiating a car price?"],
        "responses": [
            "When negotiating a good deal for a new car, consider the following: 1. Do your research on the car's market value. 2. Be prepared to walk away if the deal doesn't meet your expectations. 3. Don't be afraid to negotiate on things other than price, like additional features or services. 4. Be patient and don't rush the process.",
            "Negotiating a car deal involves several steps: 1. Know the market value of the car. 2. Have a maximum price in mind and stick to it. 3. Negotiate the total car price, not the monthly payment. 4. Don't be afraid to walk away if the deal isn't right. 5. Consider getting pre-approved for a loan to know your budget.",
            "Here are some tips for negotiating a car price: 1. Do your homework and know the car's market value. 2. Keep your emotions in check and be ready to walk away. 3. Negotiate the total price, not the monthly payment. 4. Don't rush the process. 5. Consider getting a pre-approved loan to strengthen your bargaining position."]
    },
    {
        "tag": "reliable_dealerships",
        "patterns": ["Can you recommend any reliable dealerships?", "Where should I buy a new car?",
                     "What are some reputable car dealerships?"],
        "responses": [
            "While I can't recommend specific dealerships, I can suggest some general tips: 1. Look for dealerships that have good customer reviews. 2. Consider dealerships that offer certified pre-owned vehicles. 3. Check if the dealership is accredited by the Better Business Bureau or a similar organization. 4. Visit multiple dealerships to compare prices and services.",
            "Choosing where to buy a new car depends on several factors: 1. Customer reviews and ratings of the dealership. 2. The dealership's reputation for customer service. 3. The range of cars the dealership offers. 4. The dealership's policies on warranties and returns. 5. The dealership's location and accessibility.",
            "Finding a reputable car dealership involves: 1. Checking customer reviews and ratings. 2. Looking at the dealership's inventory to see if they have the car you want. 3. Checking if the dealership offers good after-sales service. 4. Visiting the dealership to get a feel for their customer service. 5. Comparing prices and services offered by different dealerships."]
    },
    {
        "tag": "best_car_brands_uk",
        "patterns": ["What are the best car brands in the UK?", "Which car brands are popular in the UK?",
                     "Can you tell me about the top car brands in the UK?"],
        "responses": [
            "The best car brands in the UK can vary depending on what you're looking for in a car. However, some consistently high-ranking brands based on various factors such as reliability, customer satisfaction, and resale value include: Ford, Vauxhall, Volkswagen, Mercedes-Benz, and BMW.",
            "Popular car brands in the UK include a mix of domestic and foreign manufacturers. Some of the most popular brands are Ford, Vauxhall, Volkswagen, Audi, and BMW. These brands are popular for their reliability, performance, and wide range of models.",
            "Top car brands in the UK include a variety of manufacturers. Ford and Vauxhall are popular for their range of affordable and reliable models. German brands like Volkswagen, Mercedes-Benz, and BMW are known for their quality and performance. Other notable brands include Audi, Nissan, and Toyota."]
    },
    {
        "tag": "cheap_cars",
        "patterns": ["Can you recommend cars less than £1500?", "What are some affordable cars?",
                     "I need a car but my budget is tight, any recommendations?"],
        "responses": [
            "Finding a car for less than £1500 can be challenging, but it's not impossible. You might want to consider older models of cars like Ford Fiesta, Vauxhall Corsa, or Peugeot 206. Remember to check the car's condition and history before making a purchase. Please do further research to find the best deal.",
            "At this price range, you're likely looking at used cars. Some models to consider might include the Ford Focus, Honda Civic, or Toyota Corolla. Always check the vehicle's history and consider having it inspected by a mechanic before purchase. Please do further research to find the best deal.",
            "In the under £1500 range, you might want to look at used cars. Some reliable models that often fall in this price range include the Ford Fiesta, Vauxhall Astra, and Peugeot 206. Remember to check the car's mileage, service history, and overall condition. Please do further research to find the best deal."]
    },
    {
        "tag": "fuel_efficient_cars",
        "patterns": ["Can you recommend reliable and fuel efficient cars?", "What are some cars with good gas mileage?",
                     "I'm looking for a car that's easy on the wallet when it comes to fuel, any suggestions?"],
        "responses": [
            "If you're looking for reliable and fuel-efficient cars, consider models like the Toyota Prius, Ford Fiesta, or Honda Civic. These cars are known for their fuel efficiency and reliability. However, fuel efficiency can vary based on driving conditions and maintenance. Please do further research to find the best car for your needs.",
            "Some cars known for their fuel efficiency include the Toyota Corolla, Honda Fit, and Hyundai Ioniq. These models are not only fuel-efficient but also have a reputation for reliability. Please do further research to find the best car for your needs.",
            "If fuel efficiency is a priority, consider hybrid or electric vehicles. Models like the Toyota Prius, Nissan Leaf, or Chevrolet Bolt are known for their fuel efficiency. For non-electric options, consider cars like the Honda Civic or Ford Fiesta. Please do further research to find the best car for your needs."]
    },
    {
        "tag": "used_cars",
        "patterns": ["How can I get a fairly used car?", "Where can I buy used cars?",
                     "What should I look for when buying a used car?"],
        "responses": [
            "When looking for a used car, you can check online marketplaces, visit local dealerships, or look at car auctions. Always check the car's history, ask for a mechanic's inspection, and take a test drive before purchasing. Please do further research to find the best deal.",
            "You can find used cars on online platforms like Auto Trader, Gumtree, or eBay Motors. You can also visit local dealerships or car supermarkets. Remember to check the car's history report and consider having it inspected by a mechanic. Please do further research to find the best deal.",
            "Buying a used car involves several steps: 1. Determine your budget and the type of car you need. 2. Research to find cars that fit your criteria. 3. Check the car's history report. 4. Inspect the car or have it inspected by a professional. 5. Negotiate the price. Please do further research to find the best deal."]
    },
    {
        "tag": "family_cars",
        "patterns": ["Can you recommend cars for a family of 4, 5 or 6?", "What are some good family cars?",
                     "I need a car for my family, any suggestions?"],
        "responses": [
            "For a family of 4 to 6, consider cars with ample space and safety features. Some options might include the Ford Galaxy, Volkswagen Touran, or Peugeot 5008. These cars offer plenty of room for passengers and luggage, and they have good safety ratings. Please do further research to find the best car for your family.",
            "If you're looking for a family car, consider models like the Skoda Octavia, Ford S-Max, or Honda CR-V. These cars are known for their spacious interiors, safety features, and reliability. Please do further research to find the best car for your family.",
            "For families, you might want to consider SUVs or minivans as they offer more space and comfort. Models like the Kia Sorento, Volkswagen Sharan, or Ford Galaxy could be suitable. These vehicles offer plenty of seating, cargo space, and safety features. Please do further research to find the best car for your family."]
    },
    {
        "tag": "car_insurance",
        "patterns": ["How do I get car insurance?", "What are some good car insurance companies?",
                     "Can you explain how car insurance works?"],
        "responses": [
            "Car insurance can be obtained from a variety of providers. You'll need to provide some personal information, details about the car, and your driving history. It's important to shop around and get quotes from multiple providers to ensure you're getting the best deal. Please do further research to find the best insurance for your needs.",
            "There are many reputable car insurance companies. Some of the most popular ones in the UK include Aviva, Direct Line, and Admiral. However, the best company for you will depend on your specific needs and circumstances. Please do further research to find the best insurance for your needs.",
            "Car insurance provides financial protection against physical damage or bodily injury resulting from traffic collisions and against liability that could also arise from incidents in a vehicle. It may also offer financial protection against theft of the vehicle and damage to the vehicle sustained from events other than traffic collisions. Please do further research to understand how it works."]
    },
    {
        "tag": "car_loan",
        "patterns": ["How do I get a car loan?", "What are some good car loan providers?",
                     "Can you explain how car loans work?"],
        "responses": [
            "Car loans can be obtained from banks, credit unions, or online lenders. You'll need to provide some personal information, details about the car, and your financial information. It's important to shop around and get quotes from multiple lenders to ensure you're getting the best deal. Please do further research to find the best loan for your needs.",
            "There are many reputable car loan providers. Some of the most popular ones in the UK include Barclays, Santander, and Nationwide. However, the best provider for you will depend on your specific needs and circumstances. Please do further research to find the best loan for your needs.",
            "A car loan is a type of personal loan used to buy a car. The loan is repaid over a set term with interest. The interest rate can be fixed or variable, and the term can range from one to seven years. Please do further research to understand how it works."]
    },
    {
        "tag": "car_maintenance",
        "patterns": ["How do I maintain a car?", "What are some car maintenance tips?",
                     "Can you explain how car maintenance works?"],
        "responses": [
            "Car maintenance involves regular servicing, checking fluid levels, tire pressure, and brakes, and keeping the car clean. Regular maintenance can help prevent breakdowns and prolong the life of your car. Please do further research to understand how it works.",
            "Some car maintenance tips include regular oil changes, checking tire pressure and tread depth, replacing windshield wipers, and keeping the car clean. It's also important to have your car serviced regularly by a professional. Please do further research to find the best maintenance tips for your car.",
            "Car maintenance involves regular checks and services to keep the car in good working condition. This includes oil changes, tire rotations, brake checks, and more. Regular maintenance can help prevent major repairs down the line. Please do further research to understand how it works."]
    },
    {
        "tag": "electric_cars",
        "patterns": ["What are some good electric cars?", "Can you recommend any electric cars?",
                     "What should I consider when buying an electric car?"],
        "responses": [
            "There are many great electric cars on the market. Some of the most popular ones in the UK include the Nissan Leaf, Tesla Model 3, and Renault Zoe. When buying an electric car, consider the range, charging options, and cost. Please do further research to find the best electric car for your needs.",
            "Electric cars are becoming increasingly popular. Some top models include the Tesla Model S, BMW i3, and Hyundai Kona Electric. When considering an electric car, think about the range, charging infrastructure in your area, and any government incentives. Please do further research to find the best electric car for your needs.",
            "When buying an electric car, consider factors like the range, charging time, cost, and availability of charging stations. Some popular electric cars include the Nissan Leaf, Tesla Model 3, and Chevrolet Bolt. Please do further research to find the best electric car for your needs."]
    },
    {
        "tag": "used_car_history",
        "patterns": ["How do I check a used car's history report?", "Where can I get a car's history report?",
                     "What should I look for in a car's history report?"],
        "responses": [
            "You can check a used car's history report through services like HPI Check or Experian Auto Check in the UK. These reports can provide information about the car's past, including any accidents, title problems, mileage readings, and more. It's a good idea to check this report before buying a used car. Please do further research to understand how to interpret the report.",
            "A car's history report can be obtained from various online services. In the UK, popular providers include HPI Check, Experian Auto Check, and My Car Check. These reports can reveal potential issues such as outstanding finance, stolen vehicle records, and previous accident damage. Please do further research to find the best service for your needs.",
            "When looking at a car's history report, check for any record of accidents, title issues, and whether the car was ever reported stolen or salvaged. Also, verify the mileage and ensure it matches up with the car's condition and age. Please do further research to understand how to interpret the report."]
    }

]

# Create Training data/Knowledge base & Training the ChatBot
#
# This is where we prepare the intents and train an ML for the chatbot

# initiate the vectorizer and  the model to use

vectorizer= TfidfVectorizer()
model=LogisticRegression(random_state=0,max_iter=10000)

# data preprocessing
tags=[]
patterns=[]
for intent in intents:
    for pattern in intent['patterns']:
        tags.append(intent['tag'])
        patterns.append(pattern)
# training the chatbot
X=vectorizer.fit_transform(patterns) # independent variable
y=tags   # target variable

# fit the model chatbot
model.fit(X,y)


# Defining a function to building the trained chatbot

def new_car_chatbot(input_text):
    input_text = vectorizer.transform([input_text])
    tag = model.predict(input_text)[0]
    for intent in intents:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            return response


# Deploy the Car ChatBot Using STREAMLIT
#
# Streamlit as mentioned above is used for end to end chat deployment and it involves creating an interface for
# a machine learning application where users can interact with the chatbot as an application on the web without the backend code.


counter = 0


def main():
    global counter
    st.header("Car ChatBot".upper())
    st.markdown("Car Chatbot: This chatbot is focused on giving responses to purchasing of a new or used cars in the UK in 2024."
                " From cheap cars, car maintenance to car insurance and fuel efficient cars")
    st.divider()  # 👈 Draws a horizontal rule
    st.title("Car ChatBot")
    st.write("Welcome to new car purchase chatbot in the United Kingdom. Please type a message and press Enter. ")
    st.divider()  # 👈 Draws a horizontal rule

    counter += 1
    user_input = st.text_input("You:", key=f"user_input_{counter}")

    if user_input:
        #
        response = new_car_chatbot(user_input)
        st.text_area("Car ChatBot:", value=response, height=100, max_chars=None, key=f"chatbot_response_{counter}")

        if response.lower() in ["goodbye", "bye"]:
            st.write("Thank You for chatting with me.Have a great day!")
            st.stop()


if __name__ == "__main__":
    main()