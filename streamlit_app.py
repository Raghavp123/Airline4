import streamlit as st
import openai
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI

st.title("Tell us about your travel.")
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

user_prompt = st.text_area("Tell us about your latest flight experience?")

Airline_related_issue_negative = PromptTemplate(
    template="""If the text below talks about a bad experience because of an airline problem, 
    reply with a message saying you’re sorry and that customer service will reach out to help. 
    If it doesn’t, don’t reply.\n\nText: {text}""",
    input_variables=["text"]
)

Issues_outside_airline_control = PromptTemplate(
    template="""If the text below talks about a bad experience not caused by the airline (like a bad food), 
    respond with a kind message, 
    explaining that the airline isn’t responsible.\n\nText: {text}""",
    input_variables=["text"]
)

positive_experience_template = PromptTemplate(
    template="""If the text below shares a good experience, reply with an appreciation message thanking the user for their time.\n\nText: {text}""",
    input_variables=["text"]
)

llm = ChatOpenAI(api_key=openai.api_key, model="gpt-3.5-turbo")

# Different responses
airline_issue_response_chain = LLMChain(
    llm=llm, prompt=Airline_related_issue_negative, output_parser=StrOutputParser()
)

outside_airline_control_response_chain = LLMChain(
    llm=llm, prompt=Issues_outside_airline_control, output_parser=StrOutputParser()
)

positive_experience_response_chain = LLMChain(
    llm=llm, prompt=positive_experience_template, output_parser=StrOutputParser()
)

# Give a reply based on the user’s input
if st.button("Submit Feedback"):
    if user_prompt:
        response = airline_issue_response_chain.run({"text": user_prompt}).strip()

        if response:
            st.write(response)
        else:
            response = outside_airline_control_response_chain.run({"text": user_prompt}).strip()

            if response:
                st.write(response)
            else:
                response = positive_experience_response_chain.run({"text": user_prompt}).strip()

                if response:
                    st.write(response)
                else:
                    # Default message if no specific response
                    st.write("Thank you for your feedback.")
    else:
        st.write("Please enter your experience.")
