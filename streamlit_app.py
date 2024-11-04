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

airline_related_issue_negative = PromptTemplate(
    template="""If the text below talks about a bad experience because of an airline problem, 
    reply with a message saying you’re sorry and that customer service will reach out to help. 
    If it doesn’t, don’t reply.\n\nText: {text}""",
    input_variables=["text"]
)

issues_outside_airline_control = PromptTemplate(
    template="""If the text below talks about a bad experience not caused by the airline (like a bad food), 
    respond with a kind message, 
    explaining that the airline isn’t responsible.\n\nText: {text}""",
    input_variables=["text"]
)

positive_experience_template = PromptTemplate(
    template="""If the text below shares a good experience, reply with a warm thank-you message. For example:
    "Thank you so much for sharing your positive experience with us! We’re delighted to know that you enjoyed your journey, and we look forward to welcoming you on board again soon!"\n\nText: {text}""",
    input_variables=["text"]
)

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# Different responses
airline_issue_response_chain = LLMChain(
    llm=llm, prompt=airline_related_issue_negative, output_parser=StrOutputParser()
)

outside_airline_control_response_chain = LLMChain(
    llm=llm, prompt=issues_outside_airline_control, output_parser=StrOutputParser()
)

positive_experience_response_chain = LLMChain(
    llm=llm, prompt=positive_experience_template, output_parser=StrOutputParser()
)

# Generate and display responses
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
                    st.write("Thank you for your feedback.")
    else:
        st.write("Please enter your experience.")
