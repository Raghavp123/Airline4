import streamlit as st
import openai
import os

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html

from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI
from langchain_core.runnables import RunnableBranch

# https://github.com/elhamod/openaistreamlit/blob/main/streamlit_app.py
# https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html
# https://medium.com/data-professor/beginners-guide-to-openai-api-a0420bc58ee5

st.title("Tell us about your travel.")
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

user_prompt = st.text_area("Tell us about your latest flight experience?")

airline_related_issue_negative = PromptTemplate(
    template="""If the text below mentions a negative experience specifically related to airline services (e.g., delays, service issues, lost luggage), respond with an empathetic message and assure that customer service will follow up to resolve any concerns. 
    Otherwise, respond with a general empathetic message.\n\nText: {text}""",
    input_variables=["text"]
)

issues_outside_airline_control = PromptTemplate(
    template="""If the text below describes a negative experience not caused by the airline (e.g., weather, traffic, airport food), respond with a kind message explaining that the airline isn’t responsible but still show empathy for the inconvenience.
    \n\nText: {text}""",
    input_variables=["text"]
)

positive_experience_template = PromptTemplate(
    template="""If the text below shares a positive experience (e.g., good service, pleasant flight), respond with a warm thank-you message like:
    "Thank you for sharing your positive experience with us! We're delighted to hear you enjoyed your journey, and we look forward to serving you again soon!"\n\nText: {text}""",
    input_variables=["text"]
)

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")
# https://github.com/langchain-ai/langchain/issues/1438
# https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html

# Chains for each type of response
airline_issue_response_chain = LLMChain(
    llm=llm, prompt=airline_related_issue_negative, output_parser=StrOutputParser()
)

outside_airline_control_response_chain = LLMChain(
    llm=llm, prompt=issues_outside_airline_control, output_parser=StrOutputParser()
)

positive_experience_response_chain = LLMChain(
    llm=llm, prompt=positive_experience_template, output_parser=StrOutputParser()
)
# https://medium.com/@twjjosiah/chain-loops-in-langchain-expression-language-lcel-a38894db0cee
#  routing logic based on keywords in the user prompt
# routing based on keywords in the user prompt
branch = RunnableBranch(
    (lambda x: any(word in x["text"].lower() for word in ["good", "pleasant", "fantastic", "positive"]), positive_experience_response_chain),
    (lambda x: any(word in x["text"].lower() for word in ["delay", "lost", "service", "airline"]), airline_issue_response_chain),
    outside_airline_control_response_chain  
)

# response based on the user input
if st.button("Submit Feedback"):
    if user_prompt:
        result = branch.invoke({"text": user_prompt})  
        response = result.get("text", "").strip()  
        st.write(response)
    else:
        st.write("Please enter your experience.")
