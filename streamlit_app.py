import streamlit as st
import openai
import os
# https://python.langchain.com/api_reference/langchain/chains/langchain.chains.llm.LLMChain.html
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import ChatOpenAI

# https://api.python.langchain.com/en/latest/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html


st.title("Tell us about your travel.")
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

# https://medium.com/data-professor/beginners-guide-to-openai-api-a0420bc58ee5
# https://github.com/elhamod/openaistreamlit/blob/main/streamlit_app.py
# https://python.langchain.com/api_reference/core/prompts/langchain_core.prompts.prompt.PromptTemplate.html

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
# https://medium.com/data-professor/beginners-guide-to-openai-api-a0420bc58ee5
llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# https://github.com/langchain-ai/langchain/discussions/11253
# https://stackoverflow.com/questions/78355357/llm-prompt-chain-does-not-behave-as-expected
airline_issue_response_chain = LLMChain(
    llm=llm, prompt=airline_related_issue_negative, output_parser=StrOutputParser()
)

outside_airline_control_response_chain = LLMChain(
    llm=llm, prompt=issues_outside_airline_control, output_parser=StrOutputParser()
)

positive_experience_response_chain = LLMChain(
    llm=llm, prompt=positive_experience_template, output_parser=StrOutputParser()
)
# https://docs.streamlit.io/develop/concepts/design/buttons
# https://stackoverflow.com/questions/74003574/how-to-create-a-button-with-hyperlink-in-streamlit
# https://docs.streamlit.io/develop/api-reference/widgets/st.link_button
if st.button("Submit Feedback"):
    if user_prompt:
        response = positive_experience_response_chain.run({"text": user_prompt}).strip()
        
        if not response:
            response = airline_issue_response_chain.run({"text": user_prompt}).strip()
            
            if not response:
                response = outside_airline_control_response_chain.run({"text": user_prompt}).strip()

        st.write(response)
    else:
        st.write("Please enter your experience.")
        
# https://github.com/langchain-ai/langchain/issues/1438
