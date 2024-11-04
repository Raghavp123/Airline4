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
    template="""If the text below specifically describes a negative experience due to an issue with the airline, 
    generate a sympathetic response and mention that customer service will follow up to address the concern. 
    If the text does not mention an airline-related issue, respond with "We're sorry to hear about your experience and will work to improve.".\n\nText: {text}""",
    input_variables=["text"]
)

 
issues_outside_airline_control = PromptTemplate(
    template="""If the text below specifically describes a negative experience not related to the airline's control 
    (for example, a food issue or weather-related delay), provide a kind response acknowledging the issue and explain 
    that the airline is not liable but will take note of the feedback. If the text does not mention a relevant issue, 
    respond with "We value your feedback and are committed to improving all aspects of the travel experience.".\n\nText: {text}""",
    input_variables=["text"]
)

 
positive_experience_template = PromptTemplate(
    template="""If the text below clearly describes a positive experience, respond with a warm thank-you message. 
    If the text is unclear or does not appear positive, respond with "Thank you for sharing your thoughts with us; we aim to make every journey as enjoyable as possible!"\n\nText: {text}""",
    input_variables=["text"]
)


llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo")

# Different response chains
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
# Generate and display responses
if st.button("Submit Feedback"):
    if user_prompt:
        # Check for airline-related issues
        response = airline_issue_response_chain.run({"text": user_prompt}).strip()
        
        if response:
            st.write(response)
        else:
            # Check for issues outside of airline control
            response = outside_airline_control_response_chain.run({"text": user_prompt}).strip()
            
            if response:
                st.write(response)
            else:
                # Check for positive feedback
                response = positive_experience_response_chain.run({"text": user_prompt}).strip()
                
                if response:
                    st.write(response)
                else:
                    # Default empathy message if no specific response is generated
                    st.write("Thank you for your feedback. We value every input and are here to enhance your experience.")
    else:
        st.write("Please enter your experience.")
