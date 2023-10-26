import streamlit as st
import io
import os
import PyPDF2
from langchain.llms import OpenAIChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate

# Function to extract text from a PDF file
def extract_text_from_binary(file):
    pdf_data = io.BytesIO(file)
    reader = PyPDF2.PdfReader(pdf_data)
    num_pages = len(reader.pages)
    text = ""

    for page in range(num_pages):
        current_page = reader.pages[page]
        text += current_page.extract_text()
    return text

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

# Define the path to your PDF file
resume_file_path = r"OmkarResume.pdf"  # Replace with your actual file path

with open(resume_file_path, 'rb') as file:
    pdf_text = extract_text_from_binary(file.read())

template = """Format the provided resume to this YAML template:
        ---
    NAME: ''

    PHONE NUMBER:
    - ''

    WEBSITES:
    - ''

    EMAILS:
    - ''

    DATE OF BIRTH: ''

    ADDRESS:
    - Street: ''
      City: ''
      State: ''
      Zip: ''
      Country: ''

    SUMMARY: ''

    EDUCATION:
    - School: ''
      Degree: ''
      Field Of Study: ''
      Start Date: ''
      End Date: ''

    WORK EXPERIENCE:
    - Company: ''
      Position: ''
      Start Date: ''
      End Date: ''

    SKILLS:
    - Name: ''

    CERTIFICATIONS:
    - Name: ''

    {chat_history}
    {human_input}"""

prompt = PromptTemplate(
        input_variables=["chat_history", "human_input"],
        template=template
    )

memory = ConversationBufferMemory(memory_key="chat_history")

llm_chain = LLMChain(
        llm=OpenAIChat(model="gpt-3.5-turbo"),
        prompt=prompt,
        verbose=True,
        memory=memory,
    )

# Streamlit app
def main():
    st.title("Resume Parser App")
    st.write("Upload a PDF file to extract information:")

    uploaded_file = st.file_uploader("Choose a PDF file...", type=["pdf"])

    if uploaded_file is not None:
        # Check if the file type is PDF
        if uploaded_file.type == "application/pdf":
            # Extract text from the PDF
            pdf_text = extract_text_from_binary(uploaded_file.read())

            # Parse the resume text
            parsed_resume = pdf_text  # Placeholder, replace with actual parsing code

            # Predict using the parsed resume
            res = llm_chain.predict(human_input=parsed_resume)

            # Display the result
            st.subheader("Result:")
            st.write(res)
        else:
            st.warning("Please upload a PDF file.")

if __name__ == "__main__":
    main()
