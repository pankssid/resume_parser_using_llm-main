import io,os,re
import PyPDF2
import pandas as pd
import streamlit as st
from langchain.llms import OpenAIChat
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain import LLMChain, PromptTemplate


def extract_text_from_binary(file):
    pdf_data = io.BytesIO(file)
    reader = PyPDF2.PdfReader(pdf_data)
    num_pages = len(reader.pages)
    text = ""

    for page in range(num_pages):
        current_page = reader.pages[page]
        text += current_page.extract_text()
    return text

# Define the folder containing resumes
resume_folder_path = 'content/resumes/'

# Define your job description here
template = """
Compare the following Job Title and Responsibilities with the provided, Display the matching percentage of scores:

Context:
- In the field of Data Science, it's crucial to have a strong foundation in various algorithms and models.

**Job Title:** Data Scientist (Machine Learning)

**Responsibilities:**
1. Develop and implement machine learning models for data analysis and predictive modeling.
2. Collaborate with cross-functional teams to gather and analyze data, ensuring accuracy and relevance.
3. Explore and apply advanced statistical techniques to derive actionable insights from large datasets.
4. Continuously research and stay updated on the latest advancements in machine learning and data science.
5. Communicate findings and recommendations to stakeholders in a clear and understandable manner.

Format the answer in below YAML Format:
---
Total Experience: ''
Matching Percentage of Score: ''

{human_input}
"""

prompt = PromptTemplate(
        input_variables=["human_input"],
        template=template
    )

# Initialize an empty list to store the results
results_list = []



# Streamlit app
def main():
    st.title("Resume Parser App")
    st.write("Upload a PDF file to extract information:")   

    uploaded_file = st.file_uploader("Choose a PDF file...", type=["pdf"],accept_multiple_files=True)

    # Create LangChain instance
    lm_chain = LLMChain(
            llm=OpenAIChat(model="gpt-3.5-turbo"),
            prompt=prompt,
            verbose=True,
        )

    for filename in uploaded_file:
    # if filename.endswith('.pdf'):
        # resume_file_path = os.path.join(resume_folder_path, filename)

        # Extract text from the resume
        with open(filename, 'rb') as file:
            pdf_text = extract_text_from_binary(file.read())

        # Get the result using LangChain
        res = llm_chain.predict(human_input=pdf_text)

        # Use regular expression to find and extract the score
        matching_percentage = re.search(r"Matching Percentage of Score: '(\d+)%'", res)

        if matching_percentage:
            matching_percentage = int(matching_percentage.group(1))
        else:
            matching_percentage = None

        # Append the result to the list
        results_list.append({
            'Resume File': filename,
            'Matching Percentage': matching_percentage
        })

        st.write(results_list)

    # if uploaded_file is not None:
    #     # Check if the file type is PDF
    #     if uploaded_file.type == "application/pdf":
    #         # Extract text from the PDF
    #         pdf_text = extract_text_from_binary(uploaded_file.read())

    #         # Parse the resume text
    #         parsed_resume = pdf_text  # Placeholder, replace with actual parsing code

    #         # Predict using the parsed resume
    #         res = llm_chain.predict(human_input=parsed_resume)

    #         # Display the result
    #         st.subheader("Result:")
    #         st.write(res)
    #     else:
    #         st.warning("Please upload a PDF file.")

if __name__ == "__main__":
    main()





# Iterate through all files in the folder
for filename in os.listdir(resume_folder_path):
    if filename.endswith('.pdf'):
        resume_file_path = os.path.join(resume_folder_path, filename)

        # Extract text from the resume
        with open(resume_file_path, 'rb') as file:
            pdf_text = extract_text_from_binary(file.read())

        # Create LangChain instance
        llm_chain = LLMChain(
            llm=OpenAIChat(model="gpt-3.5-turbo"),
            prompt=prompt,
            verbose=True,
        )

        # Get the result using LangChain
        res = llm_chain.predict(human_input=pdf_text)

        # Use regular expression to find and extract the score
        matching_percentage = re.search(r"Matching Percentage of Score: '(\d+)%'", res)

        if matching_percentage:
            matching_percentage = int(matching_percentage.group(1))
        else:
            matching_percentage = None

        # Append the result to the list
        results_list.append({
            'Resume File': filename,
            'Matching Percentage': matching_percentage
        })

# Convert the list of dictionaries to a DataFrame
results_df = pd.DataFrame(results_list)

# Print the results DataFrame
#results_df = pd.DataFrame(results_list)

# Replace NaN values with 0
results_df['Matching Percentage'].fillna(0, inplace=True)

# Print the results DataFrame
print(results_df)