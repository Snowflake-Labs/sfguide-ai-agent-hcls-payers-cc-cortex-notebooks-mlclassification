import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete
from snowflake.core import Root
import _snowflake
import snowflake.connector
import requests
import json
import os
from snowflake.snowpark import Session
import pandas as pd

session = get_active_session()
root = Root(session)

# Set pandas option to display all column content
pd.set_option("max_colwidth", None)

# Constants
DATABASE = session.get_current_database()
SCHEMA = session.get_current_schema()
STAGE = "RAW_DATA"
FILE = "DATA_PRODUCT/Call_Center_Member_Denormalized.yaml"

num_chunks = 1
num_transcripts = 2
slide_window = 3

def config_options():
    st.markdown(
        """
        <style>
             [data-testid=stSidebar] {
                background-color: ##5EA908;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # 'Start Over' button
    clear_conversation = st.sidebar.button('Start Over')

    if clear_conversation:
        # Reset selectboxes to default values
        st.session_state['phone_number'] = 'None Selected'
        st.session_state['model_name'] = 'llama3.1-70b'

        # Reset checkboxes and toggle to default values
        st.session_state['debug'] = False
        st.session_state['debug_prompt'] = False
   
        # Reset other relevant session state variables
        st.session_state.pop('restricted_member_toggle', None)
        st.session_state.pop('restricted_member', None)
        st.session_state.pop('messages', None)
        st.session_state.pop('suggestions', None)
        st.session_state.pop('active_suggestion', None)
        st.session_state.member_id = ""
        st.session_state.member_name = ""
        st.session_state.pop('caller_intent', None)
        st.session_state.pop('show_next_best_action', None)
        st.session_state.pop('phone_number_initialized', None)
        st.session_state.pop('active_predefined_question', None)
        st.session_state.pop('edited_subject', None)
        st.session_state.pop('edited_body', None)
        st.session_state.pop('trigger_action', None)

        st.sidebar.success("Conversation and selections have been reset.")

    # Initialize session state variables with default values if not already set
    st.sidebar.selectbox(
        'Select your model :',
        ('llama3.1-70b', 'mistral-large2'),
        key='model_name'
    )

    st.sidebar.selectbox(
        'Select your Cortex Complete Mode:',
        ('SQL', 'API'),
        key='cortex_complete_type'
    )

    st.sidebar.text_input(
        "Enter your email:", 
        "",
        key='user_email')

    st.sidebar.checkbox('Show prompt', key ='debug_prompt',value = False)
    st.sidebar.checkbox('Debug', key ='debug',value = False)

    st.session_state.setdefault('use_chat_history', True)
    st.session_state.setdefault('summarize_with_chat_history', True)
    st.session_state.setdefault('cortex_search', True)
    st.session_state.setdefault('debug_prompt', False)
    st.session_state.setdefault('debug', False)
    st.session_state.setdefault('member_name', '')
    st.session_state.setdefault('member_id', '')
    st.session_state.setdefault('active_predefined_question', None)
    st.session_state.setdefault('show_next_best_action', False)
    st.session_state.setdefault('edited_subject', '')
    st.session_state.setdefault('edited_body', '')
    st.session_state.setdefault('trigger_action', False)
    # Add any additional setdefaults as necessary

    return clear_conversation

def init_messages(clear_conversation):
    if clear_conversation or 'messages' not in st.session_state:
        st.session_state.messages = []
        st.session_state.suggestions = []
        st.session_state.active_suggestion = None
        st.session_state.pop('member_id', None)
        st.session_state.pop('member_id', None)
        st.session_state.pop('member_name', None)
        st.session_state.pop('caller_intent', None)
        # Do not reset 'phone_number' here
        st.session_state['show_next_best_action'] = False  # Reset the checkbox state
        # Properly reset 'phone_number_initialized' by removing it
        st.session_state.pop('phone_number_initialized', None)

def execute_cortex_complete(prompt):

    if st.session_state.cortex_complete_type == 'API':
       response_txt = execute_cortex_complete_api(f"""{prompt}.{st.session_state.restriction_prompt}""")
    else:
        response_txt = execute_cortex_complete_sql(f"""{prompt}.{st.session_state.restriction_prompt}""")
    return response_txt

def execute_cortex_complete_sql(prompt):
    cmd = "SELECT snowflake.cortex.complete(?, ?) AS response"
    df_response = session.sql(cmd, params=[st.session_state.model_name, prompt]).collect()
    response_txt = df_response[0].RESPONSE
    return response_txt

def execute_cortex_complete_api(prompt):    
    response_txt = Complete(
                    st.session_state.model_name,
                    prompt,
                    session=session
                    )

    return response_txt

@st.cache_data(show_spinner=False)
def get_similar_transcripts_cortex_search(question):
    response = (
        root.databases[DATABASE]
        .schemas[SCHEMA]
        .cortex_search_services["CALL_CENTER_RECORDING_SEARCH"]
        .search(
            question,
            ['CHUNK', 'RELATIVE_PATH'],
            limit=num_transcripts
        )
    )

    results = response.results
    df_chunks = pd.DataFrame(results, columns=['CHUNK', 'RELATIVE_PATH'])
    similar_chunks = ''.join(
        f"""
        ### 
        Beginning of Document {i+1}
        Document Name: {row['RELATIVE_PATH']}
        Content: {row['CHUNK']}
        End of Document {i+1}
        ####
        """
        for i, row in df_chunks.iterrows()
    ).replace("'", "")

    cmd2 = f"""
        SELECT get_presigned_URL(@{STAGE}, RELATIVE_PATH, 360) AS URL_LINK, RELATIVE_PATH
        FROM directory(@{STAGE})
        WHERE RELATIVE_PATH LIKE '%CALL_RECORDINGS%'
    """

    df_docs = session.sql(cmd2).to_pandas()
    df_docs['RELATIVE_PATH'] = df_docs['RELATIVE_PATH'].str.replace(r'CALL_RECORDINGS/', 'call_recordings/', regex=True)
    df_referred = pd.merge(df_chunks, df_docs, on='RELATIVE_PATH', how='inner')[['RELATIVE_PATH', 'URL_LINK']].drop_duplicates()

    return similar_chunks, df_referred

@st.cache_data(show_spinner=False)
def get_similar_chunks_cortex_search(question):
    response = (
        root.databases[DATABASE]
        .schemas[SCHEMA]
        .cortex_search_services["CALL_CENTER_FAQ_SEARCH"]
        .search(
            question,
            ['CHUNK', 'RELATIVE_PATH'],
            limit=num_chunks
        )
    )

    results = response.results
    df_chunks = pd.DataFrame(results, columns=['CHUNK', 'RELATIVE_PATH'])
    similar_chunks = ''.join(
        f"""
        ### 
        Beginning of Document {i+1}
        Document Name: {row['RELATIVE_PATH']}
        Content: {row['CHUNK']}
        End of Document {i+1}
        ####
        """
        for i, row in df_chunks.iterrows()
    ).replace("'", "")

    cmd2 = f"""
        SELECT get_presigned_URL(@{STAGE}, RELATIVE_PATH, 3600) AS URL_LINK, RELATIVE_PATH
        FROM directory(@{STAGE})
        WHERE RELATIVE_PATH LIKE '%FAQ%'
    """

    df_docs = session.sql(cmd2).to_pandas()
    df_docs['RELATIVE_PATH'] = df_docs['RELATIVE_PATH'].str.replace(r'FAQ/', 'faq/', regex=True)
    df_referred = pd.merge(df_chunks, df_docs, on='RELATIVE_PATH', how='inner')[['RELATIVE_PATH', 'URL_LINK']].drop_duplicates()
    
    return similar_chunks, df_referred

def get_chat_history():
    if st.session_state.use_chat_history:
        start_index = max(0, len(st.session_state.messages) - slide_window)
        chat_history = st.session_state.messages[start_index:]
        return chat_history
    return []

def summarize_question_with_history(chat_history, question):
    prompt = f"""
        You are a chatbot expert. Refer the latest question received by the chatbot, evaluate this in context of the Chat History found below. 
        Now share a refined query which captures the full meaning of the question being asked. 
        
        If the question appears to be a stand alone question ignore all previous interactions or chat history and focus solely on the question. 
        If it seem to be connected to the prior chat history, only then use the chat history.

        Please use the question as the prominent input and the Chat history as a support input when summarizing
        Answer with only the query. Do not add any explanation.

        Chat History: {chat_history}
        Question: {question}
    """

    summary = execute_cortex_complete(prompt)

    if st.session_state.debug:
        st.text("Summary to be used to find similar chunks")
        st.caption(summary)

    return summary

def create_prompt(myquestion, chat_history, intent):
    if st.session_state.cortex_search:
        if intent == 'recordings':
            prompt_context, df_document_urls = get_similar_transcripts_cortex_search(myquestion)
        else:
            prompt_context, df_document_urls = get_similar_chunks_cortex_search(myquestion)
    else:
        prompt_context = ""
        df_document_urls = pd.DataFrame()

    prompt = f"""
    You are an expert chat assistant that extracts information from the CONTEXT provided between <context> and </context> tags.
    You offer a chat experience considering the information included in the CHAT HISTORY provided between <chat_history> and </chat_history> tags.
    When answering the question contained between <question> and </question> tags, be concise and do not hallucinate.    
    If you don't have the information, just say so.

    Do not mention the CONTEXT in your answer.
    Do not mention the CHAT HISTORY in your answer.

    <context>
    {prompt_context}
    </context>
    <chat_history>
    {chat_history}
    </chat_history>
    <question>
    {myquestion}
    </question>
    Answer:
    """

    if st.session_state.debug_prompt:
        st.text(f"Prompt being passed to {st.session_state.model_name}")
        st.caption(prompt)

    return prompt, df_document_urls

def create_prompt_find_intent(myquestion):
    prompt = f"""
    You are an expert that classifies the question into one of the following categories:

    1. Recordings
    2. FAQ
    3. Data

    If the question is related to information that can be pulled from a table holding information on:
        - Gender
        - Address
        - Claims
        - Grievances
    Then respond with category as 'Data'.
    If the question is related to information from any prior call recordings or calls made to the call center, respond with category as 'Recordings'.
    If the question is related to generic information similar to what you would find in an FAQ document, respond with category as 'FAQ'.

    For e.g.

           Question : Can you give me a summary from the previous call made by Nicholas Carter
           Answer: Recordings
           
           Question : Can you share me the current benefit plan for Member ID M123456?
           Answer: Data

           Question : How can a member find out the available list of providers?
           Answer: FAQ

           Question : Can you share me the address information of Member Name Nicholas Carter ?
           Answer: Data

           Question : Can you share me all available information on claim ID C1022345?
           Answer: Data

           Question : Where there any recent changes on COVID coverages being offered?
           Answer: FAQ

    Be concise and ensure the response is strictly one word only and do not hallucinate.
    If you don't have the information, just say so.

    Do not mention the CONTEXT in your answer.
    Do not mention the CHAT HISTORY in your answer.

    <question>
    {myquestion}
    </question>
    Answer:
    """
    return prompt

def create_prompt_summarize_cortex_analyst_results(myquestion, df, sql):
    prompt = f"""
    You are an expert data analyst who translated the question contained between <question> and </question> tags:

    <question>
    {myquestion}
    </question>

    Into the SQL query contained between <SQL> and </SQL> tags:

    <SQL>
    {sql}
    </SQL>

    And retrieved the following result set contained between <df> and </df> tags:

    <df>
    {df}
    </df>

    Now share an answer to this question based on the SQL query and result set.
    Be concise and use mainly the CONTEXT provided and do not hallucinate.
    If you don't have the information, just say so.

    Whenever possible, arrange your response as bullet points.

    Example:
    - Claim ID:
    - Service:
    - Provider:

    Do not mention the CONTEXT in your answer. 

    Answer:
    """
    if st.session_state.debug_prompt:
        st.text(f"Prompt being passed to {st.session_state.model_name}")
        st.caption(prompt)

    return prompt

def complete(myquestion, chat_history, intent):
    prompt, df_document_urls = create_prompt(myquestion, chat_history, intent)
    response_txt = execute_cortex_complete(prompt)
    return response_txt, df_document_urls

def complete_for_cortex_analyst(prompt):
    response_txt = execute_cortex_complete(prompt)
    return response_txt

def find_question_type(myquestion):
    prompt = create_prompt_find_intent(myquestion)
    response_txt = execute_cortex_complete(prompt)
    if st.session_state.debug:
        st.text(f"Prompt Passed to intent finding LLM {prompt}")
        st.caption(f"""Intent found =  {response_txt}""")

    return response_txt

def find_violation(myquestion):
    if st.session_state.restricted_member:
        prompt = f"""
        You are an expert that determines whether a question violates the policy of accessing only data for the selected member.

        If the question contains any member names other than {st.session_state.member_name} or member IDs other than {st.session_state.member_id} , then respond with 'Yes'.
        Otherwise, respond with 'No'.

        Be concise and ensure the response is strictly one word only and do not hallucinate.

        <question>
        {myquestion}
        </question>
        Answer:
        """
        response_txt = execute_cortex_complete(prompt)
        response_txt = response_txt.strip().lower()
        return response_txt == 'yes'
    else:
        return False

def send_message(prompt: str) -> dict:
    request_body = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
    }
    resp = _snowflake.send_snow_api_request(
        "POST",
        f"/api/v2/cortex/analyst/message",
        {},
        {},
        request_body,
        {},
        30000,
    )
    if resp["status"] < 400:
        return json.loads(resp["content"])
    else:
        raise Exception(f"Failed request with status {resp['status']}: {resp}")

def process_message(prompt: str, question_summary: str, summary_msg: str):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        if len(st.session_state.messages) > 1:
            st.markdown(summary_msg)
        msg = f"""Based on the insights from Cortex AI, this seems to be a question appropriate for the Contact Center Member 360 Data Product. \n Initiating Contact Center Analyst agent to help answer this question."""
        st.write(msg)
        with st.spinner("Contact Center Analyst Agent thinking..."):
            response = send_message(prompt= f"{question_summary} . {st.session_state.restriction_prompt}")
            content = response["message"]["content"]
            response_string = display_content_new(content=content, prompt=question_summary)
    st.session_state.messages.append({"role": "assistant", "content": response_string})

def suggestion_click(suggestion):
    st.session_state.active_suggestion = suggestion

def display_content_new(content: list, prompt, message_index: int = None):
    message_index = message_index or len(st.session_state.messages)
    final_response = "Please refine that question"
    sql_statement = "Not Applicable"
    df = ""
    for item in content:
        if item["type"] == "text":
            final_response = item["text"]
        elif item["type"] == "suggestions":
            final_response = ""
            with st.expander("Suggestions", expanded=True):
                for suggestion_index, suggestion in enumerate(item["suggestions"]):
                    if st.button(
                        suggestion, key=f"{message_index}_{suggestion_index}", on_click=suggestion_click, args=[suggestion]
                    ):
                        st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            sql_statement = item["statement"]
            df = session.sql(sql_statement).collect()
            question_summary = prompt
            prompt_refined = create_prompt_summarize_cortex_analyst_results(question_summary, df, sql_statement)
            final_response = complete_for_cortex_analyst(prompt_refined)
        
        # Display the answer with some minimal refinement
        if item["type"] == "sql" or item["type"] == "suggestions":
            final_response = final_response.replace(
                    "There is no information available regarding any relevant supporting documentation.", ""
                )
            st.markdown(final_response)
        else:
            st.markdown(final_response)

        # Display SQL and Resultset if applicable
        if sql_statement != "Not Applicable":
            with st.expander("SQL Query"):
                st.code(sql_statement)
            with st.expander("Resultset"):
             st.write(df)

    return final_response

@st.cache_data(show_spinner=False)
def get_member_details(phone_number):
    query = f"""
        SELECT MEMBER_ID, NAME, 
        CASE WHEN POTENTIAL_CALLER_INTENT = 'Active Grievance' THEN POTENTIAL_CALLER_INTENT||':'||GRIEVANCE_TYPE
                      ELSE POTENTIAL_CALLER_INTENT
                END AS POTENTIAL_CALLER_INTENT,
        CASE 
                        WHEN POTENTIAL_CALLER_INTENT = 'Active Grievance' AND Grievance_Type = 'Inadequate Care' Then ' Retrieve related provider details as well'
                        WHEN POTENTIAL_CALLER_INTENT = 'Active Grievance' AND Grievance_Type = 'Delay in claim processing' Then ' Retrieve related Claim details as well'
                        ELSE '' 
                    END ADDITIONAL_INFO
        FROM {DATABASE}.{SCHEMA}.CALL_CENTER_MEMBER_DENORMALIZED_WITH_INTENT
        WHERE member_phone = ?
    """

    result_df = session.sql(query, params=[phone_number]).to_pandas()
    if len(result_df) == 0:
        return None, None, None, None  # Ensure four values are returned
    return (
        result_df['MEMBER_ID'].iloc[0],
        result_df['NAME'].iloc[0],
        result_df['POTENTIAL_CALLER_INTENT'].iloc[0],
        result_df['ADDITIONAL_INFO'].iloc[0]
    )

def on_phone_number_change():
    st.session_state.messages = [] #resetting messages
    phone_number = st.session_state.phone_number
    try:
        member_id, member_name, caller_intent, additional_info = get_member_details(phone_number)
        if member_id and member_name and caller_intent:
            st.session_state.member_id = member_id
            st.session_state.member_name = member_name
            st.session_state.caller_intent = caller_intent
            st.session_state.initial_chat_string = (
                f"Please share relevant information related to Member: {member_name} on {caller_intent}. {additional_info}"
            )
            if st.session_state.debug:
                st.write("Session State after retrieval:", st.session_state)  # Debugging
        else:
            st.sidebar.error("No data found for the selected phone number.")
            # Optionally, reset the session state if no data is found
            st.session_state.pop('member_id', None)
            st.session_state.pop('member_name', None)
            st.session_state.pop('caller_intent', None)
            st.session_state.pop('initial_chat_string', None)
    except Exception as e:
        st.sidebar.error(f"An error occurred during retrieval: {e}")
        # Optionally, reset the session state in case of an error
        st.session_state.pop('member_id', None)
        st.session_state.pop('member_name', None)
        st.session_state.pop('caller_intent', None)
        st.session_state.pop('initial_chat_string', None)

def display_member_info():
    st.markdown(
        """
        <style>
        .sidebar-info-box {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 7px;
            margin-bottom: 15px;
            text-align: center;
            font-family: Arial, sans-serif;
        }
        .info-title {
            font-size: 14px;
            font-weight: bold;
            color: #4b4b4b;
        }
        .info-value {
            font-size: 20px;
            color: #808080;
        }
        .critical_info-value {
            font-size: 15px;
            color: #2b8cbe;
        }
        .initial_q-value {
            font-size: 12px;
            color: #2b8cbe;
            font-style: italic;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Define the list of phone numbers
    phone_numbers = ['None Selected','946-081-0513','946-081-0564', '946-081-0696']

    # Define predefined questions for each phone number
    PREDEFINED_QUESTIONS = {
        '946-081-0564': [
            "I see the member has an active grievance related to Inadequate Care. Who is the provider associated with this grievance?",
            "What is the total number of active grievances for Inadequate Care associated with this provider. Include all members and not just the member in context.",
            "How does this compare against the average active grievances for Inadequate Care per provider for all members?"
        ],
        '946-081-0696': [
            "Has the member made any calls related to a delay in claim processing? If so, share a summary of this call.",
            "Share all information on this claim."
        ],
        '946-081-0513': [
            "How can the member find out more details about the wellness programs offered?",
            "Member wants to know how to find the member forms online?",
            "What is the plan and coverage information of this member?",
            "Give me the member information on Jessica Mills"
        ]
    }

    # Get the current phone number from session state or default to 'None Selected'
    current_phone_number = st.session_state.get('phone_number', 'None Selected')
    if current_phone_number in phone_numbers:
        index = phone_numbers.index(current_phone_number)
    else:
        index = 0  # Default to the first phone number

    # Selectbox for Incoming Call From with on_change callback
    phone_number = st.sidebar.selectbox(
        "Incoming call from... (Choose a sample #)", 
        options=phone_numbers, 
        index=index, 
        key='phone_number', 
        on_change=on_phone_number_change
    )

    predefined_questions = PREDEFINED_QUESTIONS.get(phone_number, [])

    if all(key in st.session_state for key in ('member_id', 'member_name', 'caller_intent')):
        # Display member info boxes
        st.sidebar.markdown(
            f"""
            <div class="sidebar-info-box">
                <div class="info-title">Member ID | Member Name</div>
                <div class="info-value">{st.session_state.member_id} | {st.session_state.member_name}</div>
            </div>
            <div class="sidebar-info-box">
                <div class="info-title">Predicted Caller Intent</div>
                <div class="critical_info-value">{st.session_state.caller_intent}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.sidebar.toggle('Limit question only on selected member',key ='restricted_member_toggle',value = True)
        if st.session_state.restricted_member_toggle:
            st.session_state.restricted_member = True
        else:
            st.session_state.restricted_member = False

        # Retrieve predefined questions based on the selected phone number
        predefined_questions = PREDEFINED_QUESTIONS.get(phone_number, [])

        if predefined_questions:
            st.sidebar.markdown("Sample Question(s)")
            for i, question in enumerate(predefined_questions):
                # Use unique keys for each button
                if st.sidebar.button(question, key=f"predefined_q_{phone_number}_{i}"):
                    # Handle the button click by setting the active predefined question
                    st.session_state['active_predefined_question'] = question
        else:
            st.sidebar.write("No predefined questions available for this phone number.")
    else:
        st.sidebar.write("Select a phone number to retrieve caller intent and member details.")

def determine_next_best_action(chat_history):
    prompt = f"""
    You are an intelligent call center assistant.
    Below is the conversation related to an ongoing call center interaction.
    Please analyze the chat Chat History:{chat_history}
    and determine:
    What is the most appropriate next action to take? 
    
        - If the member was requesting some FAQ oriented information
         Then regardless of whether this FAQ based information was provided on the chat. 
         It would be good to have the related details once again send out to {st.session_state.member_name} via an email.
        
        - If the member was raising a concern/request related to something else. And only if you are inferring that there is still some pending concern/request left unaddressed
         Then the best action to take would be to Send out a contextualized email addressing the concern/request from {st.session_state.member_name} to the appropriate internal team

       - If unable to determine
         Then respond stating 'Unable to determine next best action with information available'

    In scenarios where you see a reason for both an email to the member and email to the internal department need to go out
    Then choose only one among those emails in your recommended action to take.
       
    Refer the below information when defining the next best action    

    Internal Team Mapping    
        Claim Related Concerns (e.g. Inaccurate Billing, Delay in claim processing etc)  : Claim Ops
        New Enrollment Related Concerns  : Enrollment Ops
        Provider Related Concerns (e.g. Inadequate Care) : Provider Ops
    
    Please provide a short response in the following format:
    - [Action to take]
    
    
    Please don't include any additional info in the response.
    """
    
    try:
        response_txt = execute_cortex_complete(prompt)
        if not response_txt:
              return "Unable to determine next best action with information available."
        return response_txt
    except Exception as e:
        st.error(f"An error occurred while determining the next best action: {e}")
        return "Unable to determine next best action due to an internal error."

def generate_draft_action(chat_history, next_best_action):
    prompt = f"""
    You are an intelligent call center assistant.
    Below is the conversation related to an ongoing call center interaction.
    Please analyze the Chat History: {chat_history}  
    and the Next Best Action identified: {next_best_action}

    And perform either one of the two 

    1) When the next best action is regarding sending out some information to the member
       Then generate a contextualized email with the information at hand from chat history
        Refer the below information when crafting the email
        FAQ URL :https://www.enterprise-next.com/member/FAQS

    2) When the bext best action is to send an email to the appropriate internal department
        Then generate a contextualized email detailing out the member's concern/request and requesting action from the appropriate internal department
    accordance to the next best action determined. Add in additional information from the chat_history when appropriate

    Refer the below where needed
    Member Name : {st.session_state.member_name}
    Member ID: {st.session_state.member_id}

    Internal Team Mapping    
        Claim Related Concerns  : Claim Ops
        New Enrollment Related Concerns  : Enrollment Ops
        Provider Related Concerns : Provider Ops

    Email Subject should always start with the Member ID formatted as a text,remove any commas etc:
    Sign the email with 
    Enterprise Nxt Call Center Ops
    
    Based on this, generate a contextualized email in the following format:

    Subject: [Email Subject]
    Body:
    [Email Body]

    Example:
    Subject : 943130253 | Request related to inadequate Care

    Keep the email professional and concise. 
    When email is addressed to a member. Then add a line thanking them for beeing an member of Enterprise Next
    Do not repeat the same or similar information.
    
    Please don't include any additional info in the response.
    """

    response_txt = execute_cortex_complete(prompt)
    if response_txt:
        response = response_txt
    else:
        response = None
    return response

def send_email(recipient_email, subject, body):
    st.write(f"Sending an email to: {recipient_email}")
    cmd = """CALL SYSTEM$SEND_EMAIL(
             'my_email_int',
             ?,
             ?,
            ?
            );"""
    try:
        session.sql(cmd, params=[recipient_email, subject, body]).collect()
        st.success("Email sent successfully!")
    except:
        st.write("Please enter a valid email in the sidebar configs.")

def main():
    st.title(f"Payer Call Center Agent :robot_face:")
    st.subheader(f"Powered by Snowflake Cortex :snowflake::snowflake:")

    clear_conversation = config_options()
    init_messages(clear_conversation)
    display_member_info()

    if 'restricted_member' not in st.session_state:
        st.session_state.restricted_member = False
        st.session_state.restriction_prompt = ""
    elif st.session_state.restricted_member == True :
        st.session_state.restriction_prompt = f"This request is related to the member name {st.session_state.member_name}"
    else:
        st.session_state.restriction_prompt = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Always display the input box
    if st.session_state.get('phone_number') != 'None Selected':
        user_input = st.chat_input("Ask a question")
    else:
        user_input = None
        st.write("Please select a phone number to get started.")

    # Check if a predefined question has been selected
    predefined_question = st.session_state.pop('active_predefined_question', None)

    # Determine which question to process
    if predefined_question:
        question = predefined_question
    elif user_input:
        question = user_input
    else:
        question = None

   
    if question:
        st.session_state['show_next_best_action'] = False
        chat_history = get_chat_history()
        with st.chat_message("user"):
                st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})
     
        question_summary = question
        if chat_history and st.session_state.summarize_with_chat_history:
            question_summary = summarize_question_with_history(chat_history, question)
            question_summary = question_summary.replace("or supporting documentation", "")

        summary_msg = f"""By evaluating the question and the chat history. Cortex AI is interpreting the question as below \n \n  *'{question_summary}'* \n \n Please 'start over' or refine the question if this intepretation is inaccurate. \n"""
        
        # First, check for a violation
        is_violation = find_violation(question_summary)
        if is_violation:
            with st.chat_message("assistant"):
                response_text = f"""Identified a security violation. Please ensure your question is related to the selected member {st.session_state.member_id} | {st.session_state.member_name}.
                Please refine the question."""
                st.write(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
            # Then determine the question type
            response_txt = find_question_type(question_summary)
            if len(response_txt) > 0:
                intent = response_txt.strip().lower()
            else:
                intent = 'unknown'

            if intent == 'data':
                process_message(prompt=question, question_summary=question_summary, summary_msg=summary_msg)
            else:
                with st.chat_message("assistant"):
                    if intent == 'recordings':
                        msg = "Based on the insights from Cortex AI, this seems to be a question related to call recordings.\nInitiating Call Recordings Search Agent."
                        Agent = "Call Recordings Search Agent"
                    else:
                        msg = "Based on the insights from Cortex AI, this seems to be a Contact Center Knowledge Store based question.\nInitiating Knowledge Store Search Agent."
                        Agent = "Knowledge Store Search Agent"
                    if len(st.session_state.messages) > 1:
                        st.markdown(summary_msg)
                    st.write(msg)
                    message_placeholder = st.empty()
                    question = question_summary.replace("", "")
                    question = f"{question} .{st.session_state.restriction_prompt}"

                    with st.spinner(f"{Agent} thinking..."):
                        response, df_document_urls = complete(question, chat_history, intent)
                        if len(response) > 0:
                            response_text = response
                        else:
                            response_text = "No response received from Cortex AI."

                        message_placeholder.markdown(response_text)
                        if intent == 'recordings':
                            st.markdown("The following call recordings were referred for this answer:")
                        else:
                            st.markdown("The following documents were referred for this answer:")
                        for _, row in df_document_urls.iterrows():
                            relative_path = row['RELATIVE_PATH']
                            url_link = row['URL_LINK']
                            display_url = f"[{relative_path}]({url_link})"
                            st.markdown(display_url)
                st.session_state.messages.append({"role": "assistant", "content": response_text})

    # Move Next Best Action as a checkbox under the chatbox
    if len(st.session_state.messages) > 0:
        show_next_best_action = st.checkbox("Show Next Best Action", key='show_next_best_action')
        if show_next_best_action:
            chat_history = get_chat_history()
            next_best_action = determine_next_best_action(chat_history)
            with st.expander("Next Best Action", expanded=True):
                st.write(next_best_action)

            if 'Unable to determine next best action with information available' not in next_best_action:
                evaluate_action = st.checkbox("Review/Refine AI Generated Draft")
                if evaluate_action:
                    draft_action = generate_draft_action(chat_history, next_best_action)

                    if draft_action:
                        # Parse the draft_action to extract subject and body
                        lines = draft_action.strip().split('\n')
                        subject = ''
                        body_lines = []
                        is_body = False
                        for line in lines:
                            if line.startswith('Subject:'):
                                subject = line[len('Subject:'):].strip()
                            elif line.startswith('Body:'):
                                is_body = True
                            elif is_body:
                                body_lines.append(line)
                        body = '\n'.join(body_lines).strip()

                        if 'trigger_action' not in st.session_state:
                            st.session_state.trigger_action = False

                        with st.expander("Draft Action - AI Generated", expanded=True):
                            edited_subject = st.text_input("Edit the Subject:", value=subject)
                            edited_body = st.text_area("Edit the Email Body:", value=body, height=300)

                            # When the button is clicked, set the flag and store the edited values
                            if st.button("Trigger Action"):
                                st.session_state.trigger_action = True
                                st.session_state.edited_subject = edited_subject
                                st.session_state.edited_body = edited_body

                            # Check if the action should be performed
                            if st.session_state.trigger_action:
                                #with st.spinner("Action in progress..."):
                                send_email(st.session_state.user_email, st.session_state.edited_subject, st.session_state.edited_body)
                                # Reset the flag after action is complete
                                st.session_state.trigger_action = False
                                # Store the edited email if needed
                                st.session_state['edited_subject'] = st.session_state.edited_subject
                                st.session_state['edited_body'] = st.session_state.edited_body 
                            
    if 'active_suggestion' in st.session_state and st.session_state.active_suggestion:
        process_message(
                prompt=st.session_state.active_suggestion,
                question_summary=st.session_state.active_suggestion,
                summary_msg=""
                        )
        st.session_state.active_suggestion = None

if __name__ == "__main__":
    main()