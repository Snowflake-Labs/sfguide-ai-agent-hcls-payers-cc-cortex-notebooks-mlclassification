import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import Complete
from snowflake.core import Root
import _snowflake
import json
import os
from snowflake.snowpark import Session
import pandas as pd
import pypdfium2 as pdfium

session = get_active_session()
root = Root(session)

# Set pandas option to display all column content
pd.set_option("max_colwidth", None)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Constants
DATABASE = session.get_current_database()
SCHEMA = session.get_current_schema()
STAGE = "RAW_DATA"
FILE = "DATA_PRODUCT/Call_Center_Member_Denormalized.yaml"
API_ENDPOINT = "/api/v2/cortex/agent:run"
API_TIMEOUT = 50000  # in milliseconds
MAX_DATAFRAME_ROWS = 1000

NUM_CHUNKS = 1 # number of chunks to retrieve from cortex search (FAQ docs)
NUM_TRANSCRIPTS = 2 # number of transcripts to retrieve from cortex search (Call transcripts)
SLIDE_WINDOW = 4 # window of chat history to consider for each subsequent question

def config_options():
    """
    Create sidebar configs for Streamlit app 
    """
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
        st.session_state['agent_model'] = 'mistral-large2'

        # Reset checkboxes and toggle to default values
        st.session_state['debug'] = False
        st.session_state['debug_prompt'] = False
   
        # Reset other relevant session state variables
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
        st.session_state.pop('user_email', None)

        st.sidebar.success("Conversation and selections have been reset.")

    # Initialize session state variables with default values if not already set
    # st.sidebar.selectbox(
    #     'Select your Agent model :',
    #     ("mistral-large2", "claude-3-5-sonnet", "llama3.3-70b"),
    #     key='agent_model'
    # )
    st.session_state.agent_model = 'mistral-large2'
    # st.sidebar.selectbox(
    #     'Select your Cortex Complete Mode:',
    #     ('SQL', 'API'),
    #     key='cortex_complete_type'
    # )
    st.session_state.cortex_complete_type = 'API'
    st.sidebar.text_input(
        "Enter your email:", 
        "",
        key='user_email')

    st.session_state.debug_payload_response = st.sidebar.toggle("Activate Debug Mode")

    st.session_state['debug'] = st.session_state.debug_payload_response
    st.session_state['debug_prompt'] = st.session_state.debug_payload_response

    instruction_txt = """ 
    """

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
    st.session_state.setdefault('response_instruction', instruction_txt)
    st.session_state.setdefault('show_first_tool_use_only', True)
    st.session_state.setdefault('api_history', [])
    # Add any additional setdefaults as necessary

    return clear_conversation

def init_messages(clear_conversation):
    """
    Initialize a new session state or clear existing session state
    """
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
        st.session_state.api_history = []
  
def download_file_from_stage(relative_path: str) -> str:
    """
    Download a file (PDF, audio, etc.) from a Snowflake stage to a local temp directory.
    Returns the local file path.
    """
    local_dir = "/tmp/"  # or any temp directory
    relative_path = relative_path.replace(r'call_recordings/', 'CALL_RECORDINGS/')
    session.file.get(f"@{DATABASE}.{SCHEMA}.{STAGE}/{relative_path}", local_dir)
    local_file_path = os.path.join(local_dir, os.path.basename(relative_path))
    return local_file_path

@st.cache_data
def get_presigned_url(relative_path: str, expire_seconds: int = 360) -> str:
    """
    Returns a presigned URL to an object in the Snowflake stage.
    expire_seconds controls how long (in seconds) the link remains valid.
    """
    if "DICOM" in relative_path:
        STAGE = "OUTPUT_STAGE"
    else:
     STAGE = "RAW_DATA"
    
    sql = f"""
        SELECT GET_PRESIGNED_URL(@{DATABASE}.{SCHEMA}.{STAGE}, '{relative_path}', {expire_seconds}) AS URL_LINK
    """
    res = session.sql(sql).collect()
    return res[0].URL_LINK

def get_pdf(local_pdf_path: str) -> pdfium.PdfDocument:
    """
    Cache the loaded PDF to avoid re-downloading and re-parsing on every run.
    """
    return pdfium.PdfDocument(local_pdf_path)

def display_file_with_scrollbar(relative_path: str, file_type: str = "pdf", unique_key: str = "", citation_id: str = ""):
    """
    Display a file preview (PDF or audio) inside an expander with a scrollbar.
    """
  
    with st.expander(f"Citation:{citation_id} - {os.path.basename(relative_path)}", expanded=False):
        if file_type == "pdf":
            # PDF rendering logic
            local_file_path = download_file_from_stage(relative_path)
            if not os.path.exists(local_file_path):
                st.error(f"Could not find the {file_type} at {local_file_path}.")
                return

            pdf_doc = get_pdf(local_file_path)
            total_pages = len(pdf_doc)

            # Example approach: show first 2 pages
            page_numbers = (1, min(2, total_pages))
            start_page, end_page = page_numbers

            pdf_container = st.container(height = 300)
            for page_number in range(start_page - 1, end_page):
                page = pdf_doc[page_number]
                bitmap = page.render(scale=1.0)
                pil_image = bitmap.to_pil()
                with pdf_container:
                    st.image(pil_image, use_container_width=True)
        elif file_type == "jpg":
            # Instead of downloading file, get presigned URL directly
            presigned_url = get_presigned_url(relative_path, expire_seconds=600)  # e.g. 10 mins
            st.image(presigned_url)        
        elif file_type == "audio":
            # Instead of downloading file, get presigned URL directly
            relative_path = relative_path.replace(r'call_recordings/', 'CALL_RECORDINGS/')  # new line
            st.write(relative_path)
            presigned_url = get_presigned_url(relative_path, expire_seconds=600)  # e.g. 10 mins
        else:
            st.warning(f"File type '{file_type}' not supported for preview.")

def get_chat_history():
    """
    Returns the last SLIDE_WINDOW messages of conversation from session state
    if use_chat_history is enabled; otherwise an empty list.
    """
    if st.session_state.use_chat_history:
        if len(st.session_state.messages) > 0:
            start_index = max(0, len(st.session_state.messages) - SLIDE_WINDOW)
            return st.session_state.messages[start_index:]
        else:
            return ["No prior chat"]
    return []


def execute_cortex_complete(prompt):
    """
    Execute Cortex Complete for prompts
    """
    if st.session_state.cortex_complete_type == 'API':
       response_txt = execute_cortex_complete_api(f"""{prompt}.{st.session_state.restriction_prompt}""")
    else:
        response_txt = execute_cortex_complete_sql(f"""{prompt}.{st.session_state.restriction_prompt}""")
    return response_txt

def execute_cortex_complete_sql(prompt):
    """
    Execute Cortex Complete using the SQL API
    """
    cmd = "SELECT snowflake.cortex.complete(?, ?) AS response"
    df_response = session.sql(cmd, params=[st.session_state.agent_model, prompt]).collect()
    response_txt = df_response[0].RESPONSE
    return response_txt

def execute_cortex_complete_api(prompt):    
    """
    Execute Cortex Complete using the REST API
    """
    response_txt = Complete(
                    st.session_state.agent_model,
                    prompt,
                    session=session
                    )
    return response_txt

def summarize_question_with_history(chat_history, question):
    """
    Create and execute prompt to summarize chat history
    """


   
    
    prompt = f"""
        You are a chatbot expert. Refer the latest question received by the chatbot, evaluate this in context of the Chat History found below. 
        Now share a refined query which captures the full meaning of the question being asked. 

        If thee question is NOT one of the below two
            1) Refers a check across all members
            2) Or mentions a member name other than {st.session_state.member_name} 
        Then use  {st.session_state.member_name} as the member related to this question. 
        
        If the question appears to be a stand alone question ignore all previous interactions or chat history and focus solely on the question. 
        If it seem to be connected to the prior chat history, only then use the chat history.

        Please use the question as the prominent input and the Chat history as a support input when creating the refined question.
        Please ensure no relevant information in the latest question is lost.
        
        Answer with only the query. Do not add any explanation.

        Chat History: {chat_history}
        Question: {question}
    """

    if st.session_state.debug:
        st.text("Prompt being used to create the summarized question")
        st.caption(prompt)
    
    summary = execute_cortex_complete(prompt)

    return summary

@st.cache_data
def bot_retrieve_sql_results(sql):
    """
    Execute the SQL in Snowflake, returning a pandas DataFrame.
    """
    return session.sql(sql).limit(MAX_DATAFRAME_ROWS).to_pandas()

def generate_payload(prompt):
    """
    Generates the payload object for the agent run API call.
    This object includes the model, the user prompt, and the tool specs.
    """
    return {
        "response-instruction": st.session_state['response_instruction'],
        "model": st.session_state['agent_model'],
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ],
        "tools": [
            {
                "tool_spec": {
                    "type": "cortex_analyst_text_to_sql",
                    "name": "Contact Center Analyst"
                }
            },
            {
                "tool_spec": {
                    "type": "cortex_search",
                    "name": "Knowledge_Store_Search"
                }
            },
            {
                "tool_spec": {
                    "type": "cortex_search",
                    "name": "Member_Call_Recordings_Search"
                },
            },
       
        ],
        "tool_resources": {
            "Contact Center Analyst": {
                "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}"
            },
            "Knowledge_Store_Search": {
                "name": f"{DATABASE}.{SCHEMA}.CALL_CENTER_FAQ_SEARCH",
                "max_results": NUM_CHUNKS
            },
            "Member_Call_Recordings_Search": {
                "name": f"{DATABASE}.{SCHEMA}.CALL_CENTER_RECORDING_SEARCH",
                "max_results": NUM_TRANSCRIPTS
            }
       
        }
    }
def process_prompt(prompt):
    """
    Sends the prompt to the Cortex agent run endpoint and streams the response.
    """
    payload = generate_payload(prompt)
    if st.session_state.debug_payload_response:
        st.write(payload)
    st.session_state.api_history.append({"Request": payload})

    try:
        resp = _snowflake.send_snow_api_request(
            "POST",  # method
            API_ENDPOINT,  # path
            {},  # headers
            {},  # params
            payload,  # body
            None,  # request_guid
            API_TIMEOUT,  # timeout in milliseconds,
        )
        
        if resp["status"] != 200:
            st.error(f"❌ HTTP Error: {resp['status']} - {resp.get('reason', 'Unknown reason')}")
            st.error(f"Response details: {resp}")
            return None
        
        try:
            response_content = json.loads(resp["content"])
            if st.session_state.get("debug_mode", False):
             st.write(response_content)
        except json.JSONDecodeError:
            st.error("❌ Failed to parse API response. The server may have returned an invalid JSON format.")
            st.error(f"Raw response: {resp}")
            return None
        
        st.session_state.api_history.append({"Response": response_content})    
        return response_content
            
    except Exception as e:
        st.error(f"Error making request: {str(e)}")
        return None

import re

def parse_file_references_new(text: str):
    """
    Detect any PDF, MP3, or image file references of the form:
      - Document Name :DATA_PRODUCT_POV/PROVIDER_CONTRACTS/...pdf
      - Audio File Name :DATA_PRODUCT_POV/CALL_RECORDINGS/...mp3
      - Document Name: DICOM_JPEGS/...jpg (or .jpeg)

    Example:
      "Document Name: DICOM_JPEGS/DICOM1.dcm.jpg: This Medical Image..."

    Returns:
       cleaned_text (str): Response text with references removed.
       references (list of tuples): 
           [(file_path, file_type, full_string), ...]
             file_path = matched path (like "DICOM_JPEGS/DICOM1.dcm.jpg")
             file_type = one of {'pdf', 'audio', 'jpg', 'jpeg'}
             full_string = exactly what was removed from the text
    """

    # Regex explanation:
    #  - (?P<full> ... ): capture the entire match as "full" so we can remove it directly from the text.
    #  - (?P<label>Document Name|Audio File Name): captures which label was used.
    #  - \s*:\s*: matches colon with optional whitespace around it.
    #  - (?P<path>[^\s:]+\.(?:pdf|mp3|jpg|jpeg)): captures a path that has NO whitespace or colon until the dot-extension
    #    and ends in pdf|mp3|jpg|jpeg
    #  - (?:\:?) : optionally match a trailing colon (as in "....jpg:")
    pattern = (
        r"(?P<full>"                  # Start capturing entire matched string
        r"(?P<label>Document Name|Audio File Name)\s*:\s*" 
        r"(?P<path>[^\s:]+\.(?:pdf|mp3|jpg|jpeg))"  # capture the path
        r"(?:\:?)"                    # optional trailing colon
        r")"
    )

    references = []
    # Use finditer to get match objects; each match has groups 'full', 'label', 'path'
    for match in re.finditer(pattern, text):
        full_string = match["full"]   # e.g., "Document Name: DICOM_JPEGS/DICOM1.dcm.jpg:"
        label       = match["label"]  # "Document Name" or "Audio File Name"
        file_path   = match["path"]   # e.g., "DICOM_JPEGS/DICOM1.dcm.jpg"

        # Infer file_type
        if file_path.endswith(".pdf"):
            file_type = "pdf"
        elif file_path.endswith(".mp3"):
            file_type = "audio"  # or "mp3" if you prefer
        elif file_path.endswith(".jpeg"):
            file_type = "jpeg"
        elif file_path.endswith(".jpg"):
            file_type = "jpg"
        else:
            file_type = "audio"  # Catch-all

        references.append((file_path, file_type, full_string))

    # Remove each entire matched reference from the text
    cleaned_text = text
    for _, _, full_string in references:
        cleaned_text = cleaned_text.replace(full_string, "")

    # Return the cleaned text and the references found
    return cleaned_text.strip(), references

def format_bot_message(stream_data, question_summary):
    """
    Parse the streaming data and display partial messages in real-time.
    Also append the final message details to st.session_state.messages.
    """
    if not stream_data:
        return

    bot_text_message = ""  # Accumulate partial text
    search_results = []
    sql_queries = []
    suggestions = []
    df_sql = pd.DataFrame([])
    answer_english = []
    error_info = None

    # Display the entire assistant reply as a chat message
    with st.chat_message("assistant"):
        # If you want to show the agent's refined question:
        summary_msg = (
            f"**Refined Query:** {question_summary}\n\n"
        )
  
        #st.write(f"""Cortex Agent is interepreting the question as - '{question_summary}'""")

        # Flag that tracks if we've already shown the tool-use message:
        tool_use_displayed = False

        for item in stream_data:
            if item["event"] == "error":
                # If an error occurred mid-stream
                error_info = item["data"]
                break
            elif item["event"] == "done":
                break

            if "data" in item and "delta" in item["data"]:
                chunk_content = item["data"]["delta"].get("content", [])
                for content_piece in chunk_content:
                    ctype = content_piece["type"]

                    
                    if ctype != "text":
                        if st.session_state.debug_payload_response:
                            st.write(content_piece)
                    
                    if ctype == "tool_use":
                        tool_name = content_piece.get("tool_use", {}).get("name", "Unknown Tool")
                        if st.session_state.show_first_tool_use_only:
                            # Only display the first time a tool is used:
                            if not tool_use_displayed:
                                st.markdown(f":mag_right: :blue[**Cortex Agent**] is using the tool: `{tool_name}`")
                                tool_use_displayed = True
                            # If we've already displayed once, skip it
                        else:
                            # Always show each tool use
                            st.markdown(f":mag_right: :blue[**Cortex Agent**] is using the tool: `{tool_name}`")
                    
                    elif ctype == "tool_results":
                        tool_json = content_piece.get("tool_results", {}).get("content", [{}])[0].get("json", {})

                        # Grab interesting parts
                        _search = tool_json.get("searchResults", [])
                        _sql = tool_json.get("sql", [])
                        _suggestions = tool_json.get("suggestions", [])
                        _assistant_text = tool_json.get("text", "")

                        if _assistant_text:
                            st.markdown(_assistant_text)

                        search_results.extend(_search)
                        sql_queries.extend(_sql)
                        suggestions.extend(_suggestions)

                        # Execute or display SQL if found:
                        if _sql:
                            try:
                                # NOTE: If the agent can generate multiple SQL statements
                                # for example, you might want to parse them carefully.
                                df_sql = bot_retrieve_sql_results(_sql[:-1])  
                                #st.dataframe(df_sql)
                                if not df_sql.empty:
                                    prompt_sql_to_english = create_prompt_summarize_cortex_analyst_results(question_summary, df_sql, _sql[:-1])
                                    answer_english = execute_cortex_complete(prompt_sql_to_english)
                                    st.write(answer_english)                            
                            except Exception as ex:
                                st.error(f"SQL Execution Error: {ex}")

                            with st.expander("Generated SQL Query"):
                                st.code(_sql, language="sql")

                        # Suggestions
                        if _suggestions:
                            with st.expander("Suggested Follow-ups"):
                                for s in _suggestions:
                                    st.markdown(f"- **{s}**")

                    elif ctype == "text":
                        # Plain text from the assistant
                        bot_text_message += content_piece["text"]

        # Display aggregated final text
        if bot_text_message.strip():
            st.write(bot_text_message)

            # ---- Citation Parsing Logic ----
            citation_pattern = r'【†(\d+)†】'
            cited_ids_in_text = re.findall(citation_pattern, bot_text_message)
            cited_ids_in_text = set(int(x) for x in cited_ids_in_text)
            cited_docs = [
                doc for doc in search_results
                if doc.get('source_id') in cited_ids_in_text
            ]
            if cited_docs:
                st.markdown("---")
                st.markdown("##### Citation References")
                for doc in cited_docs:
                    doc_id = doc["source_id"]
                    doc_text = doc["text"]

                    cleaned_text, file_refs = parse_file_references_new(doc_text)
                    

                    # Then display each file using the existing helper
                    for ref_path, ref_type, _ in file_refs:
                        if ref_type == "pdf":
                            display_file_with_scrollbar(ref_path, file_type="pdf", unique_key=ref_path,citation_id = doc_id)
                        elif ref_type == "jpg":
                            display_file_with_scrollbar(ref_path, file_type="jpg", unique_key=ref_path,citation_id = doc_id)  
                        else:
                            # "audio"
                            display_file_with_scrollbar(ref_path, file_type="audio", unique_key=ref_path,citation_id = doc_id)

                    
                    # with st.expander(f"Citation 【†{doc_id}†】"):
                    #     st.write(doc_text)
            # ---- End Citation Parsing ----

        if error_info:
            error_code = error_info.get('code', 'Unknown Code')
            error_msg = error_info.get('message', 'Unknown Error')
            st.error(f"**Error Code:** {error_code}\n\n**Message:** {error_msg}")

    #convert any dataframe responses to english text


    
    # Store entire chunked response as a single message
    st.session_state.messages.append({
        "role": "assistant",
        "text": bot_text_message,
       # "searchResults": search_results,
        #"sql": sql_queries,
        "df_sql": df_sql,
        "answer_english": answer_english,
        "suggestions": suggestions,
        "type": "error" if error_info else "assistant"
    })

    
def create_prompt_summarize_cortex_analyst_results(myquestion, df, sql):
    """
    Create prompt to summarize Cortex Analyst results in natural language
    """
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
        st.text(f"Prompt being passed to {st.session_state.agent_model}")
        st.caption(prompt)

    return prompt


def find_violation(myquestion):
    """
    Run execute_cortex_complete() on a prompt to detect a violation whether 
    a question contains any member names other than the one selected
    """
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


@st.cache_data(show_spinner=False)
def get_member_details(phone_number):
    """
    Run a query to get member details and extract values
    """
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
    """
    Update session state if phone number is changed
    """
    # Force the 'Limit question only on selected member' toggle to be True
    st.session_state['restricted_member_toggle'] = True
    
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
    """
    Display member info and sample questions in Streamlit app
    """
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
            "Share all available information on this submitted claim"
        ],
        '946-081-0513': [
            "How can the member find out more details about the wellness programs offered by Enterprise_Nxt?",
            "Member wants to know how to find the member forms online?",
            "What is the plan and coverage information of this member?",
            #"Give me the member information on Jessica Mills"
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

        st.sidebar.toggle('Limit question only on selected member',key ='restricted_member_toggle')
        
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
    """
    Create a prompt and run execute_cortex_complete() to determine the next best action 
    """
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
    """
    Create a prompt and run execute_cortex_complete() to craft an email 
    """
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
    """
    Send an email using SYSTEM$SEND_EMAIL
    """
    st.write(f"Sending an email to: {recipient_email}")
    cmd = """CALL SYSTEM$SEND_EMAIL(
             'my_email_int',
             ?,
             ?,
            ?
            );"""
    session.sql(cmd, params=[recipient_email, subject, body]).collect()
    try:
        session.sql(cmd, params=[recipient_email, subject, body]).collect()
        st.success("Email sent successfully!")
    except:
        st.write("Please enter a valid email in the sidebar configs.")

def main():
    st.set_page_config(layout="wide")
    st.title(f"Payer Contact Center Agent :robot_face:")
    st.subheader(f"Powered by Snowflake Cortex :snowflake::snowflake:")

    clear_conversation = config_options()
    init_messages(clear_conversation)
    display_member_info()

    if 'restricted_member' not in st.session_state or 'restriction_prompt' not in st.session_state :
        st.session_state.restricted_member = False
        st.session_state.restriction_prompt = ""
    elif st.session_state.restricted_member == True :
        st.session_state.restriction_prompt = f"This request is related to the member name {st.session_state.member_name}"
    else:
        st.session_state.restriction_prompt = ""

    # Display existing conversation
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(msg["text"])
        elif msg["role"] == "assistant":
            with st.chat_message("assistant"):
                if msg.get("text"):
                    st.markdown(msg["text"])
                if msg.get("df_sql") is not None and not msg["df_sql"].empty:
                    #st.dataframe(msg["df_sql"])
                    st.write(msg["answer_english"])
                if msg.get("suggestions"):
                    with st.expander("Suggested Follow-ups"):
                        for suggestion in msg["suggestions"]:
                            st.markdown(f"- **{suggestion}**")
                if msg.get("type") == "error":
                    st.error(msg["text"])

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
        st.session_state.messages.append({"role": "user", "text": question})
     
        question_summary = question
        #st.write(f"Checkpoint 1 - chat_history = {chat_history} . st.session_state.summarize_with_chat_history = {st.session_state.summarize_with_chat_history}")
        if chat_history and st.session_state.summarize_with_chat_history:
            question_summary = summarize_question_with_history(chat_history, question)
            question_summary = question_summary.replace("or supporting documentation", "")

        summary_msg = f"""By evaluating the question and the chat history. Cortex AI is interpreting the question as below \n \n  *'{question_summary}'* \n \n Please 'start over' or refine the question if this intepretation is inaccurate. \n"""
        st.write(summary_msg)
        # First, check for a violation
        is_violation = find_violation(question_summary)
        if is_violation:
            with st.chat_message("assistant"):
                response_text = f"""Identified a security violation. Please ensure your question is related to the selected member {st.session_state.member_id} | {st.session_state.member_name}.
                Please refine the question."""
                st.write(response_text)
            #st.session_state.messages.append({"role": "assistant", "content": response_text})
        else:
           
            with st.spinner(":blue[**Cortex**] :blue[**Agent**] is evaluating your question..."):
                st.empty()
                #response_data = process_prompt(f"{question_summary}.{st.session_state.restriction_prompt}")
                response_data = process_prompt(f"{question_summary}")
              
                if response_data is not None:
                    format_bot_message(response_data, question_summary)
            
     
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

                            if st.button("Trigger Action"):
                                user_email = st.session_state.get('user_email', '').strip()
                                if not user_email or '@' not in user_email:
                                    st.warning("Please enter a valid email address in the sidebar before triggering the email.")
                                    st.session_state.trigger_action = False
                                else:
                                    st.session_state.trigger_action = True
                                    st.session_state.edited_subject = edited_subject
                                    st.session_state.edited_body = edited_body

                            # Check if the action should be performed
                            if st.session_state.get('trigger_action', False):
                                send_email(st.session_state.user_email, st.session_state.edited_subject, st.session_state.edited_body)
                                st.session_state.trigger_action = False
                                st.session_state['edited_subject'] = st.session_state.edited_subject
                                st.session_state['edited_body'] = st.session_state.edited_body 

if __name__ == "__main__":
    main()
