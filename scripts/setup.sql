USE ROLE SYSADMIN;

CREATE OR REPLACE WAREHOUSE PAYERS_CC_WH; --by default, this creates an XS Standard Warehouse
CREATE OR REPLACE DATABASE PAYERS_CC_DB;
CREATE OR REPLACE SCHEMA PAYERS_CC_SCHEMA;

USE WAREHOUSE PAYERS_CC_WH;
USE DATABASE PAYERS_CC_DB;
USE SCHEMA PAYERS_CC_SCHEMA;

CREATE OR REPLACE STAGE RAW_DATA DIRECTORY=(ENABLE=true); --to store data assets
CREATE OR REPLACE STAGE NOTEBOOK DIRECTORY=(ENABLE=true); --to store notebook assets
CREATE OR REPLACE STAGE CHATBOT_APP DIRECTORY=(ENABLE=true); --to store streamlit assets

-- MAKE SURE TO UPLOAD ALL FILES TO THE APPROPRIATE STAGES ABOVE.
-- ONCE THAT IS COMPLETE, UNCOMMENT THE FOLLOWING LINES TO RUN:

-- CREATE OR REPLACE FILE FORMAT CSVFORMAT 
--     SKIP_HEADER = 0 
--     TYPE = 'CSV'
--     FIELD_OPTIONALLY_ENCLOSED_BY = '"';

-- TRUNCATE TABLE IF EXISTS CALL_CENTER_MEMBER_DENORMALIZED;

-- CREATE OR REPLACE TABLE CALL_CENTER_MEMBER_DENORMALIZED (
--     MEMBER_ID NUMBER(38,0),
--     NAME VARCHAR(16777216),
--     DOB DATE,
--     GENDER VARCHAR(16777216),
--     ADDRESS VARCHAR(16777216),
--     MEMBER_PHONE VARCHAR(16777216),
--     PLAN_ID VARCHAR(16777216),
--     PLAN_NAME VARCHAR(16777216),
--     CVG_START_DATE DATE,
--     CVG_END_DATE DATE,
--     PCP VARCHAR(16777216),
--     PCP_PHONE VARCHAR(16777216),
--     PLAN_TYPE VARCHAR(16777216),
--     PREMIUM NUMBER(38,0),
--     SMOKER_IND BOOLEAN,
--     LIFESTYLE_INFO VARCHAR(16777216),
--     CHRONIC_CONDITION VARCHAR(16777216),
--     GRIEVANCE_ID VARCHAR(16777216),
--     GRIEVANCE_DATE DATE,
--     GRIEVANCE_TYPE VARCHAR(16777216),
--     GRIEVANCE_STATUS VARCHAR(16777216),
--     GRIEVANCE_RESOLUTION_DATE DATE,
--     CLAIM_ID VARCHAR(16777216),
--     CLAIM_SERVICE_FROM_DATE DATE,
--     CLAIM_PROVIDER VARCHAR(16777216),
--     CLAIM_SERVICE VARCHAR(16777216),
--     CLAIM_BILL_AMT NUMBER(38,0),
--     CLAIM_ALLOW_AMT NUMBER(38,0),
--     CLAIM_COPAY_AMT NUMBER(38,0),
--     CLAIM_COINSURANCE_AMT NUMBER(38,0),
--     CLAIM_DEDUCTIBLE_AMT NUMBER(38,0),
--     CLAIM_PAID_AMT NUMBER(38,0),
--     CLAIM_STATUS VARCHAR(16777216),
--     CLAIM_PAID_DATE DATE,
--     CLAIM_SERVICE_TO_DATE DATE,
--     CLAIM_SUBMISSION_DATE DATE
-- )

-- COPY INTO CALL_CENTER_MEMBER_DENORMALIZED
-- FROM @RAW_DATA/DATA_PRODUCT/CALL_CENTER_MEMBER_DENORMALIZED.csv
-- FILE_FORMAT = CSV_FORMAT
-- ON_ERROR=CONTINUE
-- FORCE = TRUE;

-- -- load caller intent training data
-- TRUNCATE TABLE IF EXISTS CALLER_INTENT_TRAIN_DATASET;

-- CREATE OR REPLACE TABLE CALLER_INTENT_TRAIN_DATASET (
--     MEMBER_ID VARCHAR(16777216),
--     RECENT_ENROLLMENT_EVENT_IND BOOLEAN,
--     PCP_CHANGE_IND BOOLEAN,
--     ACTIVE_CM_PROGRAM_IND BOOLEAN,
--     CHRONIC_CONDITION_IND BOOLEAN,
--     ACTIVE_GRIEVANCE_IND BOOLEAN,
--     ACTIVE_CLAIM_IND BOOLEAN,
--     POTENTIAL_CALLER_INTENT_CATEGORY VARCHAR(16777216)
-- );

-- COPY INTO CALLER_INTENT_TRAIN_DATASET
-- FROM @RAW_DATA/CALLER_INTENT/CALLER_INTENT_TRAIN_DATASET.csv
-- FILE_FORMAT = CSV_FORMAT
-- ON_ERROR=CONTINUE
-- FORCE = TRUE;

-- -- load caller intent prediction data
-- TRUNCATE TABLE IF EXISTS CALLER_INTENT_PREDICT_DATASET;

-- CREATE OR REPLACE TABLE CALLER_INTENT_PREDICT_DATASET (
--     MEMBER_ID VARCHAR(16777216),
--     RECENT_ENROLLMENT_EVENT_IND BOOLEAN,
--     PCP_CHANGE_IND BOOLEAN,
--     ACTIVE_CM_PROGRAM_IND BOOLEAN,
--     CHRONIC_CONDITION_IND BOOLEAN,
--     ACTIVE_GRIEVANCE_IND BOOLEAN,
--     ACTIVE_CLAIM_IND BOOLEAN
-- );

-- COPY INTO CALLER_INTENT_PREDICT_DATASET
-- FROM @RAW_DATA/CALLER_INTENT/CALLER_INTENT_PREDICT_DATASET.csv
-- FILE_FORMAT = CSV_FORMAT
-- ON_ERROR=CONTINUE
-- FORCE = TRUE;

-- -- make sure staged files can be seen by directory
-- ALTER STAGE RAW_DATA REFRESH;