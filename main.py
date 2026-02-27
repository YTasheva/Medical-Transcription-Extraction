# main.py
import os
import json
import pandas as pd
from openai import OpenAI

# Load API key from environment
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Load the data
df = pd.read_csv("data/transcriptions.csv")
print(f"Loaded {len(df)} transcriptions.")
print(df.head())

# Function definitions for OpenAI function calling
function_definition = [
    {
        "type": "function",
        "function": {
            "name": "extract_patient_data",
            "description": "Extract patient age and recommended treatment or procedure from a medical transcription",
            "parameters": {
                "type": "object",
                "properties": {
                    "age": {
                        "type": "integer",
                        "description": "The age of the patient in years"
                    },
                    "recommended_treatment": {
                        "type": "string",
                        "description": "The recommended treatment or procedure mentioned in the transcription"
                    }
                },
                "required": ["age", "recommended_treatment"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "match_icd_code",
            "description": "Match the recommended treatment or procedure with the corresponding ICD-10 code",
            "parameters": {
                "type": "object",
                "properties": {
                    "icd_code": {
                        "type": "string",
                        "description": "The ICD-10 code corresponding to the recommended treatment or procedure"
                    },
                    "icd_description": {
                        "type": "string",
                        "description": "The description of the ICD-10 code"
                    }
                },
                "required": ["icd_code", "icd_description"]
            }
        }
    }
]


