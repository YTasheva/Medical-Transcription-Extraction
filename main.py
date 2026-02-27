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


def extract_patient_data(transcription, medical_specialty):
    """Extract age and recommended treatment from a transcription."""
    messages = [
        {
            "role": "system",
            "content": (
                "You are a medical data extraction assistant. "
                "Extract structured information from medical transcriptions accurately. "
                "Do not make assumptions about values not present in the text."
            )
        },
        {
            "role": "user",
            "content": f"Extract the patient age and recommended treatment or procedure from this transcription:\n\n{transcription}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[function_definition[0]],
        tool_choice={"type": "function", "function": {"name": "extract_patient_data"}}
    )

    patient_data = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
    patient_data["medical_specialty"] = medical_specialty
    return patient_data


def match_icd_code(recommended_treatment):
    """Match a treatment to its ICD-10 code."""
    messages = [
        {
            "role": "system",
            "content": "You are a medical coding assistant. Match treatments and procedures with their correct ICD-10 codes."
        },
        {
            "role": "user",
            "content": f"Provide the ICD-10 code and description for this treatment or procedure: {recommended_treatment}"
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[function_definition[1]],
        tool_choice={"type": "function", "function": {"name": "match_icd_code"}}
    )

    return json.loads(response.choices[0].message.tool_calls[0].function.arguments)


# Process all rows
results = []

for i, row in df.iterrows():
    print(f"Processing row {i+1}/{len(df)}...")
    try:
        patient_data = extract_patient_data(row["transcription"], row["medical_specialty"])
        icd_data = match_icd_code(patient_data["recommended_treatment"])

        combined = {
            "age": patient_data.get("age"),
            "medical_specialty": patient_data.get("medical_specialty"),
            "recommended_treatment": patient_data.get("recommended_treatment"),
            "icd_code": icd_data.get("icd_code"),
            "icd_description": icd_data.get("icd_description")
        }
        results.append(combined)

    except Exception as e:
        print(f"Error processing row {i}: {e}")
        results.append({
            "age": None,
            "medical_specialty": row["medical_specialty"],
            "recommended_treatment": None,
            "icd_code": None,
            "icd_description": None
        })

# Save results to DataFrame and CSV
df_structured = pd.DataFrame(results)
print("\nStructured output:")
print(df_structured.head())

df_structured.to_csv("data/structured_output.csv", index=False)
print("\nSaved to data/structured_output.csv")
