<h1 align="center">Medical Transcription Extraction</h1>

A Python project that uses the OpenAI API with function calling to automatically extract structured medical data from clinical transcriptions and match them to ICD-10 codes for insurance and billing purposes.

## Overview

Medical professionals write dense, natural-language transcriptions that are difficult to parse manually. This project automates:

- Extracting patient **age** and **recommended treatment** from transcriptions
- Matching treatments to their **ICD-10 codes** and descriptions
- Outputting a clean, structured **pandas DataFrame** ready for downstream use

## Project Structure

```
medical-transcription-extraction/
├── README.md
├── requirements.txt
├── main.py
├── data/
│   └── transcriptions.csv
└── .env
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/medical-transcription-icd.git
cd medical-transcription-icd
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Copy `.env.example` to `.env` and add your key:

```bash
cp .env.example .env
```

Then edit `.env`:

```
OPENAI_API_KEY=your_api_key_here
```

### 4. Add your data

Place your `transcriptions.csv` file in the `data/` folder. It should contain at minimum:

| Column | Description |
|--------|-------------|
| `medical_specialty` | The medical specialty of the transcription |
| `transcription` | The full transcription text |

### 5. Run the project

```bash
python main.py
```

## How It Works

The project uses **OpenAI function calling** with two defined functions:

### `extract_patient_data`
Extracts structured fields from raw transcription text:
- `age` — patient age in years
- `recommended_treatment` — treatment or procedure mentioned

### `match_icd_code`
Maps the extracted treatment to an ICD-10 code:
- `icd_code` — standardized ICD-10 code
- `icd_description` — human-readable description

These are called sequentially for each row, and results are combined into a single DataFrame:

```python
df_structured.head()
```

| age | medical_specialty | recommended_treatment | icd_code | icd_description |
|-----|-------------------|-----------------------|----------|-----------------|
| 45  | Orthopedics       | knee replacement      | Z96.641  | Presence of right artificial knee joint |

## Output

The final output is a pandas DataFrame `df_structured` with the following columns:

- `age`
- `medical_specialty`
- `recommended_treatment`
- `icd_code`
- `icd_description`

## Requirements

- Python 3.8+
- OpenAI API key
- See `requirements.txt` for full dependencies

## Notes

- The model will return `null` for fields not present in the transcription — it is instructed not to assume or hallucinate values
- Error handling is included: rows that fail API calls are saved with `None` values rather than crashing the pipeline
- Uses `gpt-4o-mini` for cost-efficient extraction

## Licence

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Author

- GitHub - [Yuliya Tasheva](https://github.com/YTasheva)

> [https://yuliya-tasheva.co.uk/](#) &nbsp;&middot;&nbsp;
> LinkedIn [@YTasheva](https://www.linkedin.com/in/yuliya-stella-tasheva/) &nbsp;&middot;&nbsp;
> Email [info@yuliya-tasheva.co.uk](#) &nbsp;&middot;&nbsp;

