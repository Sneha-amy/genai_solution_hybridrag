import json
import csv
from typing import List, Dict, Any
from openai import OpenAI
import httpx 
import tiktoken 
import os
#from langchain_openai import ChatOpenAI
#from langchain.prompts import ChatPromptTemplate
tiktoken_cache_dir = "./token"
os.environ["TIKTOKEN_CACHE_DIR"] = tiktoken_cache_dir 
client = httpx.Client(verify=False)
CUSTOM_API_KEY = "sk-QibkR-xrOxD2AsMgdaL0Pg"
CUSTOM_MODEL_NAME = "azure_ai/genailab-maas-DeepSeek-V3-0324"
CUSTOM_BASE_URL = "https://genailab.tcs.in/"


class SyntheticDataGenerator:
    def __init__(self, api_key: str, model: str , temperature: float = 0.0):
        self.client = OpenAI(
    base_url=CUSTOM_BASE_URL,
       api_key=CUSTOM_API_KEY,
    http_client=client)
        self.model = model
        self.temperature = temperature


        # Domain presets
        self.domain_context = {
            "healthcare": "Healthcare setting, patient records, doctors, labs, diagnoses, prescriptions.",
            "finance": "Banking, investments, loan profiles, credit scoring, transactions.",
            "retail": "E-commerce, orders, users, shipping info, product metadata.",
            "education": "Students, teachers, courses, assessments, timetables.",
            "generic": "Generic realistic human and business-like data."
        }

        # Multiple prompt templates
        self.templates = {
            "structured": """
You are a professional synthetic data generator. 
Generate {n} records of realistic structured data in {format} format.
Schema:
{schema}

Domain context:
{domain_context}

Noise required: {noise}
Noise guidelines:
- Typos or misspellings
- Missing or null values randomly
- Outlier values
- Random date jitter

Return ONLY valid {format}.
""",

            "unstructured": """
You are a professional unstructured data generator.
Generate a multi-section document batch.

Sections per document: 3 to 7  
Number of documents: {n}

Domain context:
{domain_context}

Document purpose:
{purpose}

Noise required: {noise}

Only return a JSON list:
[
  {{ "id": <id>, "document": "<long multi-section text>" }},
  ...
]
"""
        }

    def _call_llm(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=CUSTOM_MODEL_NAME,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content

    # ---------------------------
    # Structured CSV / JSON
    # ---------------------------
    def generate_structured(self, schema: Dict[str, str], n: int = 10, format: str = "json",
                            noise: bool = False, domain: str = "generic") -> Any:

        prompt = self.templates["structured"].format(
            n=n,
            schema=json.dumps(schema, indent=2),
            domain_context=self.domain_context.get(domain, "generic"),
            noise=str(noise),
            format=format
        )

        # trim the data 
        output = self._call_llm(prompt).strip().removeprefix("```json").removesuffix("```").strip()



        if format.lower() == "json":
            return json.loads(output)

        elif format.lower() == "csv":
            return output  # CSV returned as string

        else:
            raise ValueError("Format must be 'csv' or 'json'.")

    # ---------------------------
    # Unstructured multi-section docs
    # ---------------------------
    def generate_unstructured_batch(self, n: int = 5, noise: bool = False,
                                    domain: str = "generic", purpose: str = "RAG seeding") -> List[Dict[str, Any]]:

        prompt = self.templates["unstructured"].format(
            n=n,
            domain_context=self.domain_context.get(domain, "generic"),
            noise=str(noise),
            purpose=purpose
        )

        output = self._call_llm(prompt).strip().removeprefix("```json").removesuffix("```").strip()
        return json.loads(output)


# -------------------------------
# Example Usage
# -------------------------------
if __name__ == "__main__":
    import os

    generator = SyntheticDataGenerator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="azure/genailab-maas-gpt-35-turbo",
        temperature=0.4
    )

    schema = {
        "name": "string",
        "age": "integer",
        "email": "email",
        "transaction_amount": "float",
        "country": "string"
    }

    data = generator.generate_structured(schema=schema, n=5, format="json", domain="finance", noise=True)
    print("Structured (JSON):", data)

    generator = SyntheticDataGenerator(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="azure/genailab-maas-gpt-4o",
        temperature=0.4
    )

    docs = generator.generate_unstructured_batch(n=3, noise=True, domain="healthcare")
    print("Unstructured:", docs)
