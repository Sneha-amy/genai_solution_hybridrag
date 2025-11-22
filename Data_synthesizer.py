"""
llm_data_synthesizer.py

A small framework for LLM-based data synthesis using local Ollama.

Features:
- Structured data (JSON/CSV) from arbitrary runtime schema
- Unstructured multi-section documents for RAG seeding
- Domain-aware generation
- Optional noise injection (typos, missing values, outliers, format issues, irrelevant text)
- Multiple prompt templates
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
import csv
import io
import json
import requests


# =========================
# LLM CLIENT (Ollama)
# =========================

@dataclass
class LLMConfig:
    model: str = "llama3"  # change to your preferred Ollama model
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None  # Ollama uses num_predict; we map if provided
    base_url: str = "http://localhost:11434"  # Ollama default


class OllamaClient:
    def __init__(self, config: LLMConfig):
        self.config = config

    def generate(self, prompt: str) -> str:
        """
        Call Ollama /api/generate with non-streaming output.
        """
        url = f"{self.config.base_url}/api/generate"
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "top_p": self.config.top_p,
            },
        }
        if self.config.max_tokens is not None:
            # Ollama naming: num_predict
            payload["options"]["num_predict"] = self.config.max_tokens

        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        # When stream=False, Ollama returns a single JSON with a 'response' field
        return data.get("response", "")


# =========================
# SCHEMA / NOISE CONFIG
# =========================

@dataclass
class FieldSchema:
    name: str
    type: str  # e.g. "string", "int", "float", "date", "boolean"
    description: str = ""
    allowed_values: Optional[List[str]] = None
    min: Optional[float] = None
    max: Optional[float] = None
    pii: bool = False  # Is this field PII (name, email, phone, etc.)?


@dataclass
class NoiseConfig:
    """Controls how much 'mess' to inject into synthetic data."""
    missing_value_prob: float = 0.0
    out_of_range_prob: float = 0.0
    numeric_outlier_prob: float = 0.0
    text_typo_prob: float = 0.0
    format_inconsistency_prob: float = 0.0
    irrelevant_record_prob: float = 0.0

    def is_active(self) -> bool:
        return any([
            self.missing_value_prob,
            self.out_of_range_prob,
            self.numeric_outlier_prob,
            self.text_typo_prob,
            self.format_inconsistency_prob,
            self.irrelevant_record_prob,
        ])


# =========================
# PROMPT BUILDERS
# =========================

StructuredTemplateName = Literal["tabular_generic", "event_log"]
UnstructuredTemplateName = Literal["knowledge_article", "email_thread"]


class PromptBuilder:
    @staticmethod
    def _render_schema(schema: List[FieldSchema]) -> str:
        lines = []
        for field in schema:
            line = f"- {field.name} (type: {field.type}"
            if field.allowed_values:
                line += f", allowed: {', '.join(field.allowed_values)}"
            if field.min is not None or field.max is not None:
                line += f", range: {field.min}..{field.max}"
            if field.pii:
                line += ", PII"
            line += ")"
            if field.description:
                line += f". {field.description}"
            lines.append(line)
        return "\n".join(lines)

    @staticmethod
    def _render_noise(noise: NoiseConfig) -> str:
        if not noise.is_active():
            return (
                "Do NOT intentionally inject noise. "
                "Keep values clean, consistent, and within expected ranges."
            )

        parts = ["Inject realistic noise as follows:"]
        if noise.missing_value_prob > 0:
            parts.append(
                f"- Approximately {int(noise.missing_value_prob * 100)}% of values "
                f"can be missing (null or empty) in random fields."
            )
        if noise.out_of_range_prob > 0:
            parts.append(
                f"- Around {int(noise.out_of_range_prob * 100)}% of numeric or date values "
                "may slightly violate documented ranges."
            )
        if noise.numeric_outlier_prob > 0:
            parts.append(
                f"- Around {int(noise.numeric_outlier_prob * 100)}% of numeric values "
                "can be plausible outliers (extreme but possible)."
            )
        if noise.text_typo_prob > 0:
            parts.append(
                f"- Roughly {int(noise.text_typo_prob * 100)}% of free-text fields "
                "may contain minor spelling errors or extra spaces."
            )
        if noise.format_inconsistency_prob > 0:
            parts.append(
                f"- Around {int(noise.format_inconsistency_prob * 100)}% of values "
                "may use inconsistent date/number formats."
            )
        if noise.irrelevant_record_prob > 0:
            parts.append(
                f"- Up to {int(noise.irrelevant_record_prob * 100)}% of rows/records "
                "may be slightly off-topic or partially incomplete."
            )
        return "\n".join(parts)

    @staticmethod
    def structured_prompt(
        n_records: int,
        schema: List[FieldSchema],
        domain: str,
        noise: NoiseConfig,
        allow_pii: bool,
        template: StructuredTemplateName,
    ) -> str:
        schema_str = PromptBuilder._render_schema(schema)
        noise_str = PromptBuilder._render_noise(noise)

        template_hint = ""
        if template == "tabular_generic":
            template_hint = (
                "Each row should look like a typical record in a realistic "
                f"{domain} dataset. Use consistent semantics across rows."
            )
        elif template == "event_log":
            template_hint = (
                "Each row should represent a single event in a chronological log. "
                "Include realistic timestamps and references between events where applicable."
            )

        pii_hint = (
            "You ARE allowed to synthesize realistic but fake PII like names, emails, "
            "phone numbers and addresses where relevant fields are marked as PII. "
            "Never use real people."
            if allow_pii or any(f.pii for f in schema)
            else "Avoid generating PII; keep data generic and anonymous."
        )

        prompt = f"""
You are a data synthesis assistant.

Goal:
Generate a realistic synthetic dataset for the domain: "{domain}".

Records:
- Number of records: {n_records}
- Schema:
{schema_str}

Style:
- Make the dataset realistic and internally consistent.
- Use varied but plausible values.
- {template_hint}
- {pii_hint}

Noise:
{noise_str}

Output format (IMPORTANT):
- Return ONLY a valid JSON array of {n_records} objects.
- Use DOUBLE quotes for all keys and string values.
- Do NOT include any explanation, comments, markdown, or code fences.
- Do NOT add extra top-level fields.

Now generate the JSON array.
""".strip()
        return prompt

    @staticmethod
    def unstructured_prompt(
        n_docs: int,
        domain: str,
        noise: NoiseConfig,
        allow_pii: bool,
        template: UnstructuredTemplateName,
        batch_index: int,
        sections_per_doc: int = 3,
    ) -> str:
        noise_str = PromptBuilder._render_noise(noise)

        if template == "knowledge_article":
            template_hint = f"""
Create multi-section knowledge articles in the {domain} domain.
Each document:
- Has a clear title (H1 markdown heading).
- Has {sections_per_doc} or more sections, each as an H2 or H3 heading.
- Contains paragraphs, occasional bullet lists, and very small tables if relevant.
- Represents information that would be useful as a document in a RAG corpus.
"""
        elif template == "email_thread":
            template_hint = f"""
Create realistic email threads for the {domain} domain.
Each document:
- Starts with a subject line.
- Contains a series of emails (From, To, Date, Body).
- Uses varying writing styles but consistent context per thread.
"""

        pii_hint = (
            "You ARE allowed to synthesize realistic but fake PII like names, emails, "
            "phone numbers and addresses to make the documents realistic. "
            "Make sure it is clearly synthetic and not tied to real people."
            if allow_pii
            else "Avoid generating PII; use generic names or roles if identities are needed."
        )

        prompt = f"""
You are a data synthesis assistant specialized in generating unstructured corpora.

Goal:
Generate {n_docs} unstructured documents to be used as a RAG corpus
in the domain "{domain}". This is batch index {batch_index}.

{template_hint}

General style:
- Documents should be diverse but realistic.
- Within each document, keep the story/context internally consistent.
- Use markdown for headings, lists, and tables when appropriate.
- Include enough detail and nuance to be useful for question-answering.
- Optionally, you may add a small 'Metadata' section at the end of each doc
  listing things like category, tags, or approximate date range.

PII policy:
{pii_hint}

Noise:
{noise_str}

Output format (IMPORTANT):
- Return a JSON array of {n_docs} objects.
- Each object MUST have:
  - "doc_id": a unique string id (e.g. "doc-{batch_index}-{i}")
  - "title": short title
  - "body": the full markdown content of the document
- Use DOUBLE quotes for all keys and string values.
- Do NOT include any explanation, comments, markdown fences, or extra fields.

Now generate the JSON array.
""".strip()
        return prompt


# =========================
# MAIN SYNTHESIZER
# =========================

@dataclass
class LLMDataSynthesizer:
    llm_client: OllamaClient
    default_domain: str = "generic"
    default_noise: NoiseConfig = field(default_factory=NoiseConfig)

    # -------- Structured --------

    def generate_structured_json(
        self,
        n_records: int,
        schema: List[FieldSchema],
        domain: Optional[str] = None,
        noise: Optional[NoiseConfig] = None,
        allow_pii: bool = True,
        template: StructuredTemplateName = "tabular_generic",
    ) -> List[Dict[str, Any]]:
        domain = domain or self.default_domain
        noise = noise or self.default_noise

        prompt = PromptBuilder.structured_prompt(
            n_records=n_records,
            schema=schema,
            domain=domain,
            noise=noise,
            allow_pii=allow_pii,
            template=template,
        )

        raw = self.llm_client.generate(prompt)

        # Attempt to parse the result as JSON; if it fails, try to salvage.
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Naive salvage: try to trim non-JSON content
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                snippet = raw[start : end + 1]
                data = json.loads(snippet)
            else:
                raise

        if not isinstance(data, list):
            raise ValueError("Expected a JSON array from LLM for structured data.")

        # Optionally: light validation – ensure fields exist
        field_names = {f.name for f in schema}
        for i, row in enumerate(data):
            if not isinstance(row, dict):
                raise ValueError(f"Record {i} is not an object: {row!r}")
            # You could enforce exact field set, but here we only ensure required ones exist
            missing = field_names - row.keys()
            if missing:
                # You might choose to fill missing with None instead of error
                for m in missing:
                    row[m] = None

        return data

    def generate_structured_csv(
        self,
        n_records: int,
        schema: List[FieldSchema],
        domain: Optional[str] = None,
        noise: Optional[NoiseConfig] = None,
        allow_pii: bool = True,
        template: StructuredTemplateName = "tabular_generic",
    ) -> str:
        records = self.generate_structured_json(
            n_records=n_records,
            schema=schema,
            domain=domain,
            noise=noise,
            allow_pii=allow_pii,
            template=template,
        )

        if not records:
            return ""

        fieldnames = list(records[0].keys())
        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow(row)
        return buf.getvalue()

    # -------- Unstructured --------

    def generate_unstructured_corpus(
        self,
        n_docs: int,
        domain: Optional[str] = None,
        noise: Optional[NoiseConfig] = None,
        allow_pii: bool = True,
        template: UnstructuredTemplateName = "knowledge_article",
        batch_index: int = 0,
        sections_per_doc: int = 3,
    ) -> List[Dict[str, str]]:
        """
        Generate a batch of unstructured documents for RAG seeding.

        Returns a list of dicts: { "doc_id": str, "title": str, "body": str }
        """
        domain = domain or self.default_domain
        noise = noise or self.default_noise

        prompt = PromptBuilder.unstructured_prompt(
            n_docs=n_docs,
            domain=domain,
            noise=noise,
            allow_pii=allow_pii,
            template=template,
            batch_index=batch_index,
            sections_per_doc=sections_per_doc,
        )

        raw = self.llm_client.generate(prompt)

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                snippet = raw[start : end + 1]
                data = json.loads(snippet)
            else:
                raise

        if not isinstance(data, list):
            raise ValueError("Expected a JSON array from LLM for unstructured corpus.")

        for i, doc in enumerate(data):
            if not isinstance(doc, dict):
                raise ValueError(f"Document {i} is not an object: {doc!r}")
            for key in ("doc_id", "title", "body"):
                if key not in doc:
                    raise ValueError(
                        f"Document {i} is missing required key '{key}': {doc!r}"
                    )

        return data


# =========================
# EXAMPLE USAGE
# =========================

def example_a_structured():
    """
    Example A:
    Generate synthetic loan applications (structured JSON/CSV) for a loan officer app.
    """
    llm_cfg = LLMConfig(model="llama3")
    client = OllamaClient(llm_cfg)
    synthesizer = LLMDataSynthesizer(client, default_domain="consumer lending")

    schema = [
        FieldSchema(
            name="application_id",
            type="string",
            description="Unique loan application ID",
        ),
        FieldSchema(
            name="customer_name",
            type="string",
            description="Full name of the applicant",
            pii=True,
        ),
        FieldSchema
