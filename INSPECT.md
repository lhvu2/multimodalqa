# 🔍 MultiModalQA Benchmark — Inspection Notes

> **Session summary:** Deep-dive into the MultiModalQA benchmark — data formats, a concrete example, table sizes, and a lakehouse architecture proposal.

---

## 📚 Table of Contents

1. [What is MultiModalQA?](#1-what-is-multimodalqa)
2. [Concrete Example — Input & Output](#2-concrete-example--input--output)
3. [Table Sizes & Statistics](#3-table-sizes--statistics)
4. [Lakehouse Architecture Proposal](#4-lakehouse-architecture-proposal)

---

## 1. What is MultiModalQA?

**MultiModalQA** is a challenging QA benchmark from AllenAI (ICLR 2021) that requires **joint reasoning across text, tables, and images**.

| Property | Value |
|----------|-------|
| 📊 Total examples | **29,918** |
| 📝 Modalities | Text paragraphs, Wikipedia tables, Images |
| 🔗 Source | Wikipedia articles |
| 🏆 Paper | [ICLR 2021 — Talmor et al.](https://openreview.net/pdf?id=ee6W5UgQLa) |

### Question Types

The benchmark defines **16 question types** across single-hop and multi-hop categories:

```
Single-hop:   TextQ | TableQ | ImageQ | ImageListQ

Multi-hop (examples):
  Compose(TableQ, ImageListQ)          ← table lookup → image identification
  Compose(TextQ, TableQ)               ← text reading → table lookup
  Compose(ImageQ, TextQ)               ← image VQA → text reading
  Intersect(TableQ, TextQ)             ← intersection of table & text answers
  Compare(Compose(TableQ,ImageQ),TableQ)  ← comparison across modalities
  ... (16 types total)
```

### Input Files (all `.jsonl.gz`)

| File | Content |
|------|---------|
| `MultiModalQA_train/dev/test.jsonl.gz` | Questions, answers, metadata |
| `tables.jsonl.gz` | ~6,000 Wikipedia tables |
| `texts.jsonl.gz` | Wikipedia text paragraphs |
| `images.jsonl.gz` | Image metadata (title, path) |
| `images.zip` | Actual JPEG image files |

---

## 2. Concrete Example — Input & Output

### 🟡 Question

> **"What was the net worth (USD bn) of the LGBT billionaire who is completely bald and wears thick glasses?"**

- **Question ID:** `5454c14ad01e722c2619b66778daa98b`
- **Type:** `Compose(TableQ, ImageListQ)`
- **Modalities required:** `image` + `table`

---

### 📥 Input Data

#### Question file — `MultiModalQA_train.jsonl.gz`

```json
{
  "qid": "5454c14ad01e722c2619b66778daa98b",
  "question": "What was the net worth (USD bn) of the LGBT billionaire who is completely bald and wears thick glasses?",
  "answers": [
    {
      "answer": "4.0",
      "type": "string",
      "modality": "table",
      "table_indices": [[0, 2]]
    }
  ],
  "metadata": {
    "type": "Compose(TableQ,ImageListQ)",
    "modalities": ["image", "table"],
    "pseudo_language_question": "In [Members] of [LGBT billionaires] what was the [Net worth USDbn](s) when the [Name] {is completely bald and wears thick glasses?}",
    "image_doc_ids": ["89c1b7c3c061cc80bb98d99cbbec50dd", "0f3858e2186b2030b77c759fc727e20b"],
    "text_doc_ids": ["498369348c988d866b5fac0add45bac5"],
    "table_id": "46ae2a8e7928ed5a8e5f9c59323e5e49",
    "intermediate_answers": [
      [{"answer": "Domenico Dolce", "modality": "image"}]
    ]
  },
  "supporting_context": [
    {"doc_id": "46ae2a8e7928ed5a8e5f9c59323e5e49", "doc_part": "table"},
    {"doc_id": "d57e56eff064047af5a6ef074a570956", "doc_part": "image"}
  ]
}
```

#### Table file — `tables.jsonl.gz`

```json
{
  "id": "46ae2a8e7928ed5a8e5f9c59323e5e49",
  "title": "LGBT billionaires",
  "url": "https://en.wikipedia.org/wiki/LGBT_billionaires",
  "table": {
    "table_name": "Members",
    "header": [
      {"column_name": "Name"},
      {"column_name": "Country"},
      {"column_name": "Net worth USDbn"}
    ],
    "table_rows": [
      [{"text": "Domenico Dolce"}, {"text": "Italy"}, {"text": "4.0"}],
      [{"text": "Tim Cook"},       {"text": "USA"},   {"text": "1.0"}]
    ]
  }
}
```

#### Image metadata — `images.jsonl.gz`

```json
{"id": "89c1b7c3c061cc80bb98d99cbbec50dd", "title": "Domenico Dolce", "path": "Domenico_Dolce.jpg"}
{"id": "0f3858e2186b2030b77c759fc727e20b", "title": "Tim Cook",        "path": "Tim_Cook.jpg"}
```

#### Text file — `texts.jsonl.gz`

```json
{
  "id": "498369348c988d866b5fac0add45bac5",
  "title": "Domenico Dolce",
  "text": "Domenico Dolce is an Italian fashion designer and co-founder of Dolce & Gabbana..."
}
```

---

### ⚙️ What the System Does (Two-Hop Reasoning)

```
┌─────────────────────────────────────────────────────────────────┐
│  Question: "...who is completely bald and wears thick glasses?" │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                    ┌──────▼──────┐
                    │  Q-Type     │  → Compose(TableQ, ImageListQ)
                    │  Classifier │
                    └──────┬──────┘
                           │
          ┌────────────────▼────────────────┐
          │  HOP 1 — Image VQA              │
          │  Look at candidate images       │
          │  → Who is bald + thick glasses? │
          │  → "Domenico Dolce"  (bridge)   │
          └────────────────┬────────────────┘
                           │  bridge entity
          ┌────────────────▼────────────────┐
          │  HOP 2 — Table QA               │
          │  Look up "Domenico Dolce"        │
          │  in LGBT billionaires table     │
          │  → Net worth USDbn = "4.0"      │
          └────────────────┬────────────────┘
                           │
                    ┌──────▼──────┐
                    │   Answer    │  →  "4.0"
                    └─────────────┘
```

---

### 📤 Output

**Prediction file** (`predictions_.json`):
```json
{
  "5454c14ad01e722c2619b66778daa98b": ["4.0"]
}
```

**Evaluation metrics** (EM / F1):

| Method | Single-Modality | Multi-Modality | All |
|--------|:-:|:-:|:-:|
| Context-only | 7.94 / 10.73 | 6.63 / 9.00 | 7.41 / 10.03 |
| AutoRouting | 51.68 / 58.48 | 34.18 / 40.18 | 44.65 / 51.13 |
| **ImplicitDecomp** | **51.60 / 58.35** | **44.59 / 51.19** | **48.79 / 55.48** |

---

## 3. Table Sizes & Statistics

### Key Facts

| Stat | Value |
|------|-------|
| Total questions | 29,918 |
| Unique tables (approx.) | ~6,000 (one per Wikipedia article) |
| Columns per table | ~3–10 (variable, Wikipedia-sourced) |
| Rows per table | ~5–50 (variable, Wikipedia-sourced) |
| Table format | JSONL — structured rows + typed headers + entity links |
| Max tokens fed to model | 384 tokens (rows chunked if table is too long) |

> **One table per question context** — enforced in code:
> ```python
> assert len(tables) == 1, "Only support one table context for now."
> ```

### How the Model Sees a Table (Linearized)

Each row is serialized as:
```
Row 0 is: Name is Domenico Dolce ; Country is Italy ; Net worth USDbn is 4.0 .
Row 1 is: Name is Tim Cook ; Country is USA ; Net worth USDbn is 1.0 .
```

Long tables are **chunked row-by-row** to fit within the 384-token budget.

---

## 4. Lakehouse Architecture Proposal

> **Requirement:** Load all tables into a lakehouse (Presto/Spark), query via SQL into DataFrames, run Python analytics, store images in object storage with links in the lakehouse.

### ✅ Verdict: Fully Feasible

---

### Overall Architecture

```
MMQA JSONL files
      │
      ├─ tables.jsonl  ──────────────► Lakehouse (Spark/Presto/Databricks)
      │                                 ~6,000 tables, each its own schema
      │
      ├─ images.jsonl  ──────────────► Object Storage (S3 / GCS / ADLS)
      │   + images.zip                  e.g. s3://mmqa-images/Domenico_Dolce.jpg
      │                                 image metadata table keeps the URL
      │
      ├─ texts.jsonl   ──────────────► Lakehouse: mmqa.texts.paragraphs
      │
      └─ questions.jsonl ────────────► Lakehouse: mmqa.questions.dev/train/test
```

---

### Step 1 — Ingest 6,000 Tables (each with its own schema)

```python
import json, re
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql.functions import lit

spark = SparkSession.builder.appName("mmqa_ingest").enableHiveSupport().getOrCreate()

def sanitize(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', name).lower()[:128]

with open("MMQA_tables.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        table_id   = doc["id"]
        title      = doc["title"]
        table_name = doc["table"]["table_name"]
        headers    = [sanitize(col["column_name"]) for col in doc["table"]["header"]]
        rows       = [[cell["text"] for cell in row] for row in doc["table"]["table_rows"]]

        schema = StructType([StructField(h, StringType(), True) for h in headers])
        df = spark.createDataFrame(rows, schema=schema)

        # Add provenance columns
        df = df.withColumn("_table_id",   lit(table_id)) \
               .withColumn("_wiki_title", lit(title)) \
               .withColumn("_table_name", lit(table_name))

        fqn = f"mmqa.tables.{sanitize(title)}_{sanitize(table_name)}"
        df.write.mode("overwrite").saveAsTable(fqn)
```

**Also create a catalog registry** so the LLM knows which table to query:

```sql
CREATE TABLE mmqa.catalog.table_registry (
    table_id      STRING,
    wiki_title    STRING,
    table_name    STRING,
    lakehouse_fqn STRING,        -- e.g. mmqa.tables.lgbt_billionaires_members
    column_names  ARRAY<STRING>  -- sanitized column names
);
```

---

### Step 2 — Load Images into Object Storage + Metadata into Lakehouse

**Upload to S3:**
```python
import boto3
s3 = boto3.client("s3")

records = []
with open("MMQA_images.jsonl") as f:
    for line in f:
        doc = json.loads(line)
        s3.upload_file(
            f"final_dataset_images/{doc['path']}",
            "mmqa-images",
            doc['path']
        )
        records.append((
            doc["id"],
            doc["title"],
            doc["url"],
            f"s3://mmqa-images/{doc['path']}"
        ))

spark.createDataFrame(records, ["image_id","wiki_title","wiki_url","s3_path"]) \
     .write.mode("overwrite").saveAsTable("mmqa.images.metadata")
```

**Image metadata table in lakehouse:**
```sql
SELECT * FROM mmqa.images.metadata LIMIT 3;
-- image_id                          | wiki_title      | s3_path
-- 89c1b7c3c061cc80bb98d99cbbec50dd | Domenico Dolce  | s3://mmqa-images/Domenico_Dolce.jpg
-- 0f3858e2186b2030b77c759fc727e20b | Tim Cook        | s3://mmqa-images/Tim_Cook.jpg
```

---

### Step 3 — QA Pipeline: SQL → DataFrame → Python

#### Hop 1 — Image VQA (reads S3 paths from lakehouse)

```python
# Get candidate image S3 paths from lakehouse
image_urls = spark.sql("""
    SELECT m.s3_path
    FROM mmqa.questions.dev q
    JOIN mmqa.images.metadata m
      ON array_contains(q.image_doc_ids, m.image_id)
    WHERE q.qid = '5454c14ad01e722c2619b66778daa98b'
""").toPandas()["s3_path"].tolist()

# Run VQA model on images streamed from S3
bridge_entity = image_vqa_model.predict(
    question="Who is completely bald and wears thick glasses?",
    image_paths=image_urls
)
# → "Domenico Dolce"
```

#### Hop 2 — SQL → DataFrame → Python analytics

```python
# 1. Look up which lakehouse table to query
registry_row = spark.sql("""
    SELECT lakehouse_fqn, column_names
    FROM mmqa.catalog.table_registry
    WHERE table_id = '46ae2a8e7928ed5a8e5f9c59323e5e49'
""").collect()[0]

lakehouse_table = registry_row["lakehouse_fqn"]
# → "mmqa.tables.lgbt_billionaires_members"
# columns: ["name", "country", "net_worth_usdbn", "industry"]

# 2. LLM generates SQL given: question + bridge_entity + schema
generated_sql = f"""
    SELECT net_worth_usdbn
    FROM {lakehouse_table}
    WHERE name = '{bridge_entity}'
"""

# 3. Execute SQL → Spark DataFrame → Pandas
df = spark.sql(generated_sql).toPandas()
#    net_worth_usdbn
# 0  4.0

# 4. LLM generates Python analytics on the DataFrame (for count/sum/mean/compare)
# e.g. for a "sum" question:
result = df["net_worth_usdbn"].astype(float).sum()
# → 4.0

final_answer = str(result)  # → "4.0"
```

---

### Full End-to-End Flow

```
┌──────────────────────────────────────────────────────────────────┐
│                          LAKEHOUSE                               │
│                                                                  │
│  mmqa.questions.dev/train/test    (29,918 rows)                  │
│  mmqa.catalog.table_registry      (~6,000 rows, schema index)    │
│  mmqa.tables.*                    (~6,000 separate tables)       │
│  mmqa.texts.paragraphs            (text docs for TextQ)          │
│  mmqa.images.metadata             (image_id → s3_path)           │
└──────────┬───────────────────────────────────┬───────────────────┘
           │ SQL query                         │ s3_path lookup
           ▼                                   ▼
     Spark DataFrame                   Object Storage (S3)
           │                                   │
           │ .toPandas()                       │ stream/download
           ▼                                   ▼
     Pandas DataFrame              Image bytes → VQA model
           │                                   │
           │ LLM generates Python              │ bridge entity
           ▼                                   │
     Python analytics  ◄─────────────────────┘
           │
           ▼
     Final Answer: "4.0"
```

---

### Design Decisions Summary

| Decision | Recommendation |
|----------|---------------|
| 🗂️ **Table naming** | `mmqa.tables.{wiki_title}_{table_name}` (sanitized) |
| 🔤 **Column naming** | Sanitize spaces/special chars → underscores; store originals in registry |
| 🖼️ **Image storage** | S3/GCS/ADLS object storage; only the URL lives in the lakehouse |
| 📋 **Schema discovery** | `table_registry` stores column list so LLM generates correct SQL without scanning all 6,000 tables |
| 📄 **Text docs** | Single table `mmqa.texts.paragraphs` — BM25 or vector search on top |
| 🔗 **Multi-hop** | Hop 1 result (bridge entity) passed as parameter into Hop 2 SQL `WHERE` clause |
| 🔢 **Answer types** | `count/sum/mean` → Python on DataFrame after SQL fetch; `yes/no` → Python boolean; `image` → VQA model |
| ⚡ **Engine choice** | Databricks Delta Lake or AWS Athena + S3 are natural fits |

---

> **Bottom line:** The approach is fully feasible. The key insight is that each of the ~6,000 Wikipedia tables becomes its own lakehouse table with its own schema. A catalog/registry table acts as the LLM's "table of tables" — it looks up the right schema, generates SQL to pull a DataFrame, then generates Python to compute the final answer. Images live in object storage with their S3 URLs indexed in the lakehouse.