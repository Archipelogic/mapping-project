# Mapping Project

Config-driven intent normalization and taxonomy mapping for customer service data.

## Workflow

```
1. EXPLORE              2. INIT TAXONOMY           3. ANALYZE
───────────────────     ───────────────────        ───────────────────
python explore_intents.py   python analyze_intents.py  Edit taxonomy.json
                            --init-taxonomy            python analyze_intents.py
                                                       Repeat until 95%+
```

## Quick Start

```bash
pip install -r requirements.txt

# Generate sample data (optional)
python generate_synthetic_data.py

# Explore raw intents
python explore_intents.py

# Generate starter taxonomy
python analyze_intents.py --init-taxonomy

# Run analysis (after refining taxonomy.json)
python analyze_intents.py
```

## Input Schema

| Column | Description |
|--------|-------------|
| `contactid` | Unique identifier |
| `channel` | Contact channel |
| `genai_response` | JSON with Intent, Summary, etc. |
| `totalcalldurationinseconds` | Duration |

## Taxonomy Format

```json
{
  "settings": {"fuzzy_threshold": 0.80},
  "categories": {
    "payment_make": {
      "description": "Customer wants to make a payment",
      "is_billing": true,
      "variations": ["make payment", "pay bill", ...]
    }
  }
}
```

## Outputs

| File | Description |
|------|-------------|
| `cleaned_call_dataset.csv` | Normalized data with mapped intents |
| `coverage_gaps.md` | Unmapped intents needing taxonomy updates |
| `intent_distribution_report.md` | Frequency analysis |
| `data_quality_report.json` | Coverage metrics |

## Project Structure

```
mapping-project/
├── generate_synthetic_data.py   # Sample data generator
├── explore_intents.py           # Raw intent exploration
├── analyze_intents.py           # Normalization pipeline
├── taxonomy.json                # Intent taxonomy config
├── data/                        # Input data
└── output/                      # Generated reports
```
