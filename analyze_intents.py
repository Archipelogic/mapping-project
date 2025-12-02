"""
Phase 1: Friction Analysis Pipeline

Config-driven intent normalization and analysis.

Workflow:
1. Run explore_intents.py to understand raw data
2. Run: python analyze_intents.py --init-taxonomy to generate starter taxonomy.json
3. Manually refine taxonomy.json based on exploration findings
4. Run: python analyze_intents.py to normalize and generate deliverables
5. Check coverage report, refine taxonomy, repeat until 95%+ coverage
"""

import pandas as pd
import json
import re
from collections import Counter
from pathlib import Path
from difflib import SequenceMatcher
from datetime import datetime

DEFAULT_TAXONOMY_PATH = "taxonomy.json"


def load_taxonomy(taxonomy_path):
    """Load taxonomy from JSON config file."""
    path = Path(taxonomy_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Taxonomy file not found: {taxonomy_path}\n"
            f"Run with --init-taxonomy to generate a starter taxonomy from your data."
        )
    with open(path) as f:
        return json.load(f)


def build_intent_mapping(taxonomy):
    """Build lookup dictionary from taxonomy config."""
    exact_mapping = {}
    categories_meta = {}
    
    for canonical, data in taxonomy.get("categories", {}).items():
        if canonical.startswith("_"):
            continue
        categories_meta[canonical] = {
            "description": data.get("description", ""),
            "is_billing": data.get("is_billing", False)
        }
        for variation in data.get("variations", []):
            exact_mapping[variation.lower().strip()] = canonical
    
    return exact_mapping, categories_meta


def generate_starter_taxonomy(input_path, output_path="taxonomy.json", top_n=100):
    """Generate a starter taxonomy.json from raw data."""
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    intents = []
    for resp in df['genai_response']:
        try:
            data = json.loads(resp)
            # Handle nested structure: {"GenAI_Summary": {"Intent": ...}}
            if "GenAI_Summary" in data and isinstance(data["GenAI_Summary"], dict):
                intent = data["GenAI_Summary"].get('Intent', '')
            else:
                intent = data.get('Intent', '')
            if intent:
                intents.append(intent)
        except:
            pass
    
    intent_counts = Counter(intents)
    print(f"Found {len(intent_counts)} unique intents")
    
    keyword_groups = {
        "payment_make": {"keywords": ["pay ", "payment", "make payment"], "is_billing": True, "description": "Customer wants to make a payment"},
        "billing_inquiry": {"keywords": ["bill", "billing", "charge"], "is_billing": True, "description": "Customer has billing questions"},
        "balance_check": {"keywords": ["balance", "owe", "amount due"], "is_billing": True, "description": "Customer wants to check balance"},
        "autopay": {"keywords": ["autopay", "auto pay", "automatic"], "is_billing": True, "description": "Customer wants to manage autopay"},
        "dispute": {"keywords": ["dispute", "wrong", "incorrect", "overcharge"], "is_billing": True, "description": "Customer disputes a charge"},
        "account_access": {"keywords": ["login", "password", "access", "locked"], "is_billing": True, "description": "Customer has account access issues"},
        "payment_method": {"keywords": ["card", "bank account", "payment method"], "is_billing": True, "description": "Customer wants to update payment method"},
        "refund": {"keywords": ["refund", "credit"], "is_billing": True, "description": "Customer requesting refund"},
        "claims": {"keywords": ["claim", "accident", "damage"], "is_billing": False, "description": "Customer has claims-related inquiry"},
        "policy": {"keywords": ["policy", "coverage", "renewal"], "is_billing": False, "description": "Customer has policy-related inquiry"},
    }
    
    categories = {}
    assigned = set()
    
    for group_name, group_info in keyword_groups.items():
        variations = []
        for intent, count in intent_counts.most_common():
            if intent in assigned:
                continue
            if any(kw in intent.lower() for kw in group_info["keywords"]):
                variations.append(intent)
                assigned.add(intent)
        if variations:
            categories[group_name] = {
                "description": group_info["description"],
                "is_billing": group_info["is_billing"],
                "variations": sorted(variations, key=lambda x: -intent_counts[x])
            }
    
    uncategorized = [i for i, c in intent_counts.most_common(top_n) if i not in assigned and i]
    if uncategorized:
        categories["_uncategorized"] = {
            "description": "REVIEW: Intents needing manual categorization",
            "is_billing": False,
            "variations": uncategorized
        }
    
    taxonomy = {
        "_meta": {
            "generated_from": str(input_path),
            "total_unique_intents": len(intent_counts),
            "intents_categorized": len(assigned),
        },
        "settings": {"fuzzy_threshold": 0.80},
        "categories": categories
    }
    
    with open(output_path, 'w') as f:
        json.dump(taxonomy, f, indent=2)
    
    print(f"\nGenerated: {output_path}")
    print(f"  {len(categories)-1} categories, {len(assigned)} auto-categorized, {len(uncategorized)} need review")
    print(f"\nNext: Edit {output_path}, then run: python analyze_intents.py")


def parse_genai_response(response_str):
    """Parse the genai_response column."""
    if pd.isna(response_str) or not response_str:
        return {"Intent": None, "Customer_On_Hold": None, "Transfer_Details": None, "Summary": None, "parse_error": "empty"}
    try:
        data = json.loads(response_str)
        # Handle nested structure: {"GenAI_Summary": {"Intent": ...}}
        if "GenAI_Summary" in data and isinstance(data["GenAI_Summary"], dict):
            inner = data["GenAI_Summary"]
        else:
            inner = data
        return {"Intent": inner.get("Intent"), "Customer_On_Hold": inner.get("Customer_On_Hold"),
                "Transfer_Details": inner.get("Transfer_Details"), "Summary": inner.get("Summary"), "parse_error": None}
    except:
        intent_match = re.search(r'"Intent"\s*:\s*"([^"]*)"', response_str)
        return {"Intent": intent_match.group(1) if intent_match else None, "Customer_On_Hold": None,
                "Transfer_Details": None, "Summary": None, "parse_error": "fallback"}


def normalize_intent(raw_intent, exact_mapping, categories_meta, fuzzy_threshold=0.80):
    """Map raw intent to canonical category."""
    if pd.isna(raw_intent) or not raw_intent:
        return ("_empty", False, 0.0, "empty")
    
    raw_lower = raw_intent.lower().strip()
    
    if raw_lower in exact_mapping:
        canonical = exact_mapping[raw_lower]
        is_billing = categories_meta.get(canonical, {}).get("is_billing", False)
        return (canonical, is_billing, 1.0, "exact")
    
    best_match, best_score = None, 0
    for variation, canonical in exact_mapping.items():
        score = SequenceMatcher(None, raw_lower, variation).ratio()
        if score > best_score and score >= fuzzy_threshold:
            best_score, best_match = score, canonical
    
    if best_match:
        is_billing = categories_meta.get(best_match, {}).get("is_billing", False)
        return (best_match, is_billing, best_score, "fuzzy")
    
    return ("_unmapped", False, 0.0, "unmapped")


def analyze_call_data(input_path, taxonomy):
    """Main analysis pipeline."""
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} records")
    
    exact_mapping, categories_meta = build_intent_mapping(taxonomy)
    fuzzy_threshold = taxonomy.get("settings", {}).get("fuzzy_threshold", 0.80)
    print(f"Taxonomy: {len(categories_meta)} categories, {len(exact_mapping)} variations")
    
    parsed = df['genai_response'].apply(parse_genai_response)
    parsed_df = pd.DataFrame(parsed.tolist())
    df['raw_intent'] = parsed_df['Intent']
    df['customer_on_hold'] = parsed_df['Customer_On_Hold']
    df['transfer_details'] = parsed_df['Transfer_Details']
    df['summary'] = parsed_df['Summary']
    
    print("Normalizing intents...")
    normalized = df['raw_intent'].apply(lambda x: normalize_intent(x, exact_mapping, categories_meta, fuzzy_threshold))
    df['normalized_intent'] = normalized.apply(lambda x: x[0])
    df['is_billing_related'] = normalized.apply(lambda x: x[1])
    df['intent_confidence'] = normalized.apply(lambda x: x[2])
    df['match_type'] = normalized.apply(lambda x: x[3])
    
    df['customer_on_hold'] = df['customer_on_hold'].apply(lambda x: x if isinstance(x, bool) else False)
    df['transfer_occurred'] = df['transfer_details'].apply(lambda x: 'no transfer' not in str(x).lower() if pd.notna(x) else False)
    
    stats = calculate_statistics(df, categories_meta)
    return df, stats


def calculate_statistics(df, categories_meta):
    """Calculate statistics including coverage metrics."""
    total = len(df)
    stats = {"total_records": total}
    
    mapped = df[~df['normalized_intent'].isin(['_unmapped', '_empty'])]
    stats["records_mapped"] = len(mapped)
    stats["pct_mapped"] = len(mapped) / total * 100 if total > 0 else 0
    stats["records_unmapped"] = len(df[df['normalized_intent'] == '_unmapped'])
    stats["pct_unmapped"] = stats["records_unmapped"] / total * 100 if total > 0 else 0
    
    stats["match_type_distribution"] = df['match_type'].value_counts().to_dict()
    stats["billing_related_count"] = int(df['is_billing_related'].sum())
    stats["pct_billing_related"] = float(df['is_billing_related'].mean() * 100)
    
    if 'totalcalldurationinseconds' in df.columns:
        stats["avg_call_duration"] = float(df['totalcalldurationinseconds'].mean())
    
    stats["pct_on_hold"] = float(df['customer_on_hold'].mean() * 100)
    stats["pct_transferred"] = float(df['transfer_occurred'].mean() * 100)
    stats["intent_distribution"] = df['normalized_intent'].value_counts().to_dict()
    stats["unmapped_intents"] = df[df['normalized_intent'] == '_unmapped']['raw_intent'].value_counts().head(50).to_dict()
    
    if 'channel' in df.columns:
        stats["channel_distribution"] = df['channel'].value_counts().to_dict()
    
    return stats


def generate_deliverables(df, stats, taxonomy, output_dir="output"):
    """Generate all deliverables."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    # Cleaned dataset
    cols = ['contactid', 'channel', 'raw_intent', 'normalized_intent', 'is_billing_related', 
            'intent_confidence', 'match_type', 'customer_on_hold', 'transfer_occurred', 'totalcalldurationinseconds']
    available = [c for c in cols if c in df.columns]
    df[available].to_csv(output_path / "cleaned_call_dataset.csv", index=False)
    print(f"Saved: {output_path}/cleaned_call_dataset.csv")
    
    # Coverage gaps
    with open(output_path / "coverage_gaps.md", 'w') as f:
        f.write("# Coverage Gap Report\n\n")
        f.write(f"**Coverage: {stats['pct_mapped']:.1f}%** ({stats['records_mapped']:,} of {stats['total_records']:,})\n\n")
        f.write("## Unmapped Intents (add to taxonomy.json)\n\n")
        f.write("| Intent | Count |\n|--------|-------|\n")
        for intent, count in stats.get('unmapped_intents', {}).items():
            f.write(f"| `{intent}` | {count} |\n")
        if stats['pct_mapped'] >= 95:
            f.write("\n\n**Coverage target met!**\n")
        else:
            f.write(f"\n\n**Coverage below 95%** - add unmapped intents to taxonomy.json\n")
    print(f"Saved: {output_path}/coverage_gaps.md")
    
    # Distribution report
    _, categories_meta = build_intent_mapping(taxonomy)
    with open(output_path / "intent_distribution_report.md", 'w') as f:
        f.write("# Intent Distribution Report\n\n")
        f.write(f"- **Records**: {stats['total_records']:,}\n")
        f.write(f"- **Coverage**: {stats['pct_mapped']:.1f}%\n")
        f.write(f"- **Billing-Related**: {stats['billing_related_count']:,} ({stats['pct_billing_related']:.1f}%)\n\n")
        f.write("## Intent Distribution\n\n| Intent | Count | % | Billing |\n|--------|-------|---|--------|\n")
        for intent, count in stats['intent_distribution'].items():
            pct = count / stats['total_records'] * 100
            billing = "Y" if categories_meta.get(intent, {}).get("is_billing") else ""
            f.write(f"| {intent} | {count:,} | {pct:.1f}% | {billing} |\n")
    print(f"Saved: {output_path}/intent_distribution_report.md")
    
    # Quality report JSON
    with open(output_path / "data_quality_report.json", 'w') as f:
        json.dump({"coverage_pct": stats['pct_mapped'], "unmapped_count": stats['records_unmapped'],
                   "billing_pct": stats['pct_billing_related'], "match_types": stats['match_type_distribution']}, f, indent=2)
    print(f"Saved: {output_path}/data_quality_report.json")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Normalize and analyze call intents")
    parser.add_argument("--input", "-i", default="data/query_results_genai_summaries.csv")
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument("--taxonomy", "-t", default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--init-taxonomy", action="store_true", help="Generate starter taxonomy.json")
    args = parser.parse_args()
    
    if args.init_taxonomy:
        generate_starter_taxonomy(args.input, args.taxonomy)
        return
    
    try:
        taxonomy = load_taxonomy(args.taxonomy)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nRun: python analyze_intents.py --init-taxonomy")
        return
    
    df, stats = analyze_call_data(args.input, taxonomy)
    print("\nGenerating deliverables...")
    generate_deliverables(df, stats, taxonomy, args.output)
    
    print("\n" + "="*50)
    print(f"Coverage: {stats['pct_mapped']:.1f}% | Unmapped: {stats['records_unmapped']:,}")
    print(f"Billing-related: {stats['billing_related_count']:,} ({stats['pct_billing_related']:.1f}%)")
    if stats['pct_mapped'] < 95:
        print("\nReview output/coverage_gaps.md and update taxonomy.json")


if __name__ == "__main__":
    main()
