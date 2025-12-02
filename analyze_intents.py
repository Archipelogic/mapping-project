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


def _normalize_for_fuzzy(text):
    """Normalize text for fuzzy comparison."""
    if not text:
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"[^\w\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _fuzzy_cluster_intents(intent_counts, threshold=0.80, max_intents=1000):
    """Cluster similar intents using fuzzy matching."""
    # Limit to top N by frequency
    top_intents = [k for k, v in Counter(intent_counts).most_common(max_intents) if k]
    normalized = {intent: _normalize_for_fuzzy(intent) for intent in top_intents}
    
    clustered = set()
    clusters = []
    
    for i, intent1 in enumerate(top_intents):
        if intent1 in clustered:
            continue
        cluster = [(intent1, intent_counts[intent1])]
        clustered.add(intent1)
        norm1 = normalized[intent1]
        
        for intent2 in top_intents[i+1:]:
            if intent2 in clustered:
                continue
            norm2 = normalized[intent2]
            if SequenceMatcher(None, norm1, norm2).ratio() >= threshold:
                cluster.append((intent2, intent_counts[intent2]))
                clustered.add(intent2)
        
        if len(cluster) > 1:
            clusters.append(sorted(cluster, key=lambda x: -x[1]))
    
    clusters.sort(key=lambda c: -sum(count for _, count in c))
    return clusters


def generate_starter_taxonomy(input_path, output_path="taxonomy.json", top_n=100):
    """Generate a starter taxonomy.json from raw data."""
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    intents = []
    parse_errors = 0
    for i, resp in enumerate(df['genai_response']):
        try:
            data = json.loads(resp)
            # Handle various nested structures
            inner = data
            for key in ["GenAI_Summary", "ContactSummary", "genai_summary", "contact_summary"]:
                if key in data:
                    inner = data[key]
                    if isinstance(inner, str):
                        try:
                            inner = json.loads(inner)
                        except:
                            inner = data
                    break
            if isinstance(inner, dict):
                intent = inner.get('Intent') or inner.get('intent', '')
                if intent:
                    intents.append(intent)
            # Debug: show first record structure
            if i == 0:
                print(f"  Sample keys: {list(data.keys())}")
                print(f"  Inner keys: {list(inner.keys()) if isinstance(inner, dict) else 'not a dict'}")
        except Exception as e:
            parse_errors += 1
            if parse_errors == 1:
                print(f"  Parse error sample: {e}")
    
    intent_counts = Counter(intents)
    print(f"Found {len(intent_counts)} unique intents (from {len(intents)} total, {parse_errors} parse errors)")
    
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
    
    # Pass 1: Keyword-based categorization
    print("  Pass 1: Keyword matching...")
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
    print(f"    Keyword matches: {len(assigned)}")
    
    # Pass 2: Fuzzy clustering for remaining intents
    print("  Pass 2: Fuzzy clustering variations...")
    remaining = {k: v for k, v in intent_counts.items() if k not in assigned and k}
    
    if remaining:
        # Fuzzy cluster the remaining intents
        fuzzy_clusters = _fuzzy_cluster_intents(remaining, threshold=0.80, max_intents=1000)
        
        fuzzy_assigned = 0
        for cluster in fuzzy_clusters:
            if len(cluster) < 2:
                continue
            # Use most frequent as canonical name
            canonical = cluster[0][0]
            # Create a sanitized category name
            cat_name = f"auto_{re.sub(r'[^a-z0-9]+', '_', canonical.lower())[:30]}"
            variations = [intent for intent, count in cluster]
            categories[cat_name] = {
                "description": f"AUTO-GROUPED: Similar to '{canonical}'",
                "is_billing": False,  # Review needed
                "variations": variations
            }
            for intent, _ in cluster:
                assigned.add(intent)
                fuzzy_assigned += 1
        print(f"    Fuzzy clusters: {len([c for c in fuzzy_clusters if len(c) >= 2])}, intents grouped: {fuzzy_assigned}")
    
    # Remaining uncategorized
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
        # Handle various nested structures
        inner = data
        for key in ["GenAI_Summary", "ContactSummary", "genai_summary", "contact_summary"]:
            if key in data:
                inner = data[key]
                if isinstance(inner, str):
                    try:
                        inner = json.loads(inner)
                    except:
                        inner = data
                break
        if not isinstance(inner, dict):
            inner = {}
        return {"Intent": inner.get("Intent") or inner.get("intent"),
                "Customer_On_Hold": inner.get("Customer_On_Hold") or inner.get("customer_on_hold"),
                "Transfer_Details": inner.get("Transfer_Details") or inner.get("transfer_details"),
                "Summary": inner.get("Summary") or inner.get("summary"), "parse_error": None}
    except:
        intent_match = re.search(r'"Intent"\s*:\s*"([^"]*)"', response_str)
        return {"Intent": intent_match.group(1) if intent_match else None, "Customer_On_Hold": None,
                "Transfer_Details": None, "Summary": None, "parse_error": "fallback"}


def normalize_intent(raw_intent, exact_mapping, categories_meta, fuzzy_threshold=0.80, enable_fuzzy=True):
    """Map raw intent to canonical category."""
    if pd.isna(raw_intent) or not raw_intent:
        return ("_empty", False, 0.0, "empty")
    
    raw_lower = raw_intent.lower().strip()
    
    if raw_lower in exact_mapping:
        canonical = exact_mapping[raw_lower]
        is_billing = categories_meta.get(canonical, {}).get("is_billing", False)
        return (canonical, is_billing, 1.0, "exact")
    
    # Skip expensive fuzzy matching if disabled
    if not enable_fuzzy:
        return ("_unmapped", False, 0.0, "no_match")
    
    best_match, best_score = None, 0
    for variation, canonical in exact_mapping.items():
        score = SequenceMatcher(None, raw_lower, variation).ratio()
        if score > best_score and score >= fuzzy_threshold:
            best_score, best_match = score, canonical
    
    if best_match:
        is_billing = categories_meta.get(best_match, {}).get("is_billing", False)
        return (best_match, is_billing, best_score, "fuzzy")
    
    return ("_unmapped", False, 0.0, "unmapped")


def analyze_call_data(input_path, taxonomy, enable_fuzzy=True):
    """Main analysis pipeline."""
    print(f"Loading: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} records")
    
    exact_mapping, categories_meta = build_intent_mapping(taxonomy)
    fuzzy_threshold = taxonomy.get("settings", {}).get("fuzzy_threshold", 0.80)
    print(f"Taxonomy: {len(categories_meta)} categories, {len(exact_mapping)} variations")
    if not enable_fuzzy:
        print("Fuzzy matching: DISABLED (fast mode)")
    
    parsed = df['genai_response'].apply(parse_genai_response)
    parsed_df = pd.DataFrame(parsed.tolist())
    df['raw_intent'] = parsed_df['Intent']
    df['customer_on_hold'] = parsed_df['Customer_On_Hold']
    df['transfer_details'] = parsed_df['Transfer_Details']
    df['summary'] = parsed_df['Summary']
    
    print("Normalizing intents...")
    normalized = df['raw_intent'].apply(lambda x: normalize_intent(x, exact_mapping, categories_meta, fuzzy_threshold, enable_fuzzy))
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


def generate_html_taxonomy_report(taxonomy, stats, output_path):
    """Generate interactive HTML report for taxonomy review."""
    categories = taxonomy.get("categories", {})
    meta = taxonomy.get("_meta", {})
    
    # Calculate category stats
    cat_stats = []
    for cat_name, cat_data in categories.items():
        count = stats['intent_distribution'].get(cat_name, 0)
        pct = count / stats['total_records'] * 100 if stats['total_records'] > 0 else 0
        cat_stats.append({
            "name": cat_name,
            "description": cat_data.get("description", ""),
            "is_billing": cat_data.get("is_billing", False),
            "variations": cat_data.get("variations", []),
            "count": count,
            "pct": pct
        })
    cat_stats.sort(key=lambda x: -x['count'])
    
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Taxonomy Report</title>
    <style>
        * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        body {{ margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{ color: #333; margin-bottom: 5px; }}
        .subtitle {{ color: #666; margin-bottom: 20px; }}
        .stats-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-bottom: 30px; }}
        .stat-card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .stat-value {{ font-size: 32px; font-weight: bold; color: #2563eb; }}
        .stat-label {{ color: #666; font-size: 14px; }}
        .coverage-good {{ color: #16a34a; }}
        .coverage-bad {{ color: #dc2626; }}
        .category-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
        .category-card {{ background: white; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); overflow: hidden; }}
        .category-header {{ padding: 15px 20px; border-bottom: 1px solid #eee; display: flex; justify-content: space-between; align-items: center; }}
        .category-name {{ font-weight: 600; font-size: 16px; }}
        .category-name.auto {{ color: #9333ea; }}
        .category-name.uncategorized {{ color: #dc2626; }}
        .category-count {{ background: #e5e7eb; padding: 4px 12px; border-radius: 20px; font-size: 14px; }}
        .billing-badge {{ background: #dbeafe; color: #1d4ed8; padding: 4px 8px; border-radius: 4px; font-size: 12px; margin-left: 8px; }}
        .category-desc {{ padding: 10px 20px; color: #666; font-size: 14px; border-bottom: 1px solid #eee; }}
        .variations {{ padding: 15px 20px; max-height: 200px; overflow-y: auto; }}
        .variation {{ display: inline-block; background: #f3f4f6; padding: 4px 10px; margin: 3px; border-radius: 4px; font-size: 13px; font-family: monospace; }}
        .section-title {{ font-size: 20px; margin: 30px 0 15px; color: #333; }}
        .filter-bar {{ margin-bottom: 20px; }}
        .filter-bar input {{ padding: 10px 15px; width: 300px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; }}
        .filter-bar select {{ padding: 10px 15px; border: 1px solid #ddd; border-radius: 6px; font-size: 14px; margin-left: 10px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Taxonomy Report</h1>
        <p class="subtitle">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} | Source: {meta.get('generated_from', 'taxonomy.json')}</p>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-value">{stats['total_records']:,}</div>
                <div class="stat-label">Total Records</div>
            </div>
            <div class="stat-card">
                <div class="stat-value {'coverage-good' if stats['pct_mapped'] >= 95 else 'coverage-bad'}">{stats['pct_mapped']:.1f}%</div>
                <div class="stat-label">Coverage</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{len([c for c in categories if not c.startswith('_')])}</div>
                <div class="stat-label">Categories</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['billing_related_count']:,}</div>
                <div class="stat-label">Billing-Related ({stats['pct_billing_related']:.1f}%)</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">{stats['records_unmapped']:,}</div>
                <div class="stat-label">Unmapped</div>
            </div>
        </div>
        
        <div class="filter-bar">
            <input type="text" id="search" placeholder="Search categories or variations..." onkeyup="filterCards()">
            <select id="typeFilter" onchange="filterCards()">
                <option value="all">All Categories</option>
                <option value="billing">Billing Only</option>
                <option value="auto">Auto-Generated</option>
                <option value="uncategorized">Uncategorized</option>
            </select>
        </div>
        
        <h2 class="section-title">Categories</h2>
        <div class="category-grid" id="categoryGrid">
"""
    
    for cat in cat_stats:
        name_class = ""
        if cat['name'].startswith('auto_'):
            name_class = "auto"
        elif cat['name'].startswith('_'):
            name_class = "uncategorized"
        
        billing_badge = '<span class="billing-badge">BILLING</span>' if cat['is_billing'] else ''
        
        variations_html = ''.join([f'<span class="variation">{v}</span>' for v in cat['variations'][:50]])
        if len(cat['variations']) > 50:
            variations_html += f'<span class="variation">+{len(cat["variations"]) - 50} more...</span>'
        
        html += f"""
            <div class="category-card" data-name="{cat['name']}" data-billing="{str(cat['is_billing']).lower()}" data-variations="{' '.join(cat['variations'][:20]).lower()}">
                <div class="category-header">
                    <div>
                        <span class="category-name {name_class}">{cat['name']}</span>
                        {billing_badge}
                    </div>
                    <span class="category-count">{cat['count']:,} ({cat['pct']:.1f}%)</span>
                </div>
                <div class="category-desc">{cat['description']}</div>
                <div class="variations">{variations_html}</div>
            </div>
"""
    
    html += """
        </div>
    </div>
    
    <script>
    function filterCards() {
        const search = document.getElementById('search').value.toLowerCase();
        const typeFilter = document.getElementById('typeFilter').value;
        const cards = document.querySelectorAll('.category-card');
        
        cards.forEach(card => {
            const name = card.dataset.name.toLowerCase();
            const variations = card.dataset.variations;
            const isBilling = card.dataset.billing === 'true';
            
            let show = true;
            
            // Search filter
            if (search && !name.includes(search) && !variations.includes(search)) {
                show = false;
            }
            
            // Type filter
            if (typeFilter === 'billing' && !isBilling) show = false;
            if (typeFilter === 'auto' && !name.startsWith('auto_')) show = false;
            if (typeFilter === 'uncategorized' && !name.startsWith('_')) show = false;
            
            card.style.display = show ? 'block' : 'none';
        });
    }
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)


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
    
    # HTML Taxonomy Report
    generate_html_taxonomy_report(taxonomy, stats, output_path / "taxonomy_report.html")
    print(f"Saved: {output_path}/taxonomy_report.html")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Normalize and analyze call intents")
    parser.add_argument("--input", "-i", default="data/query_results_genai_summaries.csv")
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument("--taxonomy", "-t", default=DEFAULT_TAXONOMY_PATH)
    parser.add_argument("--init-taxonomy", action="store_true", help="Generate starter taxonomy.json")
    parser.add_argument("--no-fuzzy", action="store_true", help="Disable fuzzy matching (faster, exact match only)")
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
    
    df, stats = analyze_call_data(args.input, taxonomy, enable_fuzzy=not args.no_fuzzy)
    print("\nGenerating deliverables...")
    generate_deliverables(df, stats, taxonomy, args.output)
    
    print("\n" + "="*50)
    print(f"Coverage: {stats['pct_mapped']:.1f}% | Unmapped: {stats['records_unmapped']:,}")
    print(f"Billing-related: {stats['billing_related_count']:,} ({stats['pct_billing_related']:.1f}%)")
    if stats['pct_mapped'] < 95:
        print("\nReview output/coverage_gaps.md and update taxonomy.json")


if __name__ == "__main__":
    main()
