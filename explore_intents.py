"""
Intent Discovery & Exploration v2

Enhanced exploration with:
- Dataset overview statistics
- Fuzzy variation detection
- Keyword-based clustering
- Multi-field analysis (Intent, Transfer_Details, ActionItems, Summary)
- Interactive HTML report
"""

import pandas as pd
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from difflib import SequenceMatcher
from datetime import datetime

# =============================================================================
# PARSING
# =============================================================================

def parse_genai_response(response_str):
    """Extract all fields from genai_response column."""
    if pd.isna(response_str) or not response_str:
        return {
            "Intent": None,
            "Customer_On_Hold": None,
            "Transfer_Details": None,
            "ActionItems_Caller": None,
            "Summary": None,
            "parse_success": False,
            "parse_error": "empty"
        }
    
    try:
        data = json.loads(response_str)
        return {
            "Intent": data.get("Intent"),
            "Customer_On_Hold": data.get("Customer_On_Hold"),
            "Transfer_Details": data.get("Transfer_Details"),
            "ActionItems_Caller": data.get("ActionItems_Caller"),
            "Summary": data.get("Summary"),
            "parse_success": True,
            "parse_error": None
        }
    except json.JSONDecodeError as e:
        # Regex fallback
        intent_match = re.search(r'"Intent"\s*:\s*"([^"]*)"', response_str)
        return {
            "Intent": intent_match.group(1) if intent_match else None,
            "Customer_On_Hold": None,
            "Transfer_Details": None,
            "ActionItems_Caller": None,
            "Summary": None,
            "parse_success": intent_match is not None,
            "parse_error": str(e)[:50]
        }

# =============================================================================
# DATASET OVERVIEW
# =============================================================================

def generate_dataset_overview(df, parsed_df):
    """Generate high-level dataset statistics."""
    
    overview = {
        "total_rows": len(df),
        "columns": list(df.columns),
        "column_types": df.dtypes.astype(str).to_dict(),
    }
    
    # Memory usage
    overview["memory_mb"] = df.memory_usage(deep=True).sum() / 1024 / 1024
    
    # Column completeness
    overview["column_completeness"] = {
        col: (df[col].notna().sum() / len(df) * 100) 
        for col in df.columns
    }
    
    # Parsed field completeness
    overview["parsed_field_completeness"] = {
        "Intent": (parsed_df["Intent"].notna() & (parsed_df["Intent"] != "")).sum() / len(df) * 100,
        "Customer_On_Hold": parsed_df["Customer_On_Hold"].notna().sum() / len(df) * 100,
        "Transfer_Details": (parsed_df["Transfer_Details"].notna() & (parsed_df["Transfer_Details"] != "")).sum() / len(df) * 100,
        "ActionItems_Caller": parsed_df["ActionItems_Caller"].notna().sum() / len(df) * 100,
        "Summary": (parsed_df["Summary"].notna() & (parsed_df["Summary"] != "")).sum() / len(df) * 100,
    }
    
    # Parse success rate
    overview["parse_success_rate"] = parsed_df["parse_success"].mean() * 100
    overview["parse_errors"] = parsed_df[parsed_df["parse_error"].notna()]["parse_error"].value_counts().head(5).to_dict()
    
    # Channel distribution (if exists)
    if "channel" in df.columns:
        overview["channel_distribution"] = df["channel"].value_counts().to_dict()
    
    # Call duration stats (if exists)
    if "totalcalldurationinseconds" in df.columns:
        duration = df["totalcalldurationinseconds"]
        overview["call_duration"] = {
            "mean_seconds": duration.mean(),
            "median_seconds": duration.median(),
            "min_seconds": duration.min(),
            "max_seconds": duration.max(),
            "std_seconds": duration.std(),
        }
    
    return overview

# =============================================================================
# FUZZY VARIATION DETECTION
# =============================================================================

def normalize_for_comparison(text):
    """Normalize text for fuzzy comparison."""
    if not text:
        return ""
    text = str(text).lower().strip()
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", " ", text)
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    return text

def similarity_ratio(s1, s2):
    """Calculate similarity ratio between two strings."""
    return SequenceMatcher(None, s1, s2).ratio()

def find_fuzzy_clusters(intents_counter, threshold=0.75):
    """
    Find clusters of similar intents using fuzzy matching.
    Returns list of clusters, each cluster is a list of (intent, count) tuples.
    """
    intents = list(intents_counter.keys())
    normalized = {intent: normalize_for_comparison(intent) for intent in intents}
    
    # Track which intents have been clustered
    clustered = set()
    clusters = []
    
    for i, intent1 in enumerate(intents):
        if intent1 in clustered:
            continue
            
        cluster = [(intent1, intents_counter[intent1])]
        clustered.add(intent1)
        norm1 = normalized[intent1]
        
        for intent2 in intents[i+1:]:
            if intent2 in clustered:
                continue
            norm2 = normalized[intent2]
            
            # Check similarity
            if similarity_ratio(norm1, norm2) >= threshold:
                cluster.append((intent2, intents_counter[intent2]))
                clustered.add(intent2)
        
        # Only keep clusters with multiple items or high-frequency singles
        if len(cluster) > 1:
            clusters.append(sorted(cluster, key=lambda x: -x[1]))
    
    # Sort clusters by total count
    clusters.sort(key=lambda c: -sum(count for _, count in c))
    
    return clusters

# =============================================================================
# KEYWORD-BASED CLUSTERING
# =============================================================================

# Define keyword groups for billing-related categorization
KEYWORD_GROUPS = {
    "payment_action": ["pay", "payment", "paid", "paying"],
    "billing": ["bill", "billing", "billed", "charge", "charged"],
    "balance": ["balance", "owe", "due", "amount"],
    "autopay": ["autopay", "auto pay", "automatic"],
    "dispute": ["dispute", "wrong", "incorrect", "error", "overcharge"],
    "refund": ["refund", "credit", "reimburse"],
    "account_access": ["login", "log in", "password", "locked", "access", "cant access", "can't access"],
    "payment_method": ["card", "bank", "payment method", "update payment"],
    "history": ["history", "past", "previous", "record"],
    "claims": ["claim", "accident", "incident", "damage"],
    "policy": ["policy", "coverage", "covered", "renewal", "cancel"],
    "documents": ["id card", "proof", "document", "certificate"],
}

def extract_keywords(text):
    """Extract which keyword groups match the text."""
    if not text:
        return set()
    
    text_lower = str(text).lower()
    matched_groups = set()
    
    for group_name, keywords in KEYWORD_GROUPS.items():
        for keyword in keywords:
            if keyword in text_lower:
                matched_groups.add(group_name)
                break
    
    return matched_groups

def cluster_by_keywords(intents_counter):
    """Group intents by keyword categories."""
    keyword_clusters = defaultdict(list)
    uncategorized = []
    
    for intent, count in intents_counter.items():
        groups = extract_keywords(intent)
        if groups:
            # Assign to primary group (first match)
            primary = sorted(groups)[0]
            keyword_clusters[primary].append((intent, count, groups))
        else:
            uncategorized.append((intent, count))
    
    # Sort within each cluster by count
    for group in keyword_clusters:
        keyword_clusters[group].sort(key=lambda x: -x[1])
    
    return dict(keyword_clusters), uncategorized

# =============================================================================
# MULTI-FIELD ANALYSIS
# =============================================================================

def analyze_field(series, field_name):
    """Analyze a single text field."""
    non_empty = series.dropna()
    non_empty = non_empty[non_empty.astype(str).str.strip() != ""]
    
    if len(non_empty) == 0:
        return {"empty": True}
    
    value_counts = Counter(non_empty)
    
    return {
        "empty": False,
        "total_non_empty": len(non_empty),
        "pct_non_empty": len(non_empty) / len(series) * 100,
        "unique_values": len(value_counts),
        "top_values": dict(value_counts.most_common(20)),
    }

def analyze_summary_keywords(summaries):
    """Analyze billing-related keywords in summaries."""
    keyword_counts = defaultdict(int)
    records_with_keyword = defaultdict(int)
    
    for summary in summaries:
        if not summary or pd.isna(summary):
            continue
        summary_lower = str(summary).lower()
        found_groups = set()
        
        for group_name, keywords in KEYWORD_GROUPS.items():
            for keyword in keywords:
                if keyword in summary_lower:
                    keyword_counts[group_name] += summary_lower.count(keyword)
                    found_groups.add(group_name)
                    break
        
        for group in found_groups:
            records_with_keyword[group] += 1
    
    return {
        "keyword_mention_counts": dict(keyword_counts),
        "records_with_keyword": dict(records_with_keyword),
    }

def multi_field_analysis(parsed_df):
    """Analyze all parsed fields."""
    return {
        "Intent": analyze_field(parsed_df["Intent"], "Intent"),
        "Transfer_Details": analyze_field(parsed_df["Transfer_Details"], "Transfer_Details"),
        "ActionItems_Caller": analyze_field(parsed_df["ActionItems_Caller"], "ActionItems_Caller"),
        "Summary_Keywords": analyze_summary_keywords(parsed_df["Summary"]),
        "Customer_On_Hold": {
            "true_count": (parsed_df["Customer_On_Hold"] == True).sum(),
            "false_count": (parsed_df["Customer_On_Hold"] == False).sum(),
            "pct_on_hold": (parsed_df["Customer_On_Hold"] == True).mean() * 100 if parsed_df["Customer_On_Hold"].notna().any() else 0,
        }
    }

# =============================================================================
# HTML REPORT GENERATION
# =============================================================================

def generate_html_report(overview, intent_analysis, fuzzy_clusters, keyword_clusters, 
                         uncategorized, multi_field, output_path):
    """Generate interactive HTML report."""
    
    html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Intent Exploration Report</title>
    <style>
        * { box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6; 
            max-width: 1400px; 
            margin: 0 auto; 
            padding: 20px;
            background: #f5f5f5;
        }
        h1, h2, h3 { color: #333; }
        h1 { border-bottom: 3px solid #2196F3; padding-bottom: 10px; }
        h2 { border-bottom: 2px solid #ddd; padding-bottom: 5px; margin-top: 30px; }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .stat-box {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 6px;
            text-align: center;
        }
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #2196F3;
        }
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 10px 0;
        }
        th, td {
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        th {
            background: #f8f9fa;
            font-weight: 600;
            cursor: pointer;
        }
        th:hover { background: #e9ecef; }
        tr:hover { background: #f8f9fa; }
        .cluster-box {
            background: #e3f2fd;
            border-left: 4px solid #2196F3;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 6px 6px 0;
        }
        .cluster-title {
            font-weight: bold;
            color: #1565c0;
        }
        .cluster-items {
            margin-top: 5px;
            font-family: monospace;
            font-size: 0.9em;
        }
        .keyword-tag {
            display: inline-block;
            background: #e8f5e9;
            color: #2e7d32;
            padding: 3px 8px;
            border-radius: 12px;
            font-size: 0.85em;
            margin: 2px;
        }
        .pct-bar {
            background: #e0e0e0;
            border-radius: 4px;
            height: 20px;
            width: 100%;
            position: relative;
        }
        .pct-fill {
            background: #2196F3;
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s;
        }
        .pct-text {
            position: absolute;
            right: 5px;
            top: 0;
            font-size: 0.8em;
            line-height: 20px;
        }
        .collapsible {
            cursor: pointer;
            padding: 10px;
            background: #f8f9fa;
            border: none;
            width: 100%;
            text-align: left;
            font-size: 1em;
            border-radius: 4px;
            margin-top: 5px;
        }
        .collapsible:hover { background: #e9ecef; }
        .collapsible:after {
            content: '\\25BC';
            float: right;
        }
        .collapsible.active:after {
            content: '\\25B2';
        }
        .content {
            display: none;
            padding: 10px;
            background: white;
        }
        .search-box {
            width: 100%;
            padding: 10px;
            font-size: 1em;
            border: 2px solid #ddd;
            border-radius: 6px;
            margin-bottom: 15px;
        }
        .search-box:focus {
            outline: none;
            border-color: #2196F3;
        }
        .highlight { background: yellow; }
    </style>
</head>
<body>
    <h1>🔍 Intent Exploration Report</h1>
    <p><em>Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</em></p>
    
    <!-- Dataset Overview -->
    <h2>📊 Dataset Overview</h2>
    <div class="card">
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">""" + f"{overview['total_rows']:,}" + """</div>
                <div class="stat-label">Total Records</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">""" + f"{overview.get('memory_mb', 0):.1f}" + """ MB</div>
                <div class="stat-label">Memory Usage</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">""" + f"{overview['parse_success_rate']:.1f}%" + """</div>
                <div class="stat-label">Parse Success Rate</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">""" + f"{intent_analysis['unique_intents']:,}" + """</div>
                <div class="stat-label">Unique Intents</div>
            </div>
        </div>
    </div>
    
    <!-- Field Completeness -->
    <div class="card">
        <h3>Field Completeness</h3>
        <table>
            <tr><th>Field</th><th>Completeness</th><th></th></tr>
"""
    
    for field, pct in overview['parsed_field_completeness'].items():
        html += f"""
            <tr>
                <td>{field}</td>
                <td style="width: 60%">
                    <div class="pct-bar">
                        <div class="pct-fill" style="width: {pct}%"></div>
                        <span class="pct-text">{pct:.1f}%</span>
                    </div>
                </td>
                <td>{pct:.1f}%</td>
            </tr>
"""
    
    html += """
        </table>
    </div>
"""
    
    # Call duration stats if available
    if 'call_duration' in overview:
        dur = overview['call_duration']
        html += f"""
    <div class="card">
        <h3>Call Duration Statistics</h3>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">{dur['mean_seconds']/60:.1f}</div>
                <div class="stat-label">Avg Duration (min)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{dur['median_seconds']/60:.1f}</div>
                <div class="stat-label">Median (min)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{dur['min_seconds']/60:.1f}</div>
                <div class="stat-label">Min (min)</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{dur['max_seconds']/60:.1f}</div>
                <div class="stat-label">Max (min)</div>
            </div>
        </div>
    </div>
"""
    
    # Channel distribution if available
    if 'channel_distribution' in overview:
        html += """
    <div class="card">
        <h3>Channel Distribution</h3>
        <table>
            <tr><th>Channel</th><th>Count</th><th>%</th></tr>
"""
        total = sum(overview['channel_distribution'].values())
        for channel, count in sorted(overview['channel_distribution'].items(), key=lambda x: -x[1]):
            pct = count / total * 100
            html += f"<tr><td>{channel}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>\n"
        html += "</table></div>\n"
    
    # Intent Frequency Table
    html += """
    <h2>📋 All Intents by Frequency</h2>
    <div class="card">
        <input type="text" class="search-box" id="intentSearch" placeholder="Search intents..." onkeyup="filterTable()">
        <table id="intentTable">
            <thead>
                <tr>
                    <th onclick="sortTable(0)">Rank</th>
                    <th onclick="sortTable(1)">Intent</th>
                    <th onclick="sortTable(2)">Count</th>
                    <th onclick="sortTable(3)">%</th>
                    <th onclick="sortTable(4)">Cumulative %</th>
                </tr>
            </thead>
            <tbody>
"""
    
    cumulative = 0
    for i, (intent, count) in enumerate(intent_analysis['frequency'].items(), 1):
        pct = count / intent_analysis['total_with_intent'] * 100
        cumulative += pct
        display_intent = intent if intent else "(empty)"
        html += f"""
                <tr>
                    <td>{i}</td>
                    <td><code>{display_intent}</code></td>
                    <td>{count:,}</td>
                    <td>{pct:.1f}%</td>
                    <td>{cumulative:.1f}%</td>
                </tr>
"""
    
    html += """
            </tbody>
        </table>
    </div>
"""
    
    # Fuzzy Clusters
    html += f"""
    <h2>🔗 Similar Intent Clusters (Fuzzy Match)</h2>
    <p>Found <strong>{len(fuzzy_clusters)}</strong> clusters of similar intents that may need consolidation.</p>
    <div class="card">
"""
    
    for i, cluster in enumerate(fuzzy_clusters[:30], 1):  # Show top 30 clusters
        total_count = sum(c for _, c in cluster)
        items_str = " | ".join([f'"{intent}" ({count})' for intent, count in cluster])
        html += f"""
        <div class="cluster-box">
            <div class="cluster-title">Cluster {i}: {total_count} total records</div>
            <div class="cluster-items">{items_str}</div>
        </div>
"""
    
    if len(fuzzy_clusters) > 30:
        html += f"<p><em>...and {len(fuzzy_clusters) - 30} more clusters</em></p>"
    
    html += "</div>\n"
    
    # Keyword-Based Clusters
    html += """
    <h2>🏷️ Keyword-Based Categories</h2>
    <p>Intents grouped by detected keywords - potential taxonomy categories.</p>
    <div class="card">
"""
    
    for group_name, intents in sorted(keyword_clusters.items(), key=lambda x: -sum(i[1] for i in x[1])):
        total = sum(i[1] for i in intents)
        html += f"""
        <button class="collapsible"><strong>{group_name}</strong> - {len(intents)} variations, {total:,} records</button>
        <div class="content">
            <table>
                <tr><th>Intent</th><th>Count</th><th>Other Keywords</th></tr>
"""
        for intent, count, groups in intents[:20]:
            other_groups = [g for g in groups if g != group_name]
            tags = "".join([f'<span class="keyword-tag">{g}</span>' for g in other_groups])
            html += f"<tr><td><code>{intent}</code></td><td>{count}</td><td>{tags}</td></tr>\n"
        
        if len(intents) > 20:
            html += f"<tr><td colspan='3'><em>...and {len(intents) - 20} more</em></td></tr>"
        html += "</table></div>\n"
    
    # Uncategorized
    if uncategorized:
        html += f"""
        <button class="collapsible"><strong>uncategorized</strong> - {len(uncategorized)} intents, {sum(c for _, c in uncategorized):,} records</button>
        <div class="content">
            <table>
                <tr><th>Intent</th><th>Count</th></tr>
"""
        for intent, count in sorted(uncategorized, key=lambda x: -x[1])[:30]:
            html += f"<tr><td><code>{intent if intent else '(empty)'}</code></td><td>{count}</td></tr>\n"
        html += "</table></div>\n"
    
    html += "</div>\n"
    
    # Multi-Field Analysis
    html += """
    <h2>📑 Multi-Field Analysis</h2>
    <div class="card">
        <h3>Customer On Hold</h3>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value">""" + f"{multi_field['Customer_On_Hold']['pct_on_hold']:.1f}%" + """</div>
                <div class="stat-label">Calls with Hold</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">""" + f"{multi_field['Customer_On_Hold']['true_count']:,}" + """</div>
                <div class="stat-label">On Hold Count</div>
            </div>
        </div>
    </div>
"""
    
    # Summary Keywords
    if multi_field['Summary_Keywords']['records_with_keyword']:
        html += """
    <div class="card">
        <h3>Billing Keywords in Summaries</h3>
        <p>How often billing-related keywords appear in call summaries.</p>
        <table>
            <tr><th>Keyword Group</th><th>Records Containing</th><th>%</th></tr>
"""
        total_records = overview['total_rows']
        for group, count in sorted(multi_field['Summary_Keywords']['records_with_keyword'].items(), key=lambda x: -x[1]):
            pct = count / total_records * 100
            html += f"<tr><td>{group}</td><td>{count:,}</td><td>{pct:.1f}%</td></tr>\n"
        html += "</table></div>\n"
    
    # Transfer Details
    if not multi_field['Transfer_Details'].get('empty', True):
        html += """
    <div class="card">
        <h3>Transfer Details</h3>
        <table>
            <tr><th>Transfer Type</th><th>Count</th></tr>
"""
        for val, count in list(multi_field['Transfer_Details']['top_values'].items())[:15]:
            html += f"<tr><td>{val}</td><td>{count:,}</td></tr>\n"
        html += "</table></div>\n"
    
    # JavaScript for interactivity
    html += """
    <script>
        // Collapsible sections
        var coll = document.getElementsByClassName("collapsible");
        for (var i = 0; i < coll.length; i++) {
            coll[i].addEventListener("click", function() {
                this.classList.toggle("active");
                var content = this.nextElementSibling;
                if (content.style.display === "block") {
                    content.style.display = "none";
                } else {
                    content.style.display = "block";
                }
            });
        }
        
        // Table filtering
        function filterTable() {
            var input = document.getElementById("intentSearch");
            var filter = input.value.toLowerCase();
            var table = document.getElementById("intentTable");
            var tr = table.getElementsByTagName("tr");
            
            for (var i = 1; i < tr.length; i++) {
                var td = tr[i].getElementsByTagName("td")[1];
                if (td) {
                    var txtValue = td.textContent || td.innerText;
                    if (txtValue.toLowerCase().indexOf(filter) > -1) {
                        tr[i].style.display = "";
                    } else {
                        tr[i].style.display = "none";
                    }
                }
            }
        }
        
        // Table sorting
        function sortTable(n) {
            var table = document.getElementById("intentTable");
            var rows = Array.from(table.rows).slice(1);
            var isAsc = table.getAttribute("data-sort-asc") === "true";
            
            rows.sort(function(a, b) {
                var x = a.cells[n].innerText;
                var y = b.cells[n].innerText;
                
                // Try numeric sort
                var xNum = parseFloat(x.replace(/,/g, '').replace('%', ''));
                var yNum = parseFloat(y.replace(/,/g, '').replace('%', ''));
                
                if (!isNaN(xNum) && !isNaN(yNum)) {
                    return isAsc ? xNum - yNum : yNum - xNum;
                }
                return isAsc ? x.localeCompare(y) : y.localeCompare(x);
            });
            
            rows.forEach(row => table.tBodies[0].appendChild(row));
            table.setAttribute("data-sort-asc", !isAsc);
        }
    </script>
</body>
</html>
"""
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path

# =============================================================================
# MAIN
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Explore intents in call data (v2)")
    parser.add_argument("--input", "-i", default="data/query_results_genai_summaries.csv")
    parser.add_argument("--output", "-o", default="output")
    parser.add_argument("--fuzzy-threshold", type=float, default=0.75, 
                        help="Similarity threshold for fuzzy matching (0-1)")
    
    args = parser.parse_args()
    
    print(f"Loading data from: {args.input}")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df):,} records")
    
    # Parse genai_response
    print("\n[1/6] Parsing genai_response...")
    parsed = df['genai_response'].apply(parse_genai_response)
    parsed_df = pd.DataFrame(parsed.tolist())
    
    # Dataset overview
    print("[2/6] Generating dataset overview...")
    overview = generate_dataset_overview(df, parsed_df)
    
    # Intent frequency analysis
    print("[3/6] Analyzing intent frequencies...")
    intents = parsed_df['Intent'].fillna("")
    intent_counts = Counter(intents)
    intent_analysis = {
        "total_with_intent": len(intents[intents != ""]),
        "unique_intents": len([k for k in intent_counts if k]),
        "frequency": dict(intent_counts.most_common()),
    }
    
    # Fuzzy clustering
    print("[4/6] Finding fuzzy clusters...")
    fuzzy_clusters = find_fuzzy_clusters(intent_counts, threshold=args.fuzzy_threshold)
    
    # Keyword clustering
    print("[5/6] Clustering by keywords...")
    keyword_clusters, uncategorized = cluster_by_keywords(intent_counts)
    
    # Multi-field analysis
    print("[6/6] Analyzing all fields...")
    multi_field = multi_field_analysis(parsed_df)
    
    # Generate outputs
    output_path = Path(args.output)
    output_path.mkdir(exist_ok=True)
    
    # HTML Report
    html_path = output_path / "intent_exploration_report.html"
    generate_html_report(overview, intent_analysis, fuzzy_clusters, keyword_clusters,
                        uncategorized, multi_field, html_path)
    print(f"\nSaved: {html_path}")
    
    # CSV for easy analysis
    freq_df = pd.DataFrame([
        {"intent": k, "count": v, "pct": v/len(df)*100}
        for k, v in intent_counts.most_common()
    ])
    csv_path = output_path / "raw_intent_frequencies.csv"
    freq_df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")
    
    # JSON export
    json_data = {
        "overview": overview,
        "intent_analysis": intent_analysis,
        "fuzzy_clusters": [[{"intent": i, "count": c} for i, c in cluster] for cluster in fuzzy_clusters],
        "keyword_clusters": {k: [{"intent": i, "count": c} for i, c, _ in v] for k, v in keyword_clusters.items()},
        "multi_field": multi_field,
    }
    json_path = output_path / "intent_analysis.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2, default=str)
    print(f"Saved: {json_path}")
    
    # Console summary
    print("\n" + "="*60)
    print("EXPLORATION SUMMARY")
    print("="*60)
    print(f"Total records: {overview['total_rows']:,}")
    print(f"Parse success rate: {overview['parse_success_rate']:.1f}%")
    print(f"Unique intents: {intent_analysis['unique_intents']}")
    print(f"Fuzzy clusters found: {len(fuzzy_clusters)}")
    print(f"Keyword categories: {len(keyword_clusters)}")
    
    print(f"\nTop 10 intents:")
    for i, (intent, count) in enumerate(list(intent_counts.most_common(10)), 1):
        pct = count / len(df) * 100
        display = intent if intent else "(empty)"
        print(f"  {i:2}. {display}: {count:,} ({pct:.1f}%)")
    
    print(f"\n✅ Open {html_path} in browser for interactive report")
    
    return df, parsed_df

if __name__ == "__main__":
    main()
