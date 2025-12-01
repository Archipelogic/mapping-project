"""
Generate synthetic customer service call data that mimics the structure of
query_results_genai_summaries.csv for local development and testing.

Produces realistic power-law distribution with:
- High-frequency intents dominating
- Casing/spelling variations
- Long tail of rare intents
"""

import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
import uuid

# Seed for reproducibility
np.random.seed(42)
random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_RECORDS = 1000  # Number of synthetic records to generate

# =============================================================================
# REALISTIC INTENT DISTRIBUTION
# Each intent has variations and a relative frequency weight
# Higher weight = more common
# =============================================================================

INTENT_GROUPS = {
    # HIGH FREQUENCY - These dominate call volume
    "make_payment": {
        "weight": 25,  # ~25% of billing calls
        "variations": [
            "Make payment",
            "Make a payment", 
            "make payment",
            "Make Payment",
            "make a payment",
            "Pay bill",
            "pay bill",
            "Pay my bill",
            "pay my bill",
            "Submit payment",
            "I need to make a payment",
            "Want to pay my bill",
            "making a payment",
            "Payment",
            "pay",
        ]
    },
    "billing_question": {
        "weight": 18,
        "variations": [
            "Billing question",
            "billing question",
            "Question about my bill",
            "question about bill",
            "Bill question",
            "Billing inquiry",
            "billing inquiry",
            "Why is my bill so high",
            "why is my bill high",
            "Bill too high",
            "Understand my bill",
            "understand bill",
            "Explain my bill",
            "Bill explanation",
            "I have a question about my bill",
            "billing",
        ]
    },
    "check_balance": {
        "weight": 12,
        "variations": [
            "Check balance",
            "check balance",
            "Account balance",
            "account balance",
            "What do I owe",
            "what do i owe",
            "How much do I owe",
            "Balance inquiry",
            "balance",
            "Amount due",
            "What's my balance",
            "whats my balance",
        ]
    },
    "payment_issue": {
        "weight": 10,
        "variations": [
            "Payment didn't go through",
            "payment failed",
            "Payment failed",
            "Payment not posted",
            "payment not showing",
            "Double charged",
            "Charged twice",
            "Payment error",
            "payment issue",
            "Payment problem",
            "Can't make payment",
            "cant make payment",
            "Unable to pay",
        ]
    },
    
    # MEDIUM FREQUENCY
    "autopay": {
        "weight": 8,
        "variations": [
            "Set up autopay",
            "setup autopay",
            "Setup automatic payments",
            "Enroll in autopay",
            "enroll autopay",
            "Cancel autopay",
            "cancel autopay",
            "Stop autopay",
            "Autopay question",
            "autopay",
            "Auto pay",
            "automatic payment",
        ]
    },
    "payment_history": {
        "weight": 6,
        "variations": [
            "Payment history",
            "payment history",
            "View past payments",
            "Check payment history",
            "Previous payments",
            "When did I last pay",
            "Last payment",
            "payment records",
        ]
    },
    "dispute": {
        "weight": 5,
        "variations": [
            "Dispute charge",
            "dispute charge",
            "Billing dispute",
            "billing dispute",
            "Wrong charge",
            "wrong charge",
            "Incorrect charge",
            "Overcharged",
            "overcharged",
            "Dispute",
            "charge dispute",
            "I was charged incorrectly",
        ]
    },
    "account_access": {
        "weight": 5,
        "variations": [
            "Can't log in",
            "cant login",
            "Login issues",
            "login problem",
            "Reset password",
            "reset password",
            "Forgot password",
            "Account locked",
            "account locked",
            "Can't access account",
            "cant access account",
            "Access issue",
            "locked out",
        ]
    },
    "payment_method": {
        "weight": 4,
        "variations": [
            "Update payment method",
            "update payment method",
            "Change payment method",
            "Add credit card",
            "add card",
            "New payment method",
            "Remove card",
            "Update card",
            "update card on file",
            "payment method",
        ]
    },
    
    # LOW FREQUENCY - Long tail
    "refund": {
        "weight": 2,
        "variations": [
            "Refund",
            "refund",
            "Request refund",
            "Where is my refund",
            "refund status",
        ]
    },
    "payment_arrangement": {
        "weight": 2,
        "variations": [
            "Payment arrangement",
            "payment plan",
            "Payment extension",
            "Hardship",
            "Can't afford bill",
            "Need more time to pay",
        ]
    },
}

# Non-billing intents with their own distribution
NON_BILLING_GROUPS = {
    "claims": {
        "weight": 15,
        "variations": [
            "File a claim",
            "file claim",
            "Claim status",
            "claim status",
            "Check claim status",
            "Claim inquiry",
            "claim question",
            "Submit claim",
            "Claims",
        ]
    },
    "policy_question": {
        "weight": 12,
        "variations": [
            "Policy question",
            "policy question",
            "Coverage question",
            "coverage",
            "What does my policy cover",
            "Policy inquiry",
        ]
    },
    "policy_change": {
        "weight": 8,
        "variations": [
            "Update policy",
            "Change policy",
            "Add driver",
            "add driver to policy",
            "Remove vehicle",
            "Add vehicle",
            "Policy change",
        ]
    },
    "documents": {
        "weight": 6,
        "variations": [
            "Get ID card",
            "ID card",
            "Proof of insurance",
            "proof of insurance",
            "Insurance card",
            "Need documents",
        ]
    },
    "cancel": {
        "weight": 4,
        "variations": [
            "Cancel policy",
            "cancel policy",
            "Cancel insurance",
            "Want to cancel",
            "Cancellation",
        ]
    },
    "quote": {
        "weight": 5,
        "variations": [
            "Get quote",
            "get a quote",
            "Quote",
            "New quote",
            "Price quote",
        ]
    },
    "general": {
        "weight": 10,
        "variations": [
            "General inquiry",
            "general question",
            "Question",
            "Speak to agent",
            "speak to someone",
            "Need help",
            "Help",
            "Other",
            "other",
            "",  # Empty/missing intent
            None,
        ]
    },
}

# Channels
CHANNELS = ["Phone", "Chat", "Mobile App", "Web"]
CHANNEL_WEIGHTS = [0.6, 0.2, 0.1, 0.1]

# Model attributes (simulating different AI models used)
MODEL_ATTRIBUTES = [
    '"anthropic_claude.claude-3-haiku-20240307"',
    '"anthropic_claude.claude-3-sonnet-20240229"',
]

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def generate_contact_id():
    """Generate a unique contact ID."""
    return str(uuid.uuid4())[:8] + "-" + str(random.randint(1000, 9999))

def generate_call_duration(intent):
    """Generate realistic call duration based on intent complexity."""
    # Billing disputes and complex issues take longer
    if any(word in intent.lower() for word in ["dispute", "wrong", "incorrect", "can't", "locked"]):
        return int(np.random.normal(600, 180))  # ~10 min average
    elif any(word in intent.lower() for word in ["payment", "pay", "autopay"]):
        return int(np.random.normal(300, 90))   # ~5 min average
    else:
        return int(np.random.normal(240, 60))   # ~4 min average

def generate_summary(intent, customer_on_hold, transfer_occurred):
    """Generate a realistic call summary based on intent."""
    
    summaries_by_category = {
        "payment": [
            f"Customer called to {intent.lower()}. The agent was able to assist the customer's account, confirm the payment amount, and process the payment using the customer's card on file. The customer was satisfied with the resolution.",
            f"The customer called State Farm customer care to inquire about {intent.lower()}. The agent verified the customer's information and was able to complete the requested transaction. Customer confirmed receipt of confirmation number.",
            f"Customer contacted support regarding {intent.lower()}. After verifying the account, the agent processed the request successfully. The customer expressed appreciation for the quick service.",
        ],
        "autopay": [
            f"Customer called to {intent.lower()}. The agent explained the autopay options and enrollment process. After confirming the payment method and preferred date, the agent completed the setup. Customer will receive confirmation email.",
            f"The customer requested assistance with {intent.lower()}. The agent navigated to the billing section and walked the customer through the available options. Changes were saved successfully.",
        ],
        "billing_inquiry": [
            f"Customer called with a {intent.lower()}. The agent pulled up the account and reviewed recent charges. The agent explained each line item and the customer understood the billing breakdown.",
            f"The customer contacted support to {intent.lower()}. After reviewing the account, the agent provided a detailed explanation of the current balance and recent transactions.",
        ],
        "dispute": [
            f"Customer called regarding a {intent.lower()}. The agent reviewed the account history and identified the charge in question. A case was opened for further investigation. Customer will be contacted within 5-7 business days.",
            f"The customer disputed a charge on their account. The agent gathered details about the disputed amount and submitted a formal dispute request. Reference number provided to customer.",
        ],
        "account_access": [
            f"Customer called because they {intent.lower()}. The agent verified the customer's identity using security questions and was able to reset the account credentials. Customer confirmed they can now access their account.",
            f"The customer experienced issues trying to {intent.lower()}. After identity verification, the agent unlocked the account and provided instructions for setting up a new password.",
        ],
        "claims": [
            f"Customer called to {intent.lower()}. The agent collected incident details and initiated the claims process. Claim number was provided and customer was informed of next steps.",
            f"The customer contacted claims department regarding {intent.lower()}. The agent reviewed the claim status and provided an update on the current processing stage.",
        ],
        "policy": [
            f"Customer called with a {intent.lower()}. The agent reviewed the current policy details and explained the coverage options. Customer requested time to consider changes.",
            f"The customer requested to {intent.lower()}. After reviewing eligibility and requirements, the agent processed the requested policy modification.",
        ],
        "general": [
            f"Customer called for {intent.lower()}. The agent addressed the inquiry and provided the requested information. Call resolved successfully.",
            f"The customer contacted support regarding {intent.lower()}. The agent assisted with the request and confirmed customer satisfaction before ending the call.",
        ],
    }
    
    # Determine category
    intent_lower = intent.lower()
    if any(word in intent_lower for word in ["pay", "payment", "bill"]) and "autopay" not in intent_lower:
        category = "payment"
    elif "autopay" in intent_lower or "automatic" in intent_lower:
        category = "autopay"
    elif any(word in intent_lower for word in ["question", "inquiry", "understand", "balance", "owe", "why"]):
        category = "billing_inquiry"
    elif any(word in intent_lower for word in ["dispute", "wrong", "incorrect", "overcharged"]):
        category = "dispute"
    elif any(word in intent_lower for word in ["access", "login", "password", "locked"]):
        category = "account_access"
    elif any(word in intent_lower for word in ["claim"]):
        category = "claims"
    elif any(word in intent_lower for word in ["policy", "coverage", "quote", "driver", "vehicle", "renewal", "cancel"]):
        category = "policy"
    else:
        category = "general"
    
    base_summary = random.choice(summaries_by_category.get(category, summaries_by_category["general"]))
    
    # Add hold information if applicable
    if customer_on_hold:
        hold_additions = [
            " The customer was placed on hold briefly while the agent consulted with a supervisor.",
            " Customer was on hold for approximately 3 minutes while the agent researched the issue.",
            " The agent placed the customer on a brief hold to verify information in the system.",
        ]
        base_summary += random.choice(hold_additions)
    
    # Add transfer information if applicable
    if transfer_occurred:
        transfer_additions = [
            " The call was transferred to the billing department for further assistance.",
            " Customer was transferred to a specialist who could better address their needs.",
            " The agent transferred the call to the appropriate department after initial triage.",
        ]
        base_summary += random.choice(transfer_additions)
    
    return base_summary

def generate_action_items(intent, resolved=True):
    """Generate action items based on intent."""
    if resolved:
        return random.choice([
            "None - issue resolved",
            "Customer to monitor account for confirmation",
            "No follow-up required",
            "Customer will call back if issue persists",
        ])
    else:
        return random.choice([
            "Follow up with customer in 3-5 business days",
            "Escalate to supervisor for review",
            "Send confirmation email to customer",
            "Customer to submit additional documentation",
        ])

def generate_genai_response(intent, customer_on_hold, transfer_occurred):
    """Generate the genai_response JSON-like string."""
    
    resolved = random.random() > 0.15  # 85% resolution rate
    summary = generate_summary(intent, customer_on_hold, transfer_occurred)
    action_items = generate_action_items(intent, resolved)
    
    # Create the response structure (matching the real data format)
    response = {
        "Intent": intent,
        "Customer_On_Hold": customer_on_hold,
        "Transfer_Details": "Transferred to billing" if transfer_occurred else "No transfer",
        "ActionItems_Caller": action_items,
        "Summary": summary,
        "Resolution": "Resolved" if resolved else "Pending",
    }
    
    # Format as JSON string (similar to real data)
    return json.dumps(response)

# =============================================================================
# MAIN DATA GENERATION
# =============================================================================

def select_intent_from_groups():
    """
    Select an intent using weighted groups, then pick a random variation.
    Returns (intent_string, is_billing, group_name)
    """
    # Combine all groups with billing flag
    all_groups = []
    weights = []
    
    for group_name, group_data in INTENT_GROUPS.items():
        all_groups.append((group_name, group_data, True))
        weights.append(group_data["weight"])
    
    for group_name, group_data in NON_BILLING_GROUPS.items():
        all_groups.append((group_name, group_data, False))
        weights.append(group_data["weight"])
    
    # Normalize weights
    total = sum(weights)
    weights = [w/total for w in weights]
    
    # Select group
    idx = random.choices(range(len(all_groups)), weights=weights)[0]
    group_name, group_data, is_billing = all_groups[idx]
    
    # Select random variation from group
    intent = random.choice(group_data["variations"])
    
    return intent, is_billing, group_name


def generate_dataset(num_records=NUM_RECORDS):
    """Generate the complete synthetic dataset with realistic distribution."""
    
    records = []
    
    for _ in range(num_records):
        # Select intent using weighted groups
        intent, is_billing, group_name = select_intent_from_groups()
        
        # Handle None/empty intents
        if intent is None or intent == "":
            intent_for_summary = "General inquiry"
        else:
            intent_for_summary = intent
        
        # Determine hold and transfer (more likely for complex issues)
        is_complex = group_name in ["dispute", "payment_issue", "account_access"]
        customer_on_hold = random.random() < (0.4 if is_complex else 0.15)
        transfer_occurred = random.random() < (0.3 if is_complex else 0.1)
        
        # Generate record
        record = {
            "contactid": generate_contact_id(),
            "channel": random.choices(CHANNELS, weights=CHANNEL_WEIGHTS)[0],
            "model_attributes": random.choice(MODEL_ATTRIBUTES),
            "genai_response": generate_genai_response(intent if intent else "", customer_on_hold, transfer_occurred),
            "totalcalldurationinseconds": max(60, generate_call_duration(intent_for_summary)),
        }
        
        records.append(record)
    
    return pd.DataFrame(records)

def main():
    print("Generating synthetic customer service call data...")
    print(f"Number of records: {NUM_RECORDS}")
    
    df = generate_dataset()
    
    # Save to CSV
    output_path = "data/query_results_genai_summaries.csv"
    import os
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nDataset saved to: {output_path}")
    print(f"\nDataset shape: {df.shape}")
    print(f"\nColumn types:\n{df.dtypes}")
    print(f"\nSample record:")
    print(df.iloc[0].to_dict())
    
    # Quick stats
    print(f"\n--- Quick Stats ---")
    print(f"Channels: {df['channel'].value_counts().to_dict()}")
    print(f"Avg call duration: {df['totalcalldurationinseconds'].mean():.1f} seconds")
    
    return df

if __name__ == "__main__":
    main()
