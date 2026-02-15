# genai_features.py
"""
GenAI Feature Extraction for Airbnb Listings.

Extracts features from text columns (name, description, host_about) using LLM.
Features are designed to capture subjective qualities that keyword matching cannot.

Usage:
    python genai_features.py --input data/raw/listingsLA.csv --output data/processed/genai_features_LA.csv
    python genai_features.py --input data/raw/listingsNYC.csv --output data/processed/genai_features_NYC.csv

Cost Control:
    - Use --max-cost to set a budget limit (e.g., --max-cost 5.0 for $5 max)
    - Notifications are printed every $10 spent
    - Use --sample for testing on a small subset first
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_INPUT = "data/raw/listingsLA.csv"
DEFAULT_OUTPUT = "data/processed/genai_features.csv"
BATCH_SIZE = 10  # Number of listings to process per API call
RATE_LIMIT_DELAY = 1  # Seconds to wait between API calls
COST_NOTIFICATION_INTERVAL = 10.0  # Notify every $10 spent

# Columns to extract text from
TEXT_COLUMNS = ["name", "description", "host_about"]

# =============================================================================
# Cost Tracking (prices per 1M tokens as of 2024)
# =============================================================================
PRICING = {
    # Anthropic
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    # OpenAI
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 5.00, "output": 15.00},
    # Nebius (prices may vary - update as needed)
    "meta-llama/Meta-Llama-3.1-70B-Instruct": {"input": 0.35, "output": 0.40},
    "meta-llama/Meta-Llama-3.1-8B-Instruct": {"input": 0.02, "output": 0.02},
    "mistralai/Mixtral-8x22B-Instruct-v0.1": {"input": 0.40, "output": 0.40},
    "Qwen/Qwen2-72B-Instruct": {"input": 0.35, "output": 0.40},
    "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct": {"input": 0.02, "output": 0.02},
}


class CostTracker:
    """Track API costs and notify at intervals."""

    def __init__(self, max_cost: float = None, notification_interval: float = 10.0):
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.max_cost = max_cost
        self.notification_interval = notification_interval
        self.last_notification_threshold = 0.0
        self.api_calls = 0

    def add_usage(self, model: str, input_tokens: int, output_tokens: int):
        """Add token usage and calculate cost."""
        pricing = PRICING.get(model, {"input": 1.0, "output": 1.0})

        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        call_cost = input_cost + output_cost

        self.total_cost += call_cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.api_calls += 1

        # Check for notification threshold
        current_threshold = int(self.total_cost / self.notification_interval) * self.notification_interval
        if current_threshold > self.last_notification_threshold:
            self.last_notification_threshold = current_threshold
            print(f"\n{'='*60}")
            print(f"COST NOTIFICATION: ${self.total_cost:.2f} spent so far!")
            print(f"  API calls: {self.api_calls}")
            print(f"  Input tokens: {self.total_input_tokens:,}")
            print(f"  Output tokens: {self.total_output_tokens:,}")
            print(f"{'='*60}\n")

        # Check budget limit
        if self.max_cost and self.total_cost >= self.max_cost:
            raise BudgetExceededError(
                f"Budget limit of ${self.max_cost:.2f} exceeded! "
                f"Current cost: ${self.total_cost:.2f}"
            )

        return call_cost

    def get_summary(self) -> str:
        """Return cost summary string."""
        return (
            f"\nCost Summary:\n"
            f"  Total API calls: {self.api_calls}\n"
            f"  Total input tokens: {self.total_input_tokens:,}\n"
            f"  Total output tokens: {self.total_output_tokens:,}\n"
            f"  Total cost: ${self.total_cost:.4f}\n"
        )


class BudgetExceededError(Exception):
    """Raised when the budget limit is exceeded."""
    pass

# =============================================================================
# Prompt Template
# =============================================================================
EXTRACTION_PROMPT = """Analyze this Airbnb listing and extract features. Return JSON only.

Listing Name: {name}
Description: {description}
Host About: {host_about}

Extract these features:
{{
  "sentiment_score": <float -1.0 to 1.0, overall tone of listing>,
  "professionalism_score": <int 1-5, how polished/professional the writing is>,
  "cleanliness_emphasis": <int 0 or 1, whether cleanliness is emphasized>,
  "hospitality_score": <int 1-5, how welcoming/warm the host sounds>,
  "accuracy_risk": <int 1-5, risk of guest disappointment, 5=high risk>
}}

Rules:
- If text is empty or missing, use neutral defaults: sentiment=0, scores=3, cleanliness=0, risk=3
- Be consistent across listings
- Return ONLY valid JSON, no explanation
"""


# =============================================================================
# API Clients
# =============================================================================
def get_anthropic_client():
    """Initialize Anthropic client."""
    try:
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        return Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError("anthropic package not installed. Run: pip install anthropic")


def get_openai_client():
    """Initialize OpenAI client."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment")
        return OpenAI(api_key=api_key)
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")


def get_nebius_client():
    """Initialize Nebius client (OpenAI-compatible API)."""
    try:
        from openai import OpenAI
        api_key = os.getenv("NEBIUS_API_KEY")
        if not api_key:
            raise ValueError("NEBIUS_API_KEY not found in environment")
        # Nebius uses OpenAI-compatible API
        return OpenAI(
            api_key=api_key,
            base_url="https://api.studio.nebius.ai/v1"
        )
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")


# =============================================================================
# Feature Extraction Functions
# =============================================================================
def extract_features_anthropic(
    client, name: str, description: str, host_about: str, cost_tracker: CostTracker
) -> dict:
    """Extract features using Claude API."""
    model = "claude-3-haiku-20240307"  # Cost-efficient model

    prompt = EXTRACTION_PROMPT.format(
        name=name or "N/A",
        description=description or "N/A",
        host_about=host_about or "N/A"
    )

    response = client.messages.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    # Track costs
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost_tracker.add_usage(model, input_tokens, output_tokens)

    # Parse JSON from response
    response_text = response.content[0].text.strip()
    return parse_json_response(response_text)


def extract_features_openai(
    client, name: str, description: str, host_about: str, cost_tracker: CostTracker
) -> dict:
    """Extract features using OpenAI API."""
    model = "gpt-4o-mini"  # Cost-efficient model

    prompt = EXTRACTION_PROMPT.format(
        name=name or "N/A",
        description=description or "N/A",
        host_about=host_about or "N/A"
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    # Track costs
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_tracker.add_usage(model, input_tokens, output_tokens)

    # Parse JSON from response
    response_text = response.choices[0].message.content.strip()
    return parse_json_response(response_text)


# Default Nebius model (cost-efficient)
DEFAULT_NEBIUS_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"


def extract_features_nebius(
    client, name: str, description: str, host_about: str, cost_tracker: CostTracker,
    model: str = None
) -> dict:
    """Extract features using Nebius API (OpenAI-compatible)."""
    model = model or DEFAULT_NEBIUS_MODEL

    prompt = EXTRACTION_PROMPT.format(
        name=name or "N/A",
        description=description or "N/A",
        host_about=host_about or "N/A"
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )

    # Track costs
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens
    cost_tracker.add_usage(model, input_tokens, output_tokens)

    # Parse JSON from response
    response_text = response.choices[0].message.content.strip()
    return parse_json_response(response_text)


def parse_json_response(response_text: str) -> dict:
    """Parse JSON from LLM response, handling common issues."""
    # Remove markdown code blocks if present
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(lines[1:-1])

    try:
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Return defaults if parsing fails
        return get_default_features()


def get_default_features() -> dict:
    """Return default feature values."""
    return {
        "sentiment_score": 0.0,
        "professionalism_score": 3,
        "cleanliness_emphasis": 0,
        "hospitality_score": 3,
        "accuracy_risk": 3
    }


# =============================================================================
# Single Listing Extraction (for Streamlit App)
# =============================================================================
def extract_single_listing_features(
    name: str,
    description: str,
    host_about: str,
    api_key: str,
    api: str = "openai"
) -> dict:
    """
    Extract GenAI features for a single listing.

    This function is designed for real-time use in the Streamlit app.
    Users provide their own API key - no environment variables needed.

    Args:
        name: Listing name/title
        description: Listing description
        host_about: Host's "about" text
        api_key: User's API key (OpenAI or Anthropic)
        api: Which API to use ("openai" or "anthropic")

    Returns:
        Dict with 5 GenAI features, or defaults if extraction fails
    """
    if not api_key:
        return get_default_features()

    # Truncate long texts to save tokens
    description = (description or "")[:2000]
    host_about = (host_about or "")[:500]
    name = name or ""

    prompt = EXTRACTION_PROMPT.format(
        name=name or "N/A",
        description=description or "N/A",
        host_about=host_about or "N/A"
    )

    try:
        if api == "openai":
            from openai import OpenAI
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip()

        elif api == "anthropic":
            from anthropic import Anthropic
            client = Anthropic(api_key=api_key)

            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.content[0].text.strip()

        elif api == "nebius":
            from openai import OpenAI
            client = OpenAI(
                api_key=api_key,
                base_url="https://api.studio.nebius.ai/v1"
            )

            response = client.chat.completions.create(
                model=DEFAULT_NEBIUS_MODEL,
                max_tokens=200,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = response.choices[0].message.content.strip()
        else:
            return get_default_features()

        return parse_json_response(response_text)

    except Exception as e:
        print(f"GenAI extraction error: {e}")
        return get_default_features()


# =============================================================================
# Batch Processing
# =============================================================================
def process_dataframe(
    df: pd.DataFrame,
    api: str = "anthropic",
    start_idx: int = 0,
    max_rows: int = None,
    max_cost: float = None,
    checkpoint_path: str = None,
    nebius_model: str = None
) -> pd.DataFrame:
    """
    Process DataFrame and extract GenAI features for each row.

    Args:
        df: Input DataFrame with text columns
        api: Which API to use ("anthropic", "openai", or "nebius")
        start_idx: Row index to start from (for resuming)
        max_rows: Maximum number of rows to process (None = all)
        max_cost: Maximum cost in USD before stopping (None = no limit)
        checkpoint_path: Path to save checkpoints
        nebius_model: Specific model to use with Nebius API

    Returns:
        DataFrame with extracted features
    """
    # Initialize cost tracker
    cost_tracker = CostTracker(
        max_cost=max_cost,
        notification_interval=COST_NOTIFICATION_INTERVAL
    )

    # Initialize API client
    if api == "anthropic":
        client = get_anthropic_client()
        extract_fn = lambda n, d, h: extract_features_anthropic(client, n, d, h, cost_tracker)
    elif api == "openai":
        client = get_openai_client()
        extract_fn = lambda n, d, h: extract_features_openai(client, n, d, h, cost_tracker)
    elif api == "nebius":
        client = get_nebius_client()
        extract_fn = lambda n, d, h: extract_features_nebius(client, n, d, h, cost_tracker, nebius_model)
    else:
        raise ValueError(f"Unknown API: {api}. Choose from: anthropic, openai, nebius")

    # Determine rows to process
    end_idx = len(df) if max_rows is None else min(start_idx + max_rows, len(df))

    # Initialize results list
    results = []

    # Load checkpoint if exists
    if checkpoint_path and Path(checkpoint_path).exists():
        checkpoint_df = pd.read_csv(checkpoint_path)
        results = checkpoint_df.to_dict("records")
        start_idx = len(results)
        print(f"Resuming from checkpoint at row {start_idx}")

    print(f"Processing rows {start_idx} to {end_idx} using {api} API...")
    if max_cost:
        print(f"Budget limit: ${max_cost:.2f}")

    try:
        for idx in range(start_idx, end_idx):
            row = df.iloc[idx]

            # Extract text columns
            name = str(row.get("name", "")) if pd.notna(row.get("name")) else ""
            description = str(row.get("description", "")) if pd.notna(row.get("description")) else ""
            host_about = str(row.get("host_about", "")) if pd.notna(row.get("host_about")) else ""

            # Truncate long texts to save tokens
            description = description[:2000] if len(description) > 2000 else description
            host_about = host_about[:500] if len(host_about) > 500 else host_about

            try:
                features = extract_fn(name, description, host_about)
            except BudgetExceededError:
                raise  # Re-raise budget errors
            except Exception as e:
                print(f"Error at row {idx}: {e}")
                features = get_default_features()

            # Add row identifier
            features["original_index"] = idx
            if "id" in df.columns:
                features["listing_id"] = row["id"]

            results.append(features)

            # Progress update
            if (idx + 1) % 100 == 0:
                print(f"Processed {idx + 1}/{end_idx} rows | Cost so far: ${cost_tracker.total_cost:.4f}")

                # Save checkpoint
                if checkpoint_path:
                    pd.DataFrame(results).to_csv(checkpoint_path, index=False)
                    print(f"Checkpoint saved to {checkpoint_path}")

            # Rate limiting
            time.sleep(RATE_LIMIT_DELAY)

    except BudgetExceededError as e:
        print(f"\n{e}")
        print("Saving progress before stopping...")
        if checkpoint_path:
            pd.DataFrame(results).to_csv(checkpoint_path, index=False)
            print(f"Progress saved to {checkpoint_path}")

    # Print cost summary
    print(cost_tracker.get_summary())

    return pd.DataFrame(results)


# =============================================================================
# Cost Estimation
# =============================================================================
def estimate_cost(df: pd.DataFrame, api: str = "anthropic", nebius_model: str = None) -> dict:
    """
    Estimate the cost of processing a DataFrame without making API calls.

    Args:
        df: Input DataFrame
        api: Which API to use
        nebius_model: Specific model for Nebius API

    Returns:
        Dictionary with cost estimates
    """
    # Estimate tokens per listing (based on typical text lengths)
    avg_input_tokens_per_listing = 600  # prompt + text
    avg_output_tokens_per_listing = 100  # JSON response

    total_input_tokens = len(df) * avg_input_tokens_per_listing
    total_output_tokens = len(df) * avg_output_tokens_per_listing

    # Determine model based on API
    if api == "anthropic":
        model = "claude-3-haiku-20240307"
    elif api == "openai":
        model = "gpt-4o-mini"
    elif api == "nebius":
        model = nebius_model or DEFAULT_NEBIUS_MODEL
    else:
        model = "unknown"

    pricing = PRICING.get(model, {"input": 0.10, "output": 0.10})

    input_cost = (total_input_tokens / 1_000_000) * pricing["input"]
    output_cost = (total_output_tokens / 1_000_000) * pricing["output"]
    total_cost = input_cost + output_cost

    return {
        "rows": len(df),
        "model": model,
        "estimated_input_tokens": total_input_tokens,
        "estimated_output_tokens": total_output_tokens,
        "estimated_cost": total_cost
    }


# =============================================================================
# Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Extract GenAI features from Airbnb listings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Nebius models:
  - meta-llama/Meta-Llama-3.1-8B-Instruct (default, cheapest)
  - meta-llama/Meta-Llama-3.1-70B-Instruct
  - mistralai/Mixtral-8x22B-Instruct-v0.1
  - Qwen/Qwen2-72B-Instruct
  - deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct

Examples:
  # Estimate cost only
  python genai_features.py --input data/raw/listingsLA.csv --estimate-only

  # Use Nebius with budget limit
  python genai_features.py --input data/raw/listingsLA.csv --api nebius --max-cost 5.0

  # Use specific Nebius model
  python genai_features.py --api nebius --nebius-model meta-llama/Meta-Llama-3.1-70B-Instruct
        """
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Input CSV file path")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output CSV file path")
    parser.add_argument("--api", choices=["anthropic", "openai", "nebius"], default="anthropic",
                        help="API to use (default: anthropic)")
    parser.add_argument("--nebius-model", default=None,
                        help=f"Model to use with Nebius API (default: {DEFAULT_NEBIUS_MODEL})")
    parser.add_argument("--start", type=int, default=0, help="Starting row index")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum rows to process")
    parser.add_argument("--max-cost", type=float, default=None, help="Maximum cost in USD before stopping")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint file for resuming")
    parser.add_argument("--sample", type=int, default=None, help="Process random sample of N rows (for testing)")
    parser.add_argument("--estimate-only", action="store_true", help="Only estimate cost, don't process")

    args = parser.parse_args()

    # Load input data
    print(f"Loading {args.input}...")
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} rows")

    # Sample if requested (for testing)
    if args.sample:
        df = df.sample(n=args.sample, random_state=42).reset_index(drop=True)
        print(f"Sampled {args.sample} rows for testing")

    # Estimate cost
    estimate = estimate_cost(df, args.api, args.nebius_model)
    print(f"\nCost Estimate:")
    print(f"  API: {args.api}")
    print(f"  Model: {estimate['model']}")
    print(f"  Rows to process: {estimate['rows']:,}")
    print(f"  Estimated input tokens: {estimate['estimated_input_tokens']:,}")
    print(f"  Estimated output tokens: {estimate['estimated_output_tokens']:,}")
    print(f"  Estimated total cost: ${estimate['estimated_cost']:.2f}")

    if args.estimate_only:
        print("\n--estimate-only flag set. Exiting without processing.")
        return

    # Confirm before processing
    if estimate['estimated_cost'] > 1.0 and not args.max_cost:
        print(f"\nWARNING: Estimated cost is ${estimate['estimated_cost']:.2f}")
        print("Consider using --max-cost to set a budget limit.")
        response = input("Continue? (yes/no): ")
        if response.lower() not in ["yes", "y"]:
            print("Aborted.")
            return

    # Process
    results_df = process_dataframe(
        df,
        api=args.api,
        start_idx=args.start,
        max_rows=args.max_rows,
        max_cost=args.max_cost,
        checkpoint_path=args.checkpoint,
        nebius_model=args.nebius_model
    )

    # Save results
    if len(results_df) > 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        print(f"Saved {len(results_df)} rows to {args.output}")

        # Summary
        print("\nFeature Summary:")
        for col in ["sentiment_score", "professionalism_score", "cleanliness_emphasis",
                    "hospitality_score", "accuracy_risk"]:
            if col in results_df.columns:
                print(f"  {col}: mean={results_df[col].mean():.2f}, std={results_df[col].std():.2f}")


if __name__ == "__main__":
    main()
