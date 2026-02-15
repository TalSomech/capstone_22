# GenAI Features Plan

## Objective
Use GenAI (LLM) to extract features from text columns that require language understanding, improving the model's RMSE.

## Current State
- **Test RMSE:** 0.426
- **R²:** 0.135 (model explains ~13.5% of variance)
- **Current text features:** Simple keyword counts (`luxury_count`, `warning_count`, `description_length`)

## When to Use GenAI vs Traditional Methods

| Use GenAI | Use Traditional Methods |
|-----------|------------------------|
| Subjective qualities (tone, professionalism) | Word counts, length |
| Context-dependent meaning | Keyword presence |
| Requires "reading comprehension" | Pattern matching |
| Semantic similarity | Exact string matching |
| Sentiment with nuance/sarcasm | Simple positive/negative word lists |

### Example: Why Current Features Are Limited
- "This place is NOT luxury" → keyword method still counts as luxury
- "Quiet neighborhood but noisy AC" → misses context
- "Basic but charming" → counts as negative, but could be positive

GenAI can understand context and nuance that simple keyword matching cannot.

---

## Text Columns Available for Processing

| Column | Description | Sample |
|--------|-------------|--------|
| `name` | Listing title | "Blueground \| W. Hollywood, nr shops & melrose" |
| `description` | Full listing description | Long text about the property |
| `neighborhood_overview` | Area description | Description of the neighborhood |
| `host_about` | Host bio | Host's self-description |

---

## Proposed GenAI Features

### High Priority (Expected High Impact)

| Feature | Type | Description |
|---------|------|-------------|
| `sentiment_score` | Float (-1 to 1) | Overall sentiment of description |
| `cleanliness_emphasis` | Binary (0/1) | Does description emphasize cleanliness? |
| `professionalism_score` | Int (1-5) | How professional/polished the listing sounds |

### Medium Priority

| Feature | Type | Description |
|---------|------|-------------|
| `hospitality_score` | Int (1-5) | How welcoming/warm the host sounds |
| `accuracy_risk` | Int (1-5) | Red flags/vague language that may lead to disappointment |
| `target_audience` | Categorical | Business/Family/Couples/Solo/Party |

### Lower Priority (Nice to Have)

| Feature | Type | Description |
|---------|------|-------------|
| `unique_selling_points_count` | Int | Count of distinctive features mentioned |
| `location_quality_mentions` | Int (1-5) | How positively location is described |
| `amenities_highlight_score` | Int (1-5) | How well amenities are presented |

---

## Implementation Plan

### Phase 1: Setup
1. Create `genai_features.py` script
2. Set up API client (Claude or OpenAI)
3. Design prompt for feature extraction
4. Test on sample data (10-20 rows)

### Phase 2: Batch Processing
1. Process all training data in batches
2. Save results to `data/processed/genai_features.csv`
3. Cache results to avoid re-processing
4. Handle API errors and rate limits

### Phase 3: Integration
1. Update `preprocess.py` to merge GenAI features
2. Update `feature_template.json` with new features
3. Retrain model with new features
4. Compare RMSE before/after

### Phase 4: Inference Handling
**Option A: Pre-computed only**
- GenAI features only used for training data
- New data uses median/default values
- Pro: No API calls at inference, fast
- Con: New data doesn't benefit from GenAI features

**Option B: Real-time extraction**
- Call API for new uploads
- Pro: Full benefit of GenAI features
- Con: Slow, costly, API dependency

**Recommended: Option A** (for evaluation purposes)

---

## Prompt Design (Draft)

```
Analyze this Airbnb listing and extract the following features.
Return JSON only, no explanation.

Listing Name: {name}
Description: {description}
Host About: {host_about}

Extract:
{
  "sentiment_score": <float from -1 (negative) to 1 (positive)>,
  "professionalism_score": <int 1-5, where 5 is very professional>,
  "cleanliness_emphasis": <0 or 1, whether cleanliness is emphasized>,
  "hospitality_score": <int 1-5, how welcoming the host sounds>,
  "accuracy_risk": <int 1-5, where 5 means high risk of disappointment>
}
```

---

## Cost Estimation

| API | Model | Input (per 1M) | Output (per 1M) | Est. cost for 58K listings |
|-----|-------|----------------|-----------------|---------------------------|
| Anthropic | Claude Haiku | $0.25 | $1.25 | ~$10-15 |
| Anthropic | Claude Sonnet | $3.00 | $15.00 | ~$120-180 |
| OpenAI | GPT-4o-mini | $0.15 | $0.60 | ~$6-10 |
| OpenAI | GPT-4o | $5.00 | $15.00 | ~$200-300 |
| **Nebius** | **Llama-3.1-8B** | **$0.02** | **$0.02** | **~$1-2** |
| Nebius | Llama-3.1-70B | $0.35 | $0.40 | ~$15-25 |
| Nebius | Mixtral-8x22B | $0.40 | $0.40 | ~$20-30 |

**Recommendation:** Use **Nebius Llama-3.1-8B** for lowest cost, or GPT-4o-mini for balance of cost/quality.

## Cost Control Features

The `genai_features.py` script includes several cost control mechanisms:

### 1. Cost Estimation (no API calls)
```bash
python genai_features.py --input data/raw/listingsLA.csv --estimate-only
python genai_features.py --input data/raw/listingsLA.csv --api nebius --estimate-only
```

### 2. Budget Limit
```bash
python genai_features.py --input data/raw/listingsLA.csv --max-cost 5.0
```
Stops processing when $5.00 is reached and saves progress.

### 3. Notifications Every $10
Automatic console notifications when spending reaches $10, $20, $30, etc.

### 4. Test on Sample First
```bash
python genai_features.py --input data/raw/listingsLA.csv --sample 10
python genai_features.py --input data/raw/listingsLA.csv --api nebius --sample 10
```
Process only 10 random rows to verify everything works.

### 5. Checkpointing
```bash
python genai_features.py --input data/raw/listingsLA.csv --checkpoint checkpoint.csv
```
Saves progress periodically. Resume by running the same command again.

### 6. Using Nebius (Cheapest Option)
```bash
# Default model (Llama-3.1-8B, ~$0.02/1M tokens)
python genai_features.py --input data/raw/listingsLA.csv --api nebius --max-cost 2.0

# Specific model
python genai_features.py --api nebius --nebius-model meta-llama/Meta-Llama-3.1-70B-Instruct
```

Available Nebius models:
- `meta-llama/Meta-Llama-3.1-8B-Instruct` (default, cheapest)
- `meta-llama/Meta-Llama-3.1-70B-Instruct`
- `mistralai/Mixtral-8x22B-Instruct-v0.1`
- `Qwen/Qwen2-72B-Instruct`
- `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct`

---

## Files to Create/Modify

### New Files
- `genai_features.py` - Feature extraction script
- `data/processed/genai_features.csv` - Cached GenAI features

### Files to Modify
- `preprocess.py` - Merge GenAI features during preprocessing
- `feature_template.json` - Add new feature defaults
- `requirements.txt` - Add `anthropic` or `openai` package

---

## Success Criteria
- [ ] GenAI features extracted for all training data
- [ ] Features integrated into preprocessing pipeline
- [ ] Model retrained with new features
- [ ] RMSE improved (target: < 0.40)
- [ ] No API calls required at inference time

---

## Questions to Resolve
1. Which API to use? (Claude API / OpenAI / Local model)
2. API key setup ready?
3. Budget for API calls?
