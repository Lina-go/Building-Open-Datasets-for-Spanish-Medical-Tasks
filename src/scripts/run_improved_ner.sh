#!/bin/bash
# Quick start script for improved GPT-NER
# Run everything with one command

set -e  # Exit on error

echo "=========================================="
echo "Improved GPT-NER Pipeline"
echo "Based on: arxiv.org/pdf/2304.10428.pdf"
echo "=========================================="
echo ""

# Configuration
ANATEM_ROOT="${ANATEM_ROOT:-data/AnatEM}"
FORMAT="${FORMAT:-nersuite-spanish}"
KNN_INDEX="${KNN_INDEX:-models/knn_index}"
SAMPLE_SIZE="${SAMPLE_SIZE:-200}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-sentence-transformers/all-MiniLM-L6-v2}"

# Check environment variables
if [ -z "$AZURE_OPENAI_ENDPOINT" ]; then
    echo "❌ ERROR: AZURE_OPENAI_ENDPOINT not set"
    echo "   export AZURE_OPENAI_ENDPOINT='https://your-endpoint.openai.azure.com'"
    exit 1
fi

if [ -z "$AZURE_OPENAI_API_KEY" ]; then
    echo "❌ ERROR: AZURE_OPENAI_API_KEY not set"
    echo "   export AZURE_OPENAI_API_KEY='your-api-key'"
    exit 1
fi

echo "✓ Azure OpenAI configured"
echo ""

# Check if index exists
if [ ! -d "$KNN_INDEX" ]; then
    echo "=========================================="
    echo "STEP 1: Building k-NN Index"
    echo "=========================================="
    echo "This will take ~2-5 minutes..."
    echo ""
    
    python -m src.scripts.build_ner_knn_index \
        --anatem-root "$ANATEM_ROOT" \
        --format "$FORMAT" \
        --output-dir "$KNN_INDEX" \
        --model-name "$EMBEDDING_MODEL" \
        --batch-size 32
    
    echo ""
    echo "✓ k-NN index built successfully"
else
    echo "✓ k-NN index found at $KNN_INDEX"
fi

echo ""
echo "=========================================="
echo "STEP 2: Running Improved Evaluation"
echo "=========================================="
echo "Sample size: $SAMPLE_SIZE"
echo "Strategies: zero_shot, knn_few_shot, knn_few_shot_verified"
echo ""
echo "This will take ~10-20 minutes..."
echo ""

python -m src.scripts.gpt_ner_runner_improved \
    --config configs/gpt_ner_improved.yaml \
    --anatem-root "$ANATEM_ROOT" \
    --format "$FORMAT" \
    --knn-index "$KNN_INDEX" \
    --sample-size "$SAMPLE_SIZE" \
    --strategies zero_shot knn_few_shot knn_few_shot_verified

echo ""
echo "=========================================="
echo "✅ COMPLETE!"
echo "=========================================="
echo ""
echo "Results saved to: results/gpt_ner_improved_*/"
echo ""
echo "Expected improvements:"
echo "  - k-NN few-shot:          +75-100% F1 over baseline"
echo "  - k-NN + verification:    +100-125% F1 over baseline"
echo ""
echo "Check summary CSV for detailed metrics:"
echo "  results/gpt_ner_improved_*/summary/summary.csv"
echo ""