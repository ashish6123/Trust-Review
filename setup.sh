#!/bin/bash
# Trust Review — Quick Setup Script

echo "=========================================="
echo "  Trust Review Setup"
echo "=========================================="

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "  Next Steps:"
echo "=========================================="
echo ""
echo "1. Add your datasets to the datasets/ folder:"
echo "   - deceptive-opinion.csv"
echo "   - fake reviews dataset.csv"
echo "   - yelp_data_train.csv"
echo "   - yelp_data_test.csv"
echo ""
echo "2. Run the training pipeline:"
echo "   python -m training.data_pipeline"
echo "   python -m training.train_ml"
echo "   python -m training.compare_models"
echo ""
echo "3. Start the server:"
echo "   uvicorn app.main:app --reload"
echo ""
echo "4. Run tests:"
echo "   pytest tests/ -v"
echo ""
echo "Open http://localhost:8000 in your browser."
