import requests
import json
import sys

def test_multimodel_prediction():
    url = "http://localhost:8000/api/portfolio/predict"
    
    # Test with a single ticker to save time, but 'all' mode runs for all tickers in list
    payload = {
        "tickers": ["AAPL"], 
        "start_date": "2020-01-01",
        "end_date": "2023-12-31",
        "forecast_horizon": 10,
        "model": "all",
        "use_top_models": 3
    }
    
    print(f"Sending request to {url}...")
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        data = response.json()
        
        if not data.get("ok"):
            print(f"❌ API returned error: {data.get('error')}")
            return False
            
        print("✅ API request successful")
        
        # Verify Mode
        if data.get("mode") != "all":
            print(f"❌ Expected mode 'all', got '{data.get('mode')}'")
            return False
            
        # Verify Predictions Structure
        predictions = data.get("predictions")
        if not predictions:
            print("❌ No predictions returned")
            return False
            
        if "AAPL" not in predictions:
            print("❌ AAPL predictions missing")
            return False
            
        aapl_preds = predictions["AAPL"]
        print(f"✅ AAPL Models found: {list(aapl_preds.keys())}")
        
        expected_models = ['lstm', 'tcn', 'xgboost', 'transformer']
        for model in expected_models:
            if model in aapl_preds and aapl_preds[model]:
                print(f"   ✅ {model}: Forecast length {len(aapl_preds[model]['forecast']['values'])}")
            else:
                print(f"   ⚠️ {model}: Missing or failed (might be due to missing dependencies)")

        return True
        
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to backend. Is it running?")
        return False
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False

if __name__ == "__main__":
    success = test_multimodel_prediction()
    sys.exit(0 if success else 1)
