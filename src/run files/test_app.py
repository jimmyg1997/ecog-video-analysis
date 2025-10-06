#!/usr/bin/env python3
"""
Test script for the ECoG Flask Web Application
"""

import requests
import json
import time

def test_flask_app():
    """Test the Flask application endpoints."""
    base_url = "http://localhost:5001"
    
    print("🧪 Testing ECoG Flask Web Application")
    print("=" * 50)
    
    # Test endpoints
    endpoints = [
        ("/", "Home page"),
        ("/data-overview", "Data overview page"),
        ("/feature-extraction", "Feature extraction page"),
        ("/preprocessing", "Preprocessing page"),
        ("/video-annotations", "Video annotations page"),
        ("/ecog-visualization", "ECoG visualization page"),
        ("/results-analysis", "Results analysis page"),
        ("/modelling", "ML Modelling page"),
        ("/methodology", "Methodology page"),
        ("/about", "About page"),
        ("/api/data/overview", "Data overview API"),
        ("/api/annotations", "Annotations API"),
    ]
    
    results = []
    
    for endpoint, description in endpoints:
        try:
            print(f"Testing {description}...", end=" ")
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            
            if response.status_code == 200:
                print("✅ OK")
                results.append((endpoint, "✅ OK", response.status_code))
            else:
                print(f"⚠️  Status: {response.status_code}")
                results.append((endpoint, f"⚠️  {response.status_code}", response.status_code))
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error: {e}")
            results.append((endpoint, f"❌ Error", "Error"))
    
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)
    
    for endpoint, status, code in results:
        print(f"{endpoint:<30} {status}")
    
    # Test API data
    print("\n🔍 Testing API Data Endpoints:")
    print("-" * 30)
    
    try:
        # Test data overview API
        response = requests.get(f"{base_url}/api/data/overview", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Data overview API: {len(data)} fields")
            print(f"   - Channels: {data.get('channels', 'N/A')}")
            print(f"   - Samples: {data.get('samples', 'N/A')}")
            print(f"   - Duration: {data.get('duration_minutes', 'N/A')} minutes")
        else:
            print(f"⚠️  Data overview API: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Data overview API error: {e}")
    
    try:
        # Test annotations API
        response = requests.get(f"{base_url}/api/annotations", timeout=10)
        if response.status_code == 200:
            data = response.json()
            annotations = data.get('annotations', [])
            print(f"✅ Annotations API: {len(annotations)} annotations")
        else:
            print(f"⚠️  Annotations API: Status {response.status_code}")
    except Exception as e:
        print(f"❌ Annotations API error: {e}")
    
    print("\n🎉 Flask application testing completed!")
    print("🌐 Open http://localhost:5001 in your browser to view the application")

if __name__ == "__main__":
    test_flask_app()
