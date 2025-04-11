import sys
import json
import cv2
import numpy as np
from PIL import Image
import io

def analyze_food_image(image_path):
    """
    Analyze a food image and return nutrition information.
    This is a simplified version that would be replaced with your actual ML model.
    """
    try:
        # Load the image
        img = cv2.imread(image_path)
        if img is None:
            return {"error": "Failed to load image"}
        
        # In a real implementation, this would use your trained model
        # For now, we'll return simulated data
        food_items = [
            {
                "id": "1",
                "name": "Apple",
                "confidence": 0.92,
                "calories": 95,
                "protein": 0.5,
                "carbs": 25,
                "fat": 0.3,
                "servingSize": 1,
                "servingUnit": "medium apple"
            },
            {
                "id": "2",
                "name": "Banana",
                "confidence": 0.88,
                "calories": 105,
                "protein": 1.3,
                "carbs": 27,
                "fat": 0.4,
                "servingSize": 1,
                "servingUnit": "medium banana"
            }
        ]
        
        return {
            "foodItems": food_items,
            "totalCalories": sum(item["calories"] for item in food_items),
            "totalProtein": sum(item["protein"] for item in food_items),
            "totalCarbs": sum(item["carbs"] for item in food_items),
            "totalFat": sum(item["fat"] for item in food_items)
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No image path provided"}))
        sys.exit(1)
    
    image_path = sys.argv[1]
    result = analyze_food_image(image_path)
    
    # Print the result as JSON
    print(json.dumps(result)) 