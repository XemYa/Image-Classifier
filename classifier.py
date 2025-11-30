# Classify image as dog or cat using Google Cloud Vision API
image_url = params['image_url']
oauth_account_id = params['google_cloud_vision_account']

# Call Google Cloud Vision API with LABEL_DETECTION feature
integration = Integration("google_cloud_vision_api", oauth_account_id)

response = integration.api_call(
    method="POST",
    base_url="https://vision.googleapis.com",
    path="/v1/images:annotate",
    body={
        "requests": [
            {
                "image": {
                    "source": {
                        "imageUri": image_url
                    }
                },
                "features": [
                    {
                        "type": "LABEL_DETECTION",
                        "maxResults": 10
                    }
                ]
            }
        ]
    }
)

# Extract labels from response
labels = response["responses"][0]["labelAnnotations"]

# Find dog and cat labels
dog_label = None
cat_label = None

for label in labels:
    description = label["description"].lower()
    confidence = label["score"]
    
    if "dog" in description:
        dog_label = {"description": label["description"], "confidence": confidence}
    elif "cat" in description:
        cat_label = {"description": label["description"], "confidence": confidence}

# Determine classification
classification = "Unknown"
confidence = 0
matched_label = None

if dog_label and cat_label:
    if dog_label["confidence"] > cat_label["confidence"]:
        classification = "Dog"
        confidence = dog_label["confidence"]
        matched_label = dog_label["description"]
    else:
        classification = "Cat"
        confidence = cat_label["confidence"]
        matched_label = cat_label["description"]
elif dog_label:
    classification = "Dog"
    confidence = dog_label["confidence"]
    matched_label = dog_label["description"]
elif cat_label:
    classification = "Cat"
    confidence = cat_label["confidence"]
    matched_label = cat_label["description"]

all_labels = [{"description": label["description"], "confidence": label["score"]} for label in labels]

results = {
    "classification": classification,
    "confidence": round(confidence, 4),
    "matched_label": matched_label,
    "all_detected_labels": all_labels,
    "dog_detected": dog_label is not None,
    "cat_detected": cat_label is not None
}

return results
