"""
Integration tests for the Tomato Disease Detection API.

Before running:
1. Make sure Docker is running: docker-compose up
2. Make sure the API is up at http://localhost:8000
3. Run this script: python test_integration.py

This script sends real images to the API and validates the responses.
"""

import requests
import os
import sys
import json
from pathlib import Path

# ── Configuration ────────────────────────────────────────────────────────────

API_URL = "http://localhost:8000"
PREDICT_URL = f"{API_URL}/predict"
CONFIDENCE_THRESHOLD = 0.70  # Minimum acceptable confidence (70%)

# Map each class folder name to the label the model returns
# These must exactly match what your model was trained on
CLASS_TESTS = [
    {
        "class_name": "Tomato___Early_blight",
        "folder": "td/tomato/val/Tomato___Early_blight",
    },
    {
        "class_name": "Tomato___Late_blight",
        "folder": "td/tomato/val/Tomato___Late_blight",
    },
    {
        "class_name": "Tomato___healthy",
        "folder": "td/tomato/val/Tomato___healthy",
    },
]

# ── Helper functions ──────────────────────────────────────────────────────────

def get_first_image(folder: str) -> str | None:
    """Return the path to the first image file found in a folder."""
    folder_path = Path(folder)
    if not folder_path.exists():
        print(f"  [!] Folder not found: {folder}")
        return None
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        images = list(folder_path.glob(ext))
        if images:
            return str(images[0])
    print(f"  [!] No images found in: {folder}")
    return None


def check_health() -> bool:
    """Test the GET / health check endpoint."""
    print("\n── Test 1: Health Check ─────────────────────────────")
    try:
        response = requests.get(API_URL, timeout=5)
        if response.status_code == 200 and response.json().get("status") == "ok":
            print("  ✓ API is healthy")
            return True
        else:
            print(f"  ✗ Unexpected response: {response.status_code} — {response.text}")
            return False
    except requests.exceptions.ConnectionError:
        print("  ✗ Could not connect to API at", API_URL)
        print("    Is Docker running? Try: docker-compose up")
        return False


def validate_response_format(data: dict) -> list[str]:
    """
    Check that the response JSON has the correct fields.
    Returns a list of error messages (empty = all good).
    """
    errors = []
    required_fields = ["predicted_class", "confidence", "all_probabilities"]

    for field in required_fields:
        if field not in data:
            errors.append(f"Missing field: '{field}'")

    if "confidence" in data:
        if not isinstance(data["confidence"], float):
            errors.append(f"'confidence' should be float, got {type(data['confidence'])}")
        if not (0.0 <= data["confidence"] <= 1.0):
            errors.append(f"'confidence' out of range: {data['confidence']}")

    if "all_probabilities" in data:
        if not isinstance(data["all_probabilities"], dict):
            errors.append("'all_probabilities' should be a dict")
        else:
            prob_sum = sum(data["all_probabilities"].values())
            if not (0.98 <= prob_sum <= 1.02):  # Allow tiny floating point error
                errors.append(f"Probabilities don't sum to 1.0 (got {prob_sum:.4f})")

    return errors


def test_class(class_name: str, folder: str) -> bool:
    """
    Send one image from a class folder to the API and validate the result.
    Returns True if the test passes.
    """
    print(f"\n── Testing class: {class_name} ────────────────────")

    # Step 1: Find an image to send
    image_path = get_first_image(folder)
    if image_path is None:
        print("  ✗ SKIP — no image available")
        return False
    print(f"  Image: {Path(image_path).name}")

    # Step 2: Send the image to the API
    try:
        with open(image_path, "rb") as img_file:
            files = {"file": (Path(image_path).name, img_file, "image/jpeg")}
            response = requests.post(PREDICT_URL, files=files, timeout=30)
    except requests.exceptions.ConnectionError:
        print("  ✗ FAIL — could not connect to API")
        return False
    except Exception as e:
        print(f"  ✗ FAIL — request error: {e}")
        return False

    # Step 3: Check HTTP status code
    if response.status_code != 200:
        print(f"  ✗ FAIL — HTTP {response.status_code}: {response.text}")
        return False

    # Step 4: Parse JSON
    try:
        data = response.json()
    except json.JSONDecodeError:
        print(f"  ✗ FAIL — Response is not valid JSON: {response.text[:200]}")
        return False

    # Step 5: Validate response format
    format_errors = validate_response_format(data)
    if format_errors:
        print("  ✗ FAIL — Response format errors:")
        for err in format_errors:
            print(f"    • {err}")
        return False
    print("  ✓ Response format is valid")

    # Step 6: Print what the model predicted
    predicted = data["predicted_class"]
    confidence = data["confidence"]
    print(f"  Predicted:  {predicted}")
    print(f"  Expected:   {class_name}")
    print(f"  Confidence: {confidence:.1%}")

    # Step 7: Check confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        print(f"  ✗ FAIL — Confidence {confidence:.1%} is below threshold {CONFIDENCE_THRESHOLD:.0%}")
        return False
    print(f"  ✓ Confidence above {CONFIDENCE_THRESHOLD:.0%} threshold")

    # Step 8: Check correct class was predicted
    if predicted != class_name:
        print(f"  ✗ FAIL — Wrong class predicted")
        return False
    print("  ✓ Correct class predicted")

    # Step 9: Show all probabilities
    print("  All probabilities:")
    for cls, prob in sorted(data["all_probabilities"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 20)
        print(f"    {cls:<35} {prob:.1%}  {bar}")

    return True


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Tomato Disease Detection — Integration Tests")
    print("=" * 60)

    results = []

    # Test 1: Health check (if this fails, stop immediately)
    health_ok = check_health()
    if not health_ok:
        print("\n✗ API is not reachable. Stopping tests.")
        sys.exit(1)

    # Tests 2–4: One image per class
    print("\n── Tests 2–4: Class Prediction ──────────────────────")
    for test in CLASS_TESTS:
        passed = test_class(test["class_name"], test["folder"])
        results.append((test["class_name"], passed))

    # Final report
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)
    passed_count = sum(1 for _, ok in results if ok)
    total = len(results)

    for class_name, ok in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {class_name}")

    print(f"\n  {passed_count}/{total} class tests passed")

    if passed_count == total:
        print("\n  🎉 All tests passed! Pipeline is working end-to-end.")
        sys.exit(0)
    else:
        print(f"\n  ⚠ {total - passed_count} test(s) failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()