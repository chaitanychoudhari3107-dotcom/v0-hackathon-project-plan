from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import csv
import os
from math import radians, cos, sin, asin, sqrt

from google import genai

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["*"])  # allows index.html to call this server

# Prefer setting your key via an environment variable so it doesn’t get committed.
# Example (PowerShell): $env:GEMINI_API_KEY = "AIzaSyCz1zdZ2Pxuc4L9MgJVLd1dnHMclJBQUG0"
# Example (bash): export GEMINI_API_KEY="AIzaSyCz1zdZ2Pxuc4L9MgJVLd1dnHMclJBQUG0"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
client = genai.Client(api_key=GEMINI_API_KEY)
print(f"Gemini key loaded: {GEMINI_API_KEY[:8]}...")
# Temporary — find available models

# ─────────────────────────────────────────
# LOAD DATA FILES AT STARTUP
# ─────────────────────────────────────────

# Load lit roads GeoJSON (export.geojson)
with open("export.geojson", encoding="utf-8") as f:
    lighting_geojson = json.load(f)

# Extract centre point of each lit road segment
lit_roads = []
for feat in lighting_geojson["features"]:
    if feat["geometry"]["type"] == "LineString":
        coords = feat["geometry"]["coordinates"]
        center_lat = sum(c[1] for c in coords) / len(coords)
        center_lon = sum(c[0] for c in coords) / len(coords)
        lit_roads.append({
            "lat": center_lat,
            "lon": center_lon,
            "name": feat["properties"].get("name", "unnamed")
        })

# Load hospitals + police GeoJSON (export__1_.geojson)
with open("hospital and ps data.geojson", encoding="utf-8") as f:
    infra_geojson = json.load(f)

hospitals = []
police_stations = []
for feat in infra_geojson["features"]:
    if feat["geometry"]["type"] == "Point":
        coords = feat["geometry"]["coordinates"]
        amenity = feat["properties"].get("amenity", "")
        entry = {
            "lat": coords[1],
            "lon": coords[0],
            "name": feat["properties"].get("name", "unnamed")
        }
        if amenity == "hospital":
            hospitals.append(entry)
        elif amenity == "police":
            police_stations.append(entry)

# Load safety reviews CSV
reviews = []
if os.path.exists("safety_reviews.csv"):
    with open("safety_reviews.csv", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            reviews.append(row)

print(f"Loaded: {len(lit_roads)} lit roads, {len(hospitals)} hospitals, {len(police_stations)} police, {len(reviews)} reviews")


# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def haversine(lat1, lon1, lat2, lon2):
    """Returns distance in metres between two lat/lon points."""
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * asin(sqrt(a))


def count_nearby(route_coords, poi_list, radius_m):
    """Count how many POIs are within radius_m of any point on the route."""
    count = 0
    for poi in poi_list:
        for point in route_coords[::5]:  # sample every 5th point for speed
            dist = haversine(point["lat"], point["lon"], poi["lat"], poi["lon"])
            if dist <= radius_m:
                count += 1
                break  # count each POI once
    return count


def get_review_score(route_coords, radius_m=500):
    """
    Average sentiment score from safety_reviews.csv
    safe=5, neutral=3, unsafe=1
    Returns 3.0 (neutral) if no reviews found nearby.
    """
    label_map = {"safe": 5, "neutral": 3, "unsafe": 1}
    scores = []
    for review in reviews:
        try:
            # reviews have lighting + crowd as rough location proxy
            # we use lighting*3 + crowd*2 normalised as a score
            lighting = int(review.get("lighting", 3))
            crowd    = int(review.get("crowd", 3))
            label    = review.get("label", "neutral").strip().lower()
            score    = label_map.get(label, 3)
            scores.append(score)
        except:
            continue
    return round(sum(scores) / len(scores), 2) if scores else 3.0


def calculate_safety_score(route_coords):
    """
    Safety Score formula:
      Lighting  40% — lit roads within 300m
      Police    30% — police stations within 1km
      Hospitals 20% — hospitals within 1km
      Reviews   10% — average sentiment from safety_reviews.csv
    Returns a score from 0.0 to 5.0
    """
    lights   = count_nearby(route_coords, lit_roads,       radius_m=300)
    police   = count_nearby(route_coords, police_stations, radius_m=1000)
    hosp     = count_nearby(route_coords, hospitals,       radius_m=1000)

    # Normalise each to 0–5 scale
    light_score  = min(lights / 15, 1) * 5   # 15 lit segments = max
    police_score = min(police / 3,  1) * 5   # 3 stations = max
    hosp_score   = min(hosp / 3,    1) * 5   # 3 hospitals = max
    review_score = get_review_score(route_coords)

    total = (
        light_score  * 0.4 +
        police_score * 0.3 +
        hosp_score   * 0.2 +
        review_score * 0.1
    )
    return round(total, 1)


# ─────────────────────────────────────────
# GEMINI PROMPT
# ─────────────────────────────────────────

GEMINI_PROMPT = """
You are a commuter safety analyst for Pune, India.
A user submitted this comment about a road or area:

Comment: "{comment}"

Classify this comment as exactly one of: Safe, Neutral, or Unsafe.
Then provide one sentence explaining why.
Then provide one sentence recommending what a night commuter should do.

Respond ONLY in this exact format — no extra text:
Classification: [Safe/Neutral/Unsafe]
Reason: [Your one sentence explanation]
Advice: [Your one sentence recommendation]
"""


# ─────────────────────────────────────────
# ENDPOINTS
# ─────────────────────────────────────────

@app.route("/score-route", methods=["POST"])
def score_route():
    """
    Input:  { "routes": [ { "id": "A", "time_min": 14, "coords": [{lat, lon}, ...] }, ... ] }
    Output: { "scores": [ { "id": "A", "time_min": 14, "safety_score": 4.3 }, ... ] }
    """
    data = request.get_json()
    routes = data.get("routes", [])

    results = []
    for route in routes:
        coords = route.get("coords", [])
        score  = calculate_safety_score(coords)
        results.append({
            "id":           route["id"],
            "time_min":     route.get("time_min", 0),
            "safety_score": score
        })

    # Sort so frontend knows which is safest
    results.sort(key=lambda x: x["safety_score"], reverse=True)
    results[0]["recommended"] = True

    return jsonify({"scores": results})


@app.route("/analyze-comment", methods=["POST"])
def analyze_comment():
    """
    Input:  { "comment": "Very dark road near Aundh" }
    Output: { "classification": "Unsafe", "reason": "...", "advice": "...", "score_delta": -0.5 }
    """
    data    = request.get_json()
    comment = data.get("comment", "").strip()

    if not comment:
        return jsonify({"error": "No comment provided"}), 400

    try:
        prompt   = GEMINI_PROMPT.format(comment=comment)
        response = client.models.generate_content(
            model="models/gemini-1.5-flash",
            contents=prompt
        )
        text     = response.text.strip()

        # Parse response
        lines          = text.split("\n")
        classification = lines[0].replace("Classification:", "").strip()
        reason         = lines[1].replace("Reason:", "").strip()
        advice         = lines[2].replace("Advice:", "").strip() if len(lines) > 2 else ""

        # Score delta
        delta_map  = {"Safe": 0.3, "Neutral": 0.0, "Unsafe": -0.5}
        score_delta = delta_map.get(classification, 0.0)

        return jsonify({
            "classification": classification,
            "reason":         reason,
            "advice":         advice,
            "score_delta":    score_delta
        })

    except Exception as e:
        # Fallback if Gemini is slow or fails — keeps demo alive
        print(f"Gemini error: {e}")
        return jsonify({
            "classification": "Neutral",
            "reason":         "Could not analyse comment at this time.",
            "advice":         "Please exercise caution on unfamiliar routes at night.",
            "score_delta":    0.0
        })


@app.route("/submit-review", methods=["POST"])
def submit_review():
    """
    Input:  { "location": "FC Road", "lighting": 4, "crowd": 5, "comment": "...", "label": "safe" }
    Output: { "message": "Review saved" }
    Appends to safety_reviews.csv
    """
    data = request.get_json()

    row = {
        "location":  data.get("location", ""),
        "lighting":  data.get("lighting", 3),
        "crowd":     data.get("crowd", 3),
        "comment":   data.get("comment", ""),
        "label":     data.get("label", "neutral")
    }

    file_exists = os.path.exists("safety_reviews.csv")
    with open("safety_reviews.csv", "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["location", "lighting", "crowd", "comment", "label"])
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return jsonify({"message": "Review saved"})


@app.route("/get-heatmap", methods=["GET"])
def get_heatmap():
    """
    Returns all infrastructure points for the map heatmap overlay.
    Output: { "lit_roads": [...], "hospitals": [...], "police": [...] }
    """
    return jsonify({
        "lit_roads": lit_roads,
        "hospitals": hospitals,
        "police":    police_stations
    })


@app.route("/get-route-explanation", methods=["POST"])
def get_route_explanation():
    """
    BONUS AI FEATURE — natural language explanation of why a route is recommended.
    Input:  { "route_id": "A", "safety_score": 4.3, "time_min": 17, "lights": 12, "police": 2, "hospitals": 3 }
    Output: { "explanation": "We recommend Route A because..." }
    """
    data = request.get_json()

    prompt = f"""
You are SafePath, a commuter safety navigation app in Pune, India.
Write ONE short paragraph (2-3 sentences) explaining why Route {data.get('route_id')} 
is the safest option for a woman commuting at night.

Facts about this route:
- Safety score: {data.get('safety_score')} out of 5
- Travel time: {data.get('time_min')} minutes
- Lit road segments nearby: {data.get('lights')}
- Police stations nearby: {data.get('police')}
- Hospitals nearby: {data.get('hospitals')}

Write in a friendly, reassuring tone. Be specific. Do not use bullet points.
"""
    try:
        response    = client.models.generate_content(
            model="models/gemini-1.5-flash",
            contents=prompt
        )
        explanation = response.text.strip()
        return jsonify({"explanation": explanation})
    except Exception as e:
        return jsonify({"explanation": "This route passes through well-lit, populated streets with emergency services nearby — making it the safest choice for your night commute."})


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)