import requests
import json
import os

# Ruta del archivo donde guardaremos la geometría
OUTPUT_FILE = "data/ap7_geometry.geojson"

# Query Overpass para obtener TODO el trazado oficial de la AP-7
# Incluye ways y relaciones con nombre "Autopista AP-7" o "AP-7"
OVERPASS_QUERY = """
[out:json][timeout:200];
(
  relation["ref"="AP-7"];
  relation["name"="Autopista AP-7"];
  relation["name"="AP-7"];
  way["ref"="AP-7"];
  way["name"="Autopista AP-7"];
  way["name"="AP-7"];
);
(._;>;);
out geom;
"""

def fetch_ap7_geometry():
    print("Solicitando datos a Overpass API...")
    url = "https://overpass-api.de/api/interpreter"
    response = requests.post(url, data={'data': OVERPASS_QUERY})

    if response.status_code != 200:
        raise Exception(f"Error en Overpass: {response.status_code}")

    data = response.json()

    # Convertimos a GeoJSON
    features = []

    for element in data["elements"]:
        if element["type"] == "way" and "geometry" in element:
            coords = [(pt["lon"], pt["lat"]) for pt in element["geometry"]]

            features.append({
                "type": "Feature",
                "properties": {
                    "id": element["id"],
                    "tags": element.get("tags", {})
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            })

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    os.makedirs("data", exist_ok=True)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    print(f"Guardado correctamente en {OUTPUT_FILE}")


if __name__ == "__main__":
    fetch_ap7_geometry()
