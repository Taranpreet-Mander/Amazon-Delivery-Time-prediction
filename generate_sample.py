"""
makes a small SYNTHETIC sample dataset so the training script runs out of the box.

this is NOT the real data - the real "Amazon Delivery" dataset is on kaggle
(search "amazon delivery time prediction"). i just generate fake rows that have
a realistic relationship (further distance / worse traffic / bad weather = slower)
so the model actually has something to learn. real numbers will differ.
"""
import csv
import math
import random

random.seed(42)

WEATHERS = ["Sunny", "Cloudy", "Fog", "Rainy", "Stormy"]
TRAFFIC = ["Low", "Medium", "High", "Jam"]
VEHICLES = ["motorcycle", "scooter", "van"]
AREAS = ["Urban", "Metropolitan", "Semi-Urban"]
CATEGORIES = ["Food", "Electronics", "Clothing", "Grocery", "Books"]

# how much each thing slows delivery down (minutes-ish)
WEATHER_DELAY = {"Sunny": 0, "Cloudy": 3, "Fog": 8, "Rainy": 10, "Stormy": 15}
TRAFFIC_DELAY = {"Low": 0, "Medium": 6, "High": 12, "Jam": 20}
VEHICLE_FACTOR = {"motorcycle": 1.0, "scooter": 1.1, "van": 1.4}


def haversine(lat1, lon1, lat2, lon2):
    # distance between two lat/lon points in km
    R = 6371
    p = math.pi / 180
    a = (0.5 - math.cos((lat2 - lat1) * p) / 2
         + math.cos(lat1 * p) * math.cos(lat2 * p) * (1 - math.cos((lon2 - lon1) * p)) / 2)
    return 2 * R * math.asin(math.sqrt(a))


def main(n=400, out="data/sample_deliveries.csv"):
    base_lat, base_lon = 19.07, 72.87  # roughly mumbai
    fields = ["Order_ID", "Agent_Age", "Agent_Rating",
              "Store_Latitude", "Store_Longitude", "Drop_Latitude", "Drop_Longitude",
              "Weather", "Traffic", "Vehicle", "Area", "Category", "Delivery_Time"]
    rows = []
    for i in range(n):
        age = random.randint(20, 45)
        rating = round(random.uniform(3.0, 5.0), 1)
        slat = base_lat + random.uniform(-0.1, 0.1)
        slon = base_lon + random.uniform(-0.1, 0.1)
        dlat = slat + random.uniform(-0.15, 0.15)
        dlon = slon + random.uniform(-0.15, 0.15)
        weather = random.choice(WEATHERS)
        traffic = random.choice(TRAFFIC)
        vehicle = random.choice(VEHICLES)
        area = random.choice(AREAS)
        cat = random.choice(CATEGORIES)

        dist = haversine(slat, slon, dlat, dlon)
        # the "truth" the model should learn, + some random noise
        t = 10 + dist * 3 * VEHICLE_FACTOR[vehicle]
        t += WEATHER_DELAY[weather] + TRAFFIC_DELAY[traffic]
        t += (5 - rating) * 4 + (age - 30) * 0.1
        t += random.gauss(0, 5)
        t = max(10.0, round(t, 1))

        rows.append([f"ORD{i:04d}", age, rating,
                     round(slat, 5), round(slon, 5), round(dlat, 5), round(dlon, 5),
                     weather, traffic, vehicle, area, cat, t])

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
