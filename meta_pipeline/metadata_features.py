import datetime

def parse_date(date_str):
    try:
        d = datetime.datetime.strptime(date_str, "%d/%m/%y")
        return d.month
    except:
        return 0

def extract_metadata_features(meta):

    lat = float(meta.get("Latitude", 0))
    lon = float(meta.get("Longitude", 0))

    date = meta.get("Date", "01/01/00")
    month = parse_date(date)

    type_map = {
        "Scan": 0,
        "pseudoscan": 1,
        "photograph": 2
    }

    img_type = type_map.get(meta.get("Type", ""), -1)

    return {
        "lat": lat,
        "lon": lon,
        "month": month,
        "img_type": img_type
    }