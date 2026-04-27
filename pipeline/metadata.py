import xml.etree.ElementTree as ET

def parse_xml(path):

    tree = ET.parse(path)
    root = tree.getroot()

    def get(tag):
        el = root.find(tag)
        return el.text.strip() if el is not None and el.text else None

    return {
        "ClassId": get("ClassId"),
        "Type": get("Type"),
        "Date": get("Date"),
        "Latitude": root.findtext("GPSLocality/Latitude"),
        "Longitude": root.findtext("GPSLocality/Longitude"),
    }