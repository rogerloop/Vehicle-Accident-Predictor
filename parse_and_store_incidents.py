import requests
import xml.etree.ElementTree as ET
import re
import pandas as pd
import os
import datetime

RSS_ULR = "http://www.gencat.cat/transit/opendata/incidenciesRSS.xml"
OUTPUT_DIR = "data/incidents_log/" # Directory to store incident logs

def fecth_incidents():
    """Fetch incidents from the RSS feed."""
    try:
        response = requests.get(RSS_ULR, timeout=10) # 10 seconds timeout
        response.raise_for_status() # Raise an error for bad responses
        return response.content
    
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    
def parse_pk_string(pk_text):
    """Parse the kilometer point from text.
    E.g.: 'Punt km. 477-497' --> (477.0, 497.0)"""
    
    if pk_text is None:
        return None, None
    
    match = re.search(r'Punt km\.\s*([\d\.,]+)(?:-([\d\.,]+))?', pk_text)
    
    if match:
        pk_inici_str = match.group(1).replace(',', '.')
        pk_inici = float(pk_inici_str)
        pk_fi_str = match.group(2)
        
        if pk_fi_str:
            # If there is a second number, parse it
            pk_fi_str = pk_fi_str.replace(',', '.')
            pk_fi = float(pk_fi_str)
        else:
            pk_fi = pk_inici # If not, set pk_fi equal to pk_inici

        return pk_inici, pk_fi
    
    return None, None # Not found the pattern 'Punt km'

def parse_title(title_text):
    """Extract the type of incident from the title."""
    
    if title_text is None:
        return "Unknown"
    
    # Search the text inside the first parentheses
    match = re.search(r'\((.*?)\)', title_text)
    if match:
        return match.group(1).strip()
    else:
        # If there is no parentheses, return the first part of the title
        return title_text.split('.')[0].strip()
    
def parse_incidents(xml_data):
    """Parse the XML data and extract incidents into a DataFrame."""
    
    if xml_data is None:
        return pd.DataFrame() # Return empty DataFrame if no data
    
    root = ET.fromstring(xml_data)
    incidents = []

    # Timestamp of data retrieval
    timestamp_query = datetime.datetime.now()

    namespaces = {'dc': 'http://purl.org/dc/elements/1.1/'} # Define namespace

    for item in root.findall('.//item'):
        title = item.find('title').text if item.find('title') is not None else ''
        description = item.find('description').text if item.find('description') is not None else ''
        pub_date = item.find('pubDate').text if item.find('pubDate') is not None else ''
        guid = item.find('guid').text if item.find('guid') is not None else ''
                
        # Road (from the Description)
        via = description.split('|')[0].strip() if '|' in description else 'Unknown'

        # Only parse via 'AP-7'
        if via != 'AP-7':
            continue
        
        # Type of incident (from the Title)
        incident_type = parse_title(title)        
        
        # Parse kilometer points
        pk_inici, pk_fi = parse_pk_string(description)

        incidents.append({
            'timestamp_query': timestamp_query,
            'guid': guid,
            'pubDate': pub_date,
            'incident_type': incident_type,
            'via': via,
            'pk_inici': pk_inici,
            'pk_fi': pk_fi,
            'title': title,
            'description': description
        })
    
    return pd.DataFrame(incidents)

def save_to_csv(df, output_dir):
    """Save the DataFrame to a CSV file with a timestamped filename."""
    
    if df.empty:
        print("No incidents to save.")
        return
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"incidents_{timestamp_str}.csv")
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Incidents saved to {output_path}")

if __name__ == "__main__":
    print("Fetching incidents from RSS feed...")
    xml_data = fecth_incidents()
    incidents_df = parse_incidents(xml_data)
    save_to_csv(incidents_df, OUTPUT_DIR)