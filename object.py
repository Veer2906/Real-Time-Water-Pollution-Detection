import os
import xml.etree.ElementTree as ET

def count_objects_in_xml_files(folder_path):
    total_objects = 0
    min_bottles = float('inf')  # Initialize to positive infinity
    max_bottles = 0
    num_files = 0

    # Get a list of all XML files in the specified folder
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]

    for xml_file in xml_files:
        # Parse the XML file
        xml_path = os.path.join(folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Count the number of <object> elements (bottles) in the XML file
        num_bottles_in_file = len(root.findall(".//object[name='bottle']"))

        # Update the minimum and maximum counts
        min_bottles = min(min_bottles, num_bottles_in_file)
        max_bottles = max(max_bottles, num_bottles_in_file)

        # Add the count to the total_objects variable
        total_objects += num_bottles_in_file
        num_files += 1

    # Calculate the average number of bottles
    average_bottles = total_objects / num_files

    return min_bottles, max_bottles, average_bottles

if __name__ == "__main__":
    folder_path = r"C:\Users\veera\Desktop\Veer\Coding\Python\Veritas\FloW_IMG\FloW_IMG\test\annotations"
    min_bottles, max_bottles, average_bottles = count_objects_in_xml_files(folder_path)
    
    print("Minimum number of bottles:", min_bottles)
    print("Maximum number of bottles:", max_bottles)
    print("Average number of bottles:", average_bottles)
