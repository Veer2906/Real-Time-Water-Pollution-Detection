import matplotlib.pyplot as plt
import os 
import xml.etree.ElementTree as ET

def count_objects(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    object_count= len(root.findall(".//object"))
    return object_count

def plot(xml_folder):
    xml_files = [f for f in os.listdir(xml_folder) if f.endswith('.xml')]
    object_counts = []

    for xml_file in xml_files:
        xml_path = os.path.join(xml_folder, xml_file)
        object_count = count_objects(xml_path)
        object_counts.append(object_count)

    plt.hist(object_counts, bins=range(min(object_counts), max(object_counts) + 1))
    plt.xlabel('Number of objects')
    plt.ylabel('Frequency')
    plt.title('Number of Objects in Each XML File')
    plt.xticks(rotation=90, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    xml_folder = r"C:\Users\veera\Desktop\Veer\Coding\Python\Veritas\FloW_IMG\FloW_IMG\test\annotations"
    plot(xml_folder)
