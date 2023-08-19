import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

def create_heat_map(folder_path, image_width, image_height):
    heat_map = np.zeros((image_height, image_width))
    xml_files = [f for f in os.listdir(folder_path) if f.endswith('.xml')]

    for xml_file in xml_files:
        xml_path = os.path.join(folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall(".//object[name='bottle']"):
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)

            heat_map[ymin:ymax, xmin:xmax] += 1

    return heat_map

if __name__ == "__main__":
    folder_path = r"C:\Users\veera\Desktop\Veer\Coding\Python\Veritas\FloW_IMG\FloW_IMG\test\annotations"
    image_width = 1280  
    image_height = 720  

    heat_map = create_heat_map(folder_path, image_width, image_height)

    plt.imshow(heat_map, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Object Location Heat Map')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    # plt.savefig('heatmap.png', dpi=300)
    plt.show()
