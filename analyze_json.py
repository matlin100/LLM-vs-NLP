import json
from collections import Counter

def analyze_json_file(file_path):
    print(f"Analyzing file: {file_path}")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Count total objects
    total_objects = len(data)
    print(f"\nTotal number of objects: {total_objects}")
    
    # Count total labels and their types
    total_labels = 0
    label_types = Counter()
    
    for obj in data:
        if 'tags' in obj:
            tags = obj['tags']
            total_labels += len(tags)
            for tag in tags:
                if 'label' in tag:
                    label_types[tag['label']] += 1
    
    print(f"\nTotal number of labels: {total_labels}")
    print("\nLabel type distribution:")
    for label, count in label_types.most_common():
        print(f"{label}: {count} ({(count/total_labels)*100:.2f}%)")

if __name__ == "__main__":
    analyze_json_file("data/evaluation_data_fixed.json") 