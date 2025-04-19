import json

def check_json_structure(file_path):
    print(f"Checking file: {file_path}")
    with open(file_path, 'r') as f:
        content = f.read()
        
    # Check if the file starts and ends with the correct brackets
    if not content.strip().startswith('['):
        print("Error: File must start with '['")
        return
    if not content.strip().endswith(']'):
        print("Error: File must end with ']'")
        return
        
    # Try to parse the entire file
    try:
        data = json.loads(content)
        print("JSON file is valid")
        print(f"Number of objects in array: {len(data)}")
    except json.JSONDecodeError as e:
        line_num = e.lineno
        col_num = e.colno
        print(f"JSON Error at line {line_num}, column {col_num}")
        print(f"Error message: {e.msg}")
        
        # Get the context around the error
        lines = content.split('\n')
        start_line = max(0, line_num - 5)
        end_line = min(len(lines), line_num + 5)
        
        print("\nContext around error:")
        for i in range(start_line, end_line):
            prefix = "-> " if i + 1 == line_num else "   "
            print(f"{prefix}{i+1}: {lines[i]}")

if __name__ == "__main__":
    check_json_structure("data/evaluation_data_fixed.json") 