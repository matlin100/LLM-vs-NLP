def fix_json_file(input_file, output_file):
    with open(input_file, 'r') as f:
        content = f.read()
    
    # Fix the missing comma between objects
    fixed_content = content.replace('}{', '},{')
    
    with open(output_file, 'w') as f:
        f.write(fixed_content)

if __name__ == "__main__":
    fix_json_file("data/evaluation_data.json", "data/evaluation_data_fixed.json") 