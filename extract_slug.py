import json 

def extract_slug(input_json_path, output_json_path):
    with open(input_json_path, 'r') as infile:
        payload = json.load(infile)
        data = payload['data']
        output_array = []
        for item in data:
            output_array.append(item['slug'])

    with open(output_json_path, 'w') as outfile:
        json.dump(output_array, outfile)


if __name__ == "__main__":
    input_path = 'input_data.json'
    output_path = 'output_slugs.json'
    extract_slug(input_path, output_path)
