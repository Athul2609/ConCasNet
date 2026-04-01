import os

def delete_n_files(input_path):
    for layer1 in os.listdir(input_path):
        layer1_path = os.path.join(input_path, layer1)

        if not os.path.isdir(layer1_path):
            continue

        for layer2 in os.listdir(layer1_path):
            layer2_path = os.path.join(layer1_path, layer2)

            if not os.path.isdir(layer2_path):
                continue

            # Find files starting with 'n'
            n_files = [
                f for f in os.listdir(layer2_path)
                if os.path.isfile(os.path.join(layer2_path, f)) and f.lower().startswith('n')
            ]

            # Delete up to 3 files
            for file_name in n_files[:3]:
                file_path = os.path.join(layer2_path, file_name)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")


# Your actual path (use raw string to avoid escape issues)
input_path = r"C:\Users\athul\myfiles\Work\Research\Sleep Stage\SHHS Cascading Order Analysis Result Construction\saved_SHHS\3_0_2_4_1"

delete_n_files(input_path)