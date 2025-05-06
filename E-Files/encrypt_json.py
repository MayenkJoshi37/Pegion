import os
import json
from cryptography.fernet import Fernet

# Generate a key and save it if not exists
KEY_FILE = "secret.key"

def load_or_generate_key():
    if not os.path.exists(KEY_FILE):
        key = Fernet.generate_key()
        with open(KEY_FILE, "wb") as f:
            f.write(key)
    else:
        with open(KEY_FILE, "rb") as f:
            key = f.read()
    return key

def encrypt_json(json_file, output_dir="E-Files", delete_original=True):
    key = load_or_generate_key()
    fernet = Fernet(key)

    with open(json_file, "rb") as f:
        data = f.read()

    encrypted_data = fernet.encrypt(data)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(json_file)
    encrypted_file_path = os.path.join(output_dir, filename + ".enc")

    with open(encrypted_file_path, "wb") as f:
        f.write(encrypted_data)

    print(f"Encrypted and saved to {encrypted_file_path}")

    if delete_original:
        os.remove(json_file)
        print(f"Original file {json_file} deleted.")

# Example usage
if __name__ == "__main__":
    encrypt_json("data.json")
