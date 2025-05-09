import os
import sys
from cryptography.fernet import Fernet

ENCRYPTED_DIR = "E-Files"
KEYS_DIR = os.path.join(ENCRYPTED_DIR, "secret_keys")
DECRYPTED_DIR = "Decrypted-Files"

def decrypt_file(file_name):
    enc_file_path = os.path.join(ENCRYPTED_DIR, file_name + ".enc")
    key_file_path = os.path.join(KEYS_DIR, file_name + ".key")
    meta_file_path = os.path.join(KEYS_DIR, file_name + ".meta")

    if not all(os.path.exists(p) for p in [enc_file_path, key_file_path, meta_file_path]):
        print("Encrypted file, key, or metadata file not found.")
        return

    with open(key_file_path, "rb") as kf:
        key = kf.read()

    with open(meta_file_path, "r") as mf:
        original_ext = mf.read().strip()

    fernet = Fernet(key)
    with open(enc_file_path, "rb") as ef:
        encrypted_data = ef.read()

    try:
        decrypted_data = fernet.decrypt(encrypted_data)

        os.makedirs(DECRYPTED_DIR, exist_ok=True)
        output_file = os.path.join(DECRYPTED_DIR, file_name + original_ext)

        with open(output_file, "wb") as df:
            df.write(decrypted_data)

        print(f"✅ Decrypted file saved as: {output_file}")
        try:
            os.startfile(output_file)  # Windows only
        except:
            print("Open the file manually to view it.")
    except Exception as e:
        print(f"❌ Failed to decrypt: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python decrypt_file.py <filename_without_ext>")
    else:
        decrypt_file(sys.argv[1])
