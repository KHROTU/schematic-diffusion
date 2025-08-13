import os
import time
import logging
import subprocess
import webbrowser
import sys
import json
import cloudscraper
from tqdm import tqdm

LITEMATIC_INPUT_DIR = "../data/3_litematics_to_convert" 
SCHEM_OUTPUT_DIR = "../data/2_named_schematics"

CONVERTER_URL = "https://abfielder.com/tools/converter/uploadLitematic.php"
CONVERTER_PAGE = "https://abfielder.com/tools/converter/litematictoschem"

SERVER_SCRIPT_NAME = "converter_server.py"
SESSION_FILE = "converter_session.json"

logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

def get_converter_session():
    logging.info("Phase 1: Automating browser to get converter session...")
    server_process = None
    try:
        server_process = subprocess.Popen([sys.executable, SERVER_SCRIPT_NAME], stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        time.sleep(2)
        webbrowser.open(CONVERTER_PAGE)
        
        try:
            server_process.communicate(timeout=60) 
        except subprocess.TimeoutExpired:
            server_process.kill()
            raise TimeoutError("Browser did not send session data in time. Make sure Tampermonkey script is active.")

        if not os.path.exists(SESSION_FILE):
            raise FileNotFoundError(f"Server failed to save '{SESSION_FILE}'.")
            
        with open(SESSION_FILE, 'r', encoding='utf-8') as f:
            session_info = json.load(f)
            
        if not session_info.get("cookies") or not session_info.get("userAgent"):
            raise ValueError("Incomplete session data captured. Cookies or User-Agent is missing.")
            
        logging.info("Successfully received full session data for converter.")
        return session_info
        
    finally:
        if server_process and server_process.poll() is None:
            server_process.terminate()
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)

def main():
    start_time = time.time()
    
    os.makedirs(SCHEM_OUTPUT_DIR, exist_ok=True)
    if not os.path.isdir(LITEMATIC_INPUT_DIR):
        logging.error(f"Input directory not found: '{LITEMATIC_INPUT_DIR}'")
        logging.error("Please run the previous scripts to create and populate this folder.")
        return

    files_to_convert = [f for f in os.listdir(LITEMATIC_INPUT_DIR) if f.lower().endswith(".litematic")]

    if not files_to_convert:
        logging.info("No .litematic files found in the input directory to convert. Exiting.")
        return

    logging.info(f"Found {len(files_to_convert)} litematic files to convert.")

    try:
        session_data = get_converter_session()
    except Exception as e:
        logging.critical(f"CRITICAL ERROR: Could not get browser session: {e}")
        return

    scraper = cloudscraper.create_scraper()
    headers = {
        'User-Agent': session_data['userAgent'],
        'Accept': '*/*',
        'Referer': CONVERTER_PAGE,
        'Origin': 'https://abfielder.com',
        'Cookie': session_data['cookies']
    }

    success_count = 0
    error_count = 0
    
    logging.info("\nPhase 2: Converting all files using the online service...")
    for filename in tqdm(files_to_convert, desc="Converting Litematics", unit="file"):
        original_path = os.path.join(LITEMATIC_INPUT_DIR, filename)
        
        try:
            with open(original_path, 'rb') as f_in:
                files_payload = {
                    'litematic': (filename, f_in, 'application/octet-stream')
                }
                response = scraper.post(CONVERTER_URL, headers=headers, files=files_payload, timeout=90)
                response.raise_for_status()

            base_name, _ = os.path.splitext(filename)
            final_path = os.path.join(SCHEM_OUTPUT_DIR, f"{base_name}.schem")

            with open(final_path, 'wb') as f_out:
                f_out.write(response.content)

            os.remove(original_path) 
            success_count += 1

        except Exception as e:
            logging.error(f"Failed to convert '{filename}': {e}")
            error_count += 1
            
    end_time = time.time()
    print("\n" + "="*40)
    print("CONVERSION COMPLETE")
    print("="*40)
    print(f"Total time taken: {end_time - start_time:.2f} seconds")
    print(f"Successfully converted and moved: {success_count} files")
    print(f"Errors (files left in input dir): {error_count}")
    print(f"Converted files are in: '{os.path.abspath(SCHEM_OUTPUT_DIR)}'")
    print("="*40)

if __name__ == "__main__":
    main()