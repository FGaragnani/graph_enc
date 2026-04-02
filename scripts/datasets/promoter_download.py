import os
from urllib.parse import urljoin
import ftplib
from urllib.parse import urlparse
import re
import tqdm

OUTPUT_DIR      = "/work/tesi_fgaragnani/dataset/AI4BIO/promoters"
DOWNLOAD_DIR    ="https://epd.expasy.org/ftp/epdnew/"

import urllib.request

def download_promoter_files():
    # List directories on the FTP site
    
    parsed_url = urlparse(DOWNLOAD_DIR)
    ftp = ftplib.FTP(parsed_url.netloc)
    ftp.login()
    ftp.cwd(parsed_url.path)
    
    folders = [item for item in ftp.nlst() if item not in ['.', '..']]
    
    for folder in tqdm.tqdm(folders):
        current_url = urljoin(DOWNLOAD_DIR, f"{folder}/current/")
        try:
            response = urllib.request.urlopen(current_url)
            html = response.read().decode('utf-8')
            
            # Extract .dat file links
            dat_files = re.findall(r'href=["\']([^"\']*\.dat)["\']', html)
            
            for dat_file in dat_files:
                file_url = urljoin(current_url, dat_file)
                output_path = os.path.join(OUTPUT_DIR, folder, dat_file)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                print(f"Downloading {file_url}...")
                urllib.request.urlretrieve(file_url, output_path)
        except Exception as e:
            print(f"Error processing {folder}: {e}")
    
    ftp.quit()

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    download_promoter_files()