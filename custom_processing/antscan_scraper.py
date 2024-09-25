import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import time
import json
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import NoSuchElementException
from webdriver_manager.chrome import ChromeDriverManager
from concurrent.futures import ThreadPoolExecutor, as_completed
from remove_temp_files import remove_temp_files, remove_empty_and_no_stl_directories, count_valid_scans

# URL to scrape
BASE_URL = "https://biomedisa.info/antscan/?show_all=True#"

# Directory to save downloaded files
DOWNLOAD_DIR = os.path.join(os.path.dirname(__file__), "antscan_data")

# Ensure the download directory exists
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

def download_file(url, download_dir, filename):
    local_filename = os.path.join(download_dir, filename)
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded {local_filename}")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None
    return local_filename

def get_specimen_links(base_url):
    print("Fetching specimen links...")
    response = requests.get(base_url)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, 'html.parser')
    specimen_links = set()  # Use a set to store unique specimen links

    for link in soup.find_all('a', href=True):
        href = link['href']
        if "/antscan/specimen/" in href:
            specimen_links.add(urljoin(base_url, href))

    print(f"Found {len(specimen_links)} unique specimen links.")
    return list(specimen_links)  # Convert the set back to a list

def process_specimen(specimen_url, download_dir, extension_path):
    options = Options()
    options.add_argument(f"--load-extension={extension_path}")
    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    try:
        print(f"Processing specimen: {specimen_url}")
        driver.get(specimen_url)
        time.sleep(2)  # Wait for the page to load

        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        download_buttons = soup.find_all('a', onclick=True)

        if not download_buttons:
            print(f"No download buttons found for specimen {specimen_url}")
            return None

        # Extract metadata from input fields
        metadata = {}
        for input_tag in soup.find_all('input', id=True):
            if input_tag['id'].startswith('id_'):
                key = input_tag['id'][3:]  # Remove 'id_' prefix
                value = input_tag.get('value', '').strip()
                metadata[key] = value

        # Derive the filename from the metadata
        name = metadata.get('name', 'unknown').replace(' ', '_')
        specimen_code = metadata.get('specimen_code', 'unknown').replace(' ', '_')
        base_filename = f"{name}_{specimen_code}"

        # Create a subfolder for the specimen
        specimen_dir = os.path.join(download_dir, base_filename)
        os.makedirs(specimen_dir, exist_ok=True)

        # Update WebDriver download directory for the specimen
        driver.execute_cdp_cmd('Page.setDownloadBehavior', {
            'behavior': 'allow',
            'downloadPath': specimen_dir
        })

        # Write metadata to JSON file in the subfolder
        metadata_file = os.path.join(specimen_dir, f"{base_filename}.json")
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved to {metadata_file}")

        # Check the subfolder for new files before clicking the download button
        files_before = set(os.listdir(specimen_dir))

        for button in download_buttons:
            img_tag = button.find('img', title="download")
            if img_tag:
                onclick_value = button['onclick']
                try:
                    download_button = driver.find_element(By.XPATH, f"//a[@onclick=\"{onclick_value}\"]")
                    download_button.click()

                    # Wait for the download to complete with a timeout of 1 minute
                    timeout = 60  # seconds
                    start_time = time.time()
                    download_complete = False
                    while time.time() - start_time < timeout:
                        time.sleep(2)  # Check every 2 seconds
                        files_after = set(os.listdir(specimen_dir))
                        new_files = files_after - files_before

                        if new_files:
                            for new_file in new_files:
                                if new_file.endswith('.stl'):
                                    new_file_path = os.path.join(specimen_dir, new_file)
                                    new_file_renamed = os.path.join(specimen_dir, f"{base_filename}.stl")
                                    os.rename(new_file_path, new_file_renamed)
                                    print(f"Successfully downloaded and renamed to {new_file_renamed}")
                                    download_complete = True
                                    break
                                elif new_file.endswith('.crdownload'):
                                    print(f"Download in progress: {new_file}")
                                    break  # Exit the inner loop and continue waiting
                            if download_complete:
                                break  # Exit the outer while loop if download is complete
                    else:
                        print(f"Download timed out for {specimen_url}")

                except NoSuchElementException as e:
                    print(f"Error clicking download button for {specimen_url}: {e}")
                time.sleep(5)  # Wait for 5 seconds before the next download

    finally:
        driver.quit()

    return specimen_dir  # Return the specimen directory path

def scrape_stl_files(specimen_links, download_dir, max_workers=5):
    extension_path = os.path.join(os.path.dirname(__file__), "chrome_extension")
    downloaded_files = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_specimen, url, download_dir, extension_path): url for url in specimen_links}
        for future in as_completed(futures):
            result = future.result()
            if result:
                downloaded_files.append(result)

    return downloaded_files  # Return the list of downloaded file paths

if __name__ == "__main__":
    specimen_links = get_specimen_links(BASE_URL)
    print("Starting download of STL files...")
    downloaded_files = scrape_stl_files(specimen_links, DOWNLOAD_DIR)
    print("Downloaded files:", downloaded_files)

    # Perform cleaning after scraping
    print(f"\nCleaning temporary files in {DOWNLOAD_DIR}...")
    remove_temp_files(DOWNLOAD_DIR)
    remove_empty_and_no_stl_directories(DOWNLOAD_DIR)
    valid_scans = count_valid_scans(DOWNLOAD_DIR)
    print(f"Cleaning complete. Number of valid scans: {valid_scans}")
