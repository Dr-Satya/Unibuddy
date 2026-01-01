import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from urllib.parse import urljoin, urlparse
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import queue

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

from src.config import settings
from src.database import db_manager

class ThreadedUniversityScraper:
    """Multi-threaded scraper for GD Goenka University website data."""
    
    def __init__(self, max_workers: int = 4):
        self.base_url = "https://www.gdgoenkauniversity.com"
        self.max_workers = max_workers
        self.session_pool = queue.Queue()
        self.driver_pool = queue.Queue()
        self.results_lock = threading.Lock()
        
        # Initialize session pool
        for _ in range(max_workers):
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            })
            self.session_pool.put(session)
    
    def _get_session(self) -> requests.Session:
        """Get a session from the pool."""
        return self.session_pool.get()
    
    def _return_session(self, session: requests.Session):
        """Return a session to the pool."""
        self.session_pool.put(session)
    
    def _get_selenium_driver(self) -> Optional[webdriver.Chrome]:
        """Get Selenium Chrome driver with appropriate options."""
        if not self.driver_pool.empty():
            return self.driver_pool.get()
        
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            chrome_options.add_argument("--disable-blink-features=AutomationControlled")
            
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            print(f"Error creating Selenium driver: {e}")
            return None
    
    def _return_driver(self, driver: webdriver.Chrome):
        """Return a driver to the pool."""
        if driver and self.driver_pool.qsize() < self.max_workers:
            self.driver_pool.put(driver)
        elif driver:
            try:
                driver.quit()
            except:
                pass
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,;:!?()-]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def _scrape_page_with_requests(self, url: str, content_type: str = "general") -> Optional[Dict[str, Any]]:
        """Scrape a single page using requests."""
        session = self._get_session()
        try:
            print(f"[Thread {threading.current_thread().name}] Scraping: {url}")
            
            response = session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('title') or soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else f"University Info - {content_type}"
            
            # Extract main content
            content_sections = []
            
            # Look for common content containers
            content_selectors = [
                '.content', '.main-content', '#content', '.page-content',
                '.entry-content', 'main', 'article'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    content_text = content_elem.get_text(separator='\n', strip=True)
                    if len(content_text) > 100:  # Only use if substantial content
                        content_sections.append(content_text)
                        break
            
            # Fallback: extract all text from body
            if not content_sections:
                body = soup.find('body')
                if body:
                    # Remove script and style elements
                    for script in body(["script", "style", "nav", "footer", "header"]):
                        script.decompose()
                    content_text = body.get_text(separator='\n', strip=True)
                    content_sections.append(content_text)
            
            # Extract tables (often contain important information)
            tables = soup.find_all('table')
            table_content = []
            for table in tables:
                table_text = table.get_text(separator=' | ', strip=True)
                if table_text:
                    table_content.append(f"TABLE: {table_text}")
            
            # Combine all content
            full_content = '\n\n'.join(content_sections)
            if table_content:
                full_content += '\n\n' + '\n\n'.join(table_content)
            
            # Clean the content
            full_content = self._clean_text(full_content)
            
            if len(full_content) < 100:
                print(f"[Thread {threading.current_thread().name}] Content too short for {url}, trying Selenium...")
                return self._scrape_page_with_selenium(url, content_type)
            
            # Create document data
            doc_data = {
                'id': hashlib.sha256(f"{url}{title}".encode()).hexdigest(),
                'url': url,
                'title': title,
                'content': full_content,
                'content_type': content_type,
                'scraped_at': datetime.now(),
                'metadata': {
                    'scraping_method': 'requests',
                    'page_length': len(full_content),
                    'tables_found': len(tables),
                    'response_status': response.status_code,
                    'thread_id': threading.current_thread().name
                }
            }
            
            print(f"[Thread {threading.current_thread().name}] ✓ Scraped {url}: {len(full_content)} characters")
            return doc_data
            
        except Exception as e:
            print(f"[Thread {threading.current_thread().name}] Error scraping {url} with requests: {e}")
            print(f"[Thread {threading.current_thread().name}] Trying Selenium as fallback...")
            return self._scrape_page_with_selenium(url, content_type)
        finally:
            self._return_session(session)
    
    def _scrape_page_with_selenium(self, url: str, content_type: str = "general") -> Optional[Dict[str, Any]]:
        """Scrape a single page using Selenium."""
        driver = self._get_selenium_driver()
        if not driver:
            return None
        
        try:
            print(f"[Thread {threading.current_thread().name}] Selenium scraping: {url}")
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for dynamic content to load
            time.sleep(3)
            
            # Extract title
            try:
                title = driver.title or f"University Info - {content_type}"
            except:
                title = f"University Info - {content_type}"
            
            # Extract content
            try:
                # Try to find main content area
                content_elem = None
                content_selectors = [
                    "main", ".content", ".main-content", "#content", 
                    ".page-content", ".entry-content", "article"
                ]
                
                for selector in content_selectors:
                    try:
                        content_elem = driver.find_element(By.CSS_SELECTOR, selector)
                        break
                    except:
                        continue
                
                if not content_elem:
                    content_elem = driver.find_element(By.TAG_NAME, "body")
                
                # Remove navigation, footer, etc.
                driver.execute_script("""
                    var elements = document.querySelectorAll('nav, footer, header, .nav, .footer, .header, script, style');
                    for (var i = 0; i < elements.length; i++) {
                        elements[i].remove();
                    }
                """)
                
                content_text = content_elem.text
                
            except Exception as e:
                print(f"[Thread {threading.current_thread().name}] Error extracting content with Selenium: {e}")
                content_text = driver.find_element(By.TAG_NAME, "body").text
            
            # Clean content
            content_text = self._clean_text(content_text)
            
            if len(content_text) < 50:
                print(f"[Thread {threading.current_thread().name}] Warning: Very short content extracted from {url}")
                return None
            
            doc_data = {
                'id': hashlib.sha256(f"{url}{title}".encode()).hexdigest(),
                'url': url,
                'title': self._clean_text(title),
                'content': content_text,
                'content_type': content_type,
                'scraped_at': datetime.now(),
                'metadata': {
                    'scraping_method': 'selenium',
                    'page_length': len(content_text),
                    'browser': 'chrome',
                    'thread_id': threading.current_thread().name
                }
            }
            
            print(f"[Thread {threading.current_thread().name}] ✓ Selenium scraped {url}: {len(content_text)} characters")
            return doc_data
            
        except Exception as e:
            print(f"[Thread {threading.current_thread().name}] Error scraping {url} with Selenium: {e}")
            return None
            
        finally:
            self._return_driver(driver)
    
    def scrape_all_threaded(self, progress_callback: Optional[Callable[[int, int], None]] = None) -> List[Dict[str, Any]]:
        """Scrape all university data using multiple threads."""
        print(f"🚀 Starting threaded university data scraping with {self.max_workers} workers...")
        start_time = time.time()
        
        # Define all URLs to scrape
        scraping_tasks = [
            (settings.UNIVERSITY_URL, "fee_structure"),
            ("https://www.gdgoenkauniversity.com/school-of-engineering/faculty", "faculty"),
            (urljoin(self.base_url, "/admissions/"), "admission"),
            (urljoin(self.base_url, "/courses/"), "courses"),
            (urljoin(self.base_url, "/about/"), "about"),
            (urljoin(self.base_url, "/facilities/"), "facilities"),
            (urljoin(self.base_url, "/academics/"), "courses"),
            (urljoin(self.base_url, "/programs/"), "courses"),
            (urljoin(self.base_url, "/engineering/"), "courses"),
            (urljoin(self.base_url, "/management/"), "courses"),
        ]
        
        documents = []
        completed_count = 0
        total_tasks = len(scraping_tasks)
        
        with ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="Scraper") as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._scrape_page_with_requests, url, content_type): (url, content_type)
                for url, content_type in scraping_tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                url, content_type = future_to_task[future]
                completed_count += 1
                
                try:
                    result = future.result(timeout=60)  # 60 second timeout per page
                    if result:
                        with self.results_lock:
                            documents.append(result)
                        print(f"📄 [{completed_count}/{total_tasks}] Successfully scraped: {result['title']}")
                    else:
                        print(f"❌ [{completed_count}/{total_tasks}] Failed to scrape: {url}")
                        
                except Exception as e:
                    print(f"❌ [{completed_count}/{total_tasks}] Exception scraping {url}: {e}")
                
                # Call progress callback if provided
                if progress_callback:
                    progress_callback(completed_count, total_tasks)
        
        elapsed_time = time.time() - start_time
        print(f"🎉 Threaded scraping completed in {elapsed_time:.2f} seconds!")
        print(f"📊 Successfully scraped {len(documents)}/{total_tasks} documents")
        
        return documents
    
    def store_scraped_data_threaded(self, documents: List[Dict[str, Any]]) -> int:
        """Store scraped documents in database using threading."""
        print(f"💾 Starting threaded storage of {len(documents)} documents...")
        start_time = time.time()
        
        stored_count = 0
        storage_lock = threading.Lock()
        
        def store_document(doc):
            nonlocal stored_count
            try:
                # Store in database
                db_manager.store_university_data(doc)
                with storage_lock:
                    stored_count += 1
                print(f"✓ [{stored_count}/{len(documents)}] Stored: {doc['title']}")
                
            except Exception as e:
                print(f"✗ Error storing document {doc.get('title', 'Unknown')}: {e}")
        
        with ThreadPoolExecutor(max_workers=min(4, len(documents)), thread_name_prefix="Storage") as executor:
            # Submit all storage tasks
            futures = [executor.submit(store_document, doc) for doc in documents]
            
            # Wait for all to complete
            for future in as_completed(futures):
                try:
                    future.result(timeout=30)
                except Exception as e:
                    print(f"Storage task failed: {e}")
        
        elapsed_time = time.time() - start_time
        print(f"💾 Storage completed in {elapsed_time:.2f} seconds")
        print(f"📊 Stored {stored_count}/{len(documents)} documents in database")
        
        return stored_count
    
    def cleanup(self):
        """Clean up resources."""
        print("🧹 Cleaning up scraper resources...")
        
        # Close all sessions
        while not self.session_pool.empty():
            try:
                session = self.session_pool.get_nowait()
                session.close()
            except:
                break
        
        # Close all drivers
        while not self.driver_pool.empty():
            try:
                driver = self.driver_pool.get_nowait()
                driver.quit()
            except:
                break
        
        print("✅ Scraper cleanup completed")

# Global threaded scraper instance
threaded_university_scraper = ThreadedUniversityScraper(max_workers=4)