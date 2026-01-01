import hashlib
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
import re

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

class UniversityScraper:
    """Scraper for GD Goenka University website data."""
    
    def __init__(self):
        self.base_url = "https://www.gdgoenkauniversity.com"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
    def _get_selenium_driver(self) -> Optional[webdriver.Chrome]:
        """Get Selenium Chrome driver with appropriate options."""
        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--window-size=1920,1080")
            
            driver = webdriver.Chrome(options=chrome_options)
            return driver
        except Exception as e:
            print(f"Error creating Selenium driver: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text.strip())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\\s.,;:!?()-]', ' ', text)
        
        # Remove multiple spaces
        text = re.sub(r' +', ' ', text)
        
        return text.strip()
    
    def scrape_fee_structure(self) -> Optional[Dict[str, Any]]:
        """Scrape fee structure information."""
        url = settings.UNIVERSITY_URL
        
        try:
            print(f"Scraping fee structure from: {url}")
            
            # Try requests first
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('title') or soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "Fee Structure"
            
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
                    content_text = content_elem.get_text(separator='\\n', strip=True)
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
                    content_text = body.get_text(separator='\\n', strip=True)
                    content_sections.append(content_text)
            
            # Extract tables (often contain fee information)
            tables = soup.find_all('table')
            table_content = []
            for table in tables:
                table_text = table.get_text(separator=' | ', strip=True)
                if table_text:
                    table_content.append(f"TABLE: {table_text}")
            
            # Combine all content
            full_content = '\\n\\n'.join(content_sections)
            if table_content:
                full_content += '\\n\\n' + '\\n\\n'.join(table_content)
            
            # Clean the content
            full_content = self._clean_text(full_content)
            
            if len(full_content) < 100:
                print("Content too short, trying Selenium...")
                return self._scrape_with_selenium(url)
            
            # Create document data
            doc_data = {
                'id': hashlib.sha256(f"{url}{title}".encode()).hexdigest(),
                'url': url,
                'title': title,
                'content': full_content,
                'content_type': 'fee_structure',
                'scraped_at': datetime.now(),
                'metadata': {
                    'scraping_method': 'requests',
                    'page_length': len(full_content),
                    'tables_found': len(tables),
                    'response_status': response.status_code
                }
            }
            
            return doc_data
            
        except Exception as e:
            print(f"Error scraping with requests: {e}")
            print("Trying Selenium as fallback...")
            return self._scrape_with_selenium(url)
    
    def _scrape_with_selenium(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape using Selenium for JavaScript-heavy pages."""
        driver = self._get_selenium_driver()
        if not driver:
            return None
        
        try:
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait a bit more for dynamic content
            time.sleep(3)
            
            # Extract title
            try:
                title = driver.find_element(By.TAG_NAME, "title").get_attribute("textContent")
                if not title:
                    title = driver.find_element(By.TAG_NAME, "h1").text
            except:
                title = "Fee Structure"
            
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
                print(f"Error extracting content with Selenium: {e}")
                content_text = driver.find_element(By.TAG_NAME, "body").text
            
            # Clean content
            content_text = self._clean_text(content_text)
            
            if len(content_text) < 50:
                print("Warning: Very short content extracted")
                return None
            
            doc_data = {
                'id': hashlib.sha256(f"{url}{title}".encode()).hexdigest(),
                'url': url,
                'title': self._clean_text(title),
                'content': content_text,
                'content_type': 'fee_structure',
                'scraped_at': datetime.now(),
                'metadata': {
                    'scraping_method': 'selenium',
                    'page_length': len(content_text),
                    'browser': 'chrome'
                }
            }
            
            return doc_data
            
        except Exception as e:
            print(f"Error scraping with Selenium: {e}")
            return None
            
        finally:
            try:
                driver.quit()
            except:
                pass
    
    def scrape_faculty_data(self) -> Optional[Dict[str, Any]]:
        """Scrape faculty information from School of Engineering."""
        faculty_url = "https://www.gdgoenkauniversity.com/school-of-engineering/faculty"
        
        try:
            print(f"Scraping faculty data from: {faculty_url}")
            
            response = self.session.get(faculty_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title_elem = soup.find('title') or soup.find('h1')
            title = title_elem.get_text(strip=True) if title_elem else "Faculty - School of Engineering"
            
            # Extract faculty information
            faculty_content = []
            
            # Look for faculty cards/profiles
            faculty_selectors = [
                '.faculty-card', '.faculty-profile', '.faculty-member',
                '.staff-card', '.team-member', '.professor-card',
                '.faculty', '.staff', '.team'
            ]
            
            faculty_found = False
            for selector in faculty_selectors:
                faculty_elements = soup.select(selector)
                if faculty_elements:
                    faculty_found = True
                    print(f"Found {len(faculty_elements)} faculty elements with selector: {selector}")
                    for elem in faculty_elements:
                        faculty_text = elem.get_text(separator=' ', strip=True)
                        if len(faculty_text) > 10:  # Only include substantial content
                            faculty_content.append(faculty_text)
                    break
            
            # If no specific faculty elements found, extract main content
            if not faculty_found:
                print("No specific faculty elements found, extracting main content...")
                content_selectors = [
                    '.content', '.main-content', '#content', '.page-content',
                    'main', 'article', '.faculty-list'
                ]
                
                for selector in content_selectors:
                    content_elem = soup.select_one(selector)
                    if content_elem:
                        content_text = content_elem.get_text(separator='\n', strip=True)
                        if len(content_text) > 100:
                            faculty_content.append(content_text)
                            break
            
            # Fallback: extract from body if nothing else worked
            if not faculty_content:
                print("Using fallback: extracting from body...")
                body = soup.find('body')
                if body:
                    # Remove unwanted elements
                    for element in body(["script", "style", "nav", "footer", "header"]):
                        element.decompose()
                    
                    # Look for elements containing faculty-related keywords
                    faculty_keywords = ['professor', 'dr.', 'ph.d', 'faculty', 'department', 'engineering']
                    all_text = body.get_text(separator='\n', strip=True)
                    
                    # Split into paragraphs and filter for faculty-related content
                    paragraphs = all_text.split('\n')
                    for paragraph in paragraphs:
                        if any(keyword.lower() in paragraph.lower() for keyword in faculty_keywords):
                            if len(paragraph.strip()) > 20:
                                faculty_content.append(paragraph.strip())
            
            # Combine all faculty content
            full_content = '\n\n'.join(faculty_content)
            full_content = self._clean_text(full_content)
            
            if len(full_content) < 100:
                print("Faculty content too short, trying Selenium...")
                return self._scrape_faculty_with_selenium(faculty_url)
            
            # Create document data
            doc_data = {
                'id': hashlib.sha256(f"{faculty_url}{title}".encode()).hexdigest(),
                'url': faculty_url,
                'title': title,
                'content': full_content,
                'content_type': 'faculty',
                'scraped_at': datetime.now(),
                'metadata': {
                    'scraping_method': 'requests',
                    'page_length': len(full_content),
                    'faculty_elements_found': len(faculty_content),
                    'response_status': response.status_code
                }
            }
            
            print(f"✓ Scraped faculty data: {len(full_content)} characters")
            return doc_data
            
        except Exception as e:
            print(f"Error scraping faculty with requests: {e}")
            print("Trying Selenium for faculty scraping...")
            return self._scrape_faculty_with_selenium(faculty_url)
    
    def _scrape_faculty_with_selenium(self, url: str) -> Optional[Dict[str, Any]]:
        """Scrape faculty page using Selenium for dynamic content."""
        driver = self._get_selenium_driver()
        if not driver:
            return None
        
        try:
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 15).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait for dynamic content to load
            time.sleep(5)
            
            # Extract title
            try:
                title = driver.title or "Faculty - School of Engineering"
            except:
                title = "Faculty - School of Engineering"
            
            # Extract faculty content
            faculty_content = []
            
            try:
                # Try to find faculty-specific elements
                faculty_selectors = [
                    ".faculty-card", ".faculty-profile", ".faculty-member",
                    ".staff-card", ".team-member", ".professor-card",
                    ".faculty", ".staff", ".team"
                ]
                
                faculty_found = False
                for selector in faculty_selectors:
                    try:
                        elements = driver.find_elements(By.CSS_SELECTOR, selector)
                        if elements:
                            faculty_found = True
                            print(f"Found {len(elements)} faculty elements with Selenium")
                            for element in elements:
                                text = element.text.strip()
                                if len(text) > 10:
                                    faculty_content.append(text)
                            break
                    except:
                        continue
                
                # If no specific faculty elements, get main content
                if not faculty_found:
                    try:
                        main_element = driver.find_element(By.CSS_SELECTOR, "main, .content, .main-content, #content")
                        faculty_content.append(main_element.text)
                    except:
                        # Fallback to body
                        faculty_content.append(driver.find_element(By.TAG_NAME, "body").text)
                
            except Exception as e:
                print(f"Error extracting faculty content: {e}")
                faculty_content.append(driver.find_element(By.TAG_NAME, "body").text)
            
            # Combine and clean content
            full_content = '\n\n'.join(faculty_content)
            full_content = self._clean_text(full_content)
            
            if len(full_content) < 50:
                print("Warning: Very short faculty content extracted")
                return None
            
            doc_data = {
                'id': hashlib.sha256(f"{url}{title}".encode()).hexdigest(),
                'url': url,
                'title': self._clean_text(title),
                'content': full_content,
                'content_type': 'faculty',
                'scraped_at': datetime.now(),
                'metadata': {
                    'scraping_method': 'selenium',
                    'page_length': len(full_content),
                    'browser': 'chrome'
                }
            }
            
            print(f"✓ Scraped faculty data with Selenium: {len(full_content)} characters")
            return doc_data
            
        except Exception as e:
            print(f"Error scraping faculty with Selenium: {e}")
            return None
            
        finally:
            try:
                driver.quit()
            except:
                pass
    
    def scrape_additional_pages(self) -> List[Dict[str, Any]]:
        """Scrape additional university pages."""
        additional_urls = [
            "/admissions/",
            "/courses/",
            "/about/",
            "/facilities/",
            "/academics/"
        ]
        
        documents = []
        
        for path in additional_urls:
            url = urljoin(self.base_url, path)
            
            try:
                print(f"Scraping additional page: {url}")
                
                response = self.session.get(url, timeout=30)
                if response.status_code != 200:
                    continue
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract title
                title_elem = soup.find('title') or soup.find('h1')
                title = title_elem.get_text(strip=True) if title_elem else f"University Info - {path.strip('/')}"
                
                # Extract content
                content_text = ""
                content_elem = soup.select_one('.content, .main-content, #content, main, article')
                if content_elem:
                    content_text = content_elem.get_text(separator='\\n', strip=True)
                else:
                    body = soup.find('body')
                    if body:
                        for script in body(["script", "style", "nav", "footer", "header"]):
                            script.decompose()
                        content_text = body.get_text(separator='\\n', strip=True)
                
                content_text = self._clean_text(content_text)
                
                if len(content_text) < 100:
                    continue
                
                # Determine content type based on URL
                content_type = "general"
                if "admission" in path.lower():
                    content_type = "admission"
                elif "course" in path.lower() or "academic" in path.lower():
                    content_type = "courses"
                elif "about" in path.lower():
                    content_type = "about"
                elif "facilit" in path.lower():
                    content_type = "facilities"
                
                doc_data = {
                    'id': hashlib.sha256(f"{url}{title}".encode()).hexdigest(),
                    'url': url,
                    'title': title,
                    'content': content_text,
                    'content_type': content_type,
                    'scraped_at': datetime.now(),
                    'metadata': {
                        'scraping_method': 'requests',
                        'page_length': len(content_text),
                        'source_path': path
                    }
                }
                
                documents.append(doc_data)
                
                # Be respectful with delays
                time.sleep(2)
                
            except Exception as e:
                print(f"Error scraping {url}: {e}")
                continue
        
        return documents
    
    def scrape_all(self) -> List[Dict[str, Any]]:
        """Scrape all university data."""
        documents = []
        
        print("Starting university data scraping...")
        
        # Scrape fee structure (primary target)
        fee_doc = self.scrape_fee_structure()
        if fee_doc:
            documents.append(fee_doc)
            print(f"✓ Scraped fee structure: {len(fee_doc['content'])} characters")
        else:
            print("✗ Failed to scrape fee structure")
        
        # Scrape faculty data (new addition)
        faculty_doc = self.scrape_faculty_data()
        if faculty_doc:
            documents.append(faculty_doc)
            print(f"✓ Scraped faculty data: {len(faculty_doc['content'])} characters")
        else:
            print("✗ Failed to scrape faculty data")
        
        # Scrape additional pages
        additional_docs = self.scrape_additional_pages()
        documents.extend(additional_docs)
        
        print(f"Scraping completed. Total documents: {len(documents)}")
        
        return documents
    
    def store_scraped_data(self, documents: List[Dict[str, Any]]):
        """Store scraped documents in database."""
        stored_count = 0
        
        for doc in documents:
            try:
                # Store in database
                db_manager.store_university_data(doc)
                stored_count += 1
                print(f"✓ Stored document: {doc['title']}")
                
            except Exception as e:
                print(f"✗ Error storing document {doc.get('title', 'Unknown')}: {e}")
        
        print(f"Stored {stored_count}/{len(documents)} documents in database")
        
        return stored_count

# Global scraper instance
university_scraper = UniversityScraper()
