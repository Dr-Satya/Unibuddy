#!/usr/bin/env python3
"""
Comprehensive Faculty Scraper for GD Goenka University School of Engineering
This script scrapes ALL professor profiles and updates the RAG system
"""

import requests
from bs4 import BeautifulSoup
import time
import sys
import re
from urllib.parse import urljoin, urlparse
import json
from datetime import datetime

sys.path.append('src')

class FacultyScraper:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.base_url = "https://www.gdgoenkauniversity.com"
        self.faculty_profiles = []
        
    def get_page_content(self, url):
        """Get page content with error handling"""
        try:
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def find_faculty_links(self):
        """Find all faculty profile links from the main faculty page"""
        print("🔍 Finding faculty profile links...")
        
        # Try multiple possible faculty page URLs
        faculty_pages = [
            "https://www.gdgoenkauniversity.com/school-of-engineering/faculty",
            "https://www.gdgoenkauniversity.com/academics/school-of-engineering/faculty",
            "https://www.gdgoenkauniversity.com/faculty/school-of-engineering",
            "https://www.gdgoenkauniversity.com/school-of-engineering-sciences/faculty"
        ]
        
        faculty_links = set()
        
        for page_url in faculty_pages:
            content = self.get_page_content(page_url)
            if not content:
                continue
                
            soup = BeautifulSoup(content, 'html.parser')
            
            # Look for faculty profile links
            links = soup.find_all('a', href=True)
            
            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(self.base_url, href)
                    
                    # Check if this looks like a faculty profile URL
                    if any(pattern in full_url.lower() for pattern in [
                        '/dr-', '/prof-', '/faculty/', '/school-of-engineering/',
                        'professor', 'dr.', 'faculty'
                    ]):
                        if 'gdgoenkauniversity.com' in full_url:
                            faculty_links.add(full_url)
                            print(f"Found faculty link: {full_url}")
        
        # Also add the specific Dr. Ugur Guven link we know about
        faculty_links.add("https://www.gdgoenkauniversity.com/school-of-engineering/dr-ugur-guven")
        
        print(f"✅ Found {len(faculty_links)} potential faculty links")
        return list(faculty_links)
    
    def scrape_faculty_profile(self, url):
        """Scrape individual faculty profile"""
        content = self.get_page_content(url)
        if not content:
            return None
        
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract faculty information
        profile_data = {
            'url': url,
            'name': '',
            'title': '',
            'qualifications': '',
            'specialization': '',
            'experience': '',
            'email': '',
            'research_interests': '',
            'publications': '',
            'full_content': '',
            'scraped_at': datetime.now().isoformat()
        }
        
        # Get page title
        title_tag = soup.find('title')
        if title_tag:
            profile_data['page_title'] = title_tag.get_text().strip()
        
        # Extract all text content
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text content
        text_content = soup.get_text()
        
        # Clean up the text
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        clean_text = ' '.join(chunk for chunk in chunks if chunk)
        
        profile_data['full_content'] = clean_text
        
        # Try to extract specific information
        text_lower = clean_text.lower()
        
        # Extract name (look for Dr./Prof. patterns)
        name_patterns = [
            r'dr\.?\s+([a-z]+(?:\s+[a-z]+)*(?:\s+[a-z]+)*)',
            r'prof\.?\s+([a-z]+(?:\s+[a-z]+)*(?:\s+[a-z]+)*)',
            r'professor\s+([a-z]+(?:\s+[a-z]+)*(?:\s+[a-z]+)*)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, text_lower)
            if match:
                profile_data['name'] = match.group(1).title()
                break
        
        # Extract email
        email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', clean_text)
        if email_match:
            profile_data['email'] = email_match.group(1)
        
        # Extract qualifications
        qual_patterns = [
            r'qualification[s]?:?\s*([^.]*(?:phd|ph\.d|msc|m\.sc|btech|b\.tech|mtech|m\.tech)[^.]*)',
            r'degree[s]?:?\s*([^.]*(?:phd|ph\.d|msc|m\.sc|btech|b\.tech|mtech|m\.tech)[^.]*)',
            r'education:?\s*([^.]*(?:phd|ph\.d|msc|m\.sc|btech|b\.tech|mtech|m\.tech)[^.]*)'
        ]
        
        for pattern in qual_patterns:
            match = re.search(pattern, text_lower)
            if match:
                profile_data['qualifications'] = match.group(1).strip()
                break
        
        # Extract specialization
        spec_patterns = [
            r'specialization:?\s*([^.]*)',
            r'area of specialization:?\s*([^.]*)',
            r'expertise:?\s*([^.]*)'
        ]
        
        for pattern in spec_patterns:
            match = re.search(pattern, text_lower)
            if match:
                profile_data['specialization'] = match.group(1).strip()
                break
        
        # Extract experience
        exp_patterns = [
            r'experience:?\s*([^.]*(?:\d+\s*(?:years?|yrs?))[^.]*)',
            r'(\d+\s*(?:years?|yrs?)[^.]*(?:experience|career))'
        ]
        
        for pattern in exp_patterns:
            match = re.search(pattern, text_lower)
            if match:
                profile_data['experience'] = match.group(1).strip()
                break
        
        return profile_data
    
    def scrape_all_faculty(self):
        """Scrape all faculty profiles"""
        print("🎯 Starting comprehensive faculty scraping...")
        
        # Find all faculty links
        faculty_links = self.find_faculty_links()
        
        if not faculty_links:
            print("❌ No faculty links found!")
            return []
        
        print(f"📄 Scraping {len(faculty_links)} faculty profiles...")
        
        for i, url in enumerate(faculty_links, 1):
            print(f"\n{i}/{len(faculty_links)}: Scraping {url}")
            
            profile = self.scrape_faculty_profile(url)
            if profile and profile['full_content'].strip():
                self.faculty_profiles.append(profile)
                print(f"✅ Successfully scraped profile")
                
                # Show preview
                preview = profile['full_content'][:200] + "..." if len(profile['full_content']) > 200 else profile['full_content']
                print(f"   Preview: {preview}")
            else:
                print(f"⚠️ No content found")
            
            # Rate limiting
            time.sleep(1)
        
        print(f"\n🎉 Scraping completed! Got {len(self.faculty_profiles)} faculty profiles")
        return self.faculty_profiles
    
    def save_faculty_data(self):
        """Save scraped faculty data to JSON file"""
        filename = f"faculty_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(self.faculty_profiles, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Saved faculty data to {filename}")
        return filename

def update_knowledge_base_with_faculty(faculty_profiles):
    """Update the RAG system with scraped faculty data"""
    print("🧠 Updating knowledge base with faculty profiles...")
    
    try:
        from database import db_manager
        from main import rag_system
        
        documents_added = 0
        
        for profile in faculty_profiles:
            # Create document for database
            doc_data = {
                'id': f"faculty_profile_{hash(profile['url'])}",
                'url': profile['url'],
                'title': f"Faculty Profile - {profile.get('name', 'Unknown')}",
                'content': f"Faculty Profile - {profile.get('name', 'Unknown')}\n\n" + profile['full_content'],
                'content_type': 'faculty_profile',
                'extra_metadata': {
                    'name': profile.get('name', ''),
                    'email': profile.get('email', ''),
                    'qualifications': profile.get('qualifications', ''),
                    'specialization': profile.get('specialization', ''),
                    'experience': profile.get('experience', ''),
                    'scraped_at': profile['scraped_at']
                }
            }
            
            try:
                # Store in database
                db_manager.store_university_data(doc_data)
                documents_added += 1
                print(f"✅ Added profile for {profile.get('name', 'Unknown')}")
            except Exception as e:
                print(f"⚠️ Error storing profile: {e}")
        
        print(f"📚 Added {documents_added} faculty profiles to database")
        
        # Update RAG system
        print("🔄 Updating RAG system...")
        rag_system.update_from_database()
        
        # Get updated stats
        stats = rag_system.get_statistics()
        print(f"✅ RAG system updated: {stats['total_vectors']} vectors, {stats['total_documents']} documents")
        
        return documents_added
        
    except Exception as e:
        print(f"❌ Error updating knowledge base: {e}")
        return 0

def main():
    """Main function to scrape all faculty and update the system"""
    print("🎓 COMPREHENSIVE FACULTY SCRAPER")
    print("=" * 50)
    print("Scraping ALL faculty profiles from GD Goenka University School of Engineering")
    print()
    
    scraper = FacultyScraper()
    
    # Scrape all faculty
    faculty_profiles = scraper.scrape_all_faculty()
    
    if not faculty_profiles:
        print("❌ No faculty profiles scraped!")
        return
    
    # Save data
    filename = scraper.save_faculty_data()
    
    # Update knowledge base
    updated_count = update_knowledge_base_with_faculty(faculty_profiles)
    
    print("\n" + "=" * 50)
    print("📊 SCRAPING SUMMARY")
    print("=" * 50)
    print(f"Faculty profiles scraped: {len(faculty_profiles)}")
    print(f"Profiles added to database: {updated_count}")
    print(f"Data saved to: {filename}")
    print("\n🎉 Faculty data scraping and training completed!")
    print("Your RAG system now has comprehensive faculty information!")

if __name__ == "__main__":
    main()
