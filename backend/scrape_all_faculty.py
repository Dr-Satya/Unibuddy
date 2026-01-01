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
            # Faculty listing pages
                "https://www.gdgoenkauniversity.com/faculty/ameet-singh",
    "https://www.gdgoenkauniversity.com/faculty/deepanshu-bola",
    "https://www.gdgoenkauniversity.com/faculty/dr-anil-kumar",
    "https://www.gdgoenkauniversity.com/faculty/dr-ramandeep-kaur",
    "https://www.gdgoenkauniversity.com/faculty/dr-aadya-prasad",
    "https://www.gdgoenkauniversity.com/faculty/dr-aashi-bhatnagar",
    "https://www.gdgoenkauniversity.com/faculty/dr-abhishek-jha",
    "https://www.gdgoenkauniversity.com/faculty/dr-adiba-ali",
    "https://www.gdgoenkauniversity.com/faculty/dr-akhilesh-latoria",
    "https://www.gdgoenkauniversity.com/faculty/dr-alok-srivatava",
    "https://www.gdgoenkauniversity.com/faculty/dr-anand-kumar-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-aneesha-pr",
    "https://www.gdgoenkauniversity.com/faculty/dr-anil-kumar-gupta",
    "https://www.gdgoenkauniversity.com/faculty/dr-anindita-roy-chowdhury",
    "https://www.gdgoenkauniversity.com/faculty/dr-anitha-arvind",
    "https://www.gdgoenkauniversity.com/faculty/dr-anjali-vyas",
    "https://www.gdgoenkauniversity.com/faculty/dr-ankita-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-anshu-sharma",
    "https://www.gdgoenkauniversity.com/faculty/dr-anu-gupta",
    "https://www.gdgoenkauniversity.com/faculty/dr-anureet-kaur",
    "https://www.gdgoenkauniversity.com/faculty/dr-apoorva-dixit",
    "https://www.gdgoenkauniversity.com/faculty/dr-apoorva-singh-katiyar",
    "https://www.gdgoenkauniversity.com/faculty/dr-archan-rani",
    "https://www.gdgoenkauniversity.com/faculty/dr-arti",
    "https://www.gdgoenkauniversity.com/faculty/dr-aruna-maheshwari",
    "https://www.gdgoenkauniversity.com/faculty/dr-ashi-saif",
    "https://www.gdgoenkauniversity.com/faculty/dr-ashok-yadav",
    "https://www.gdgoenkauniversity.com/faculty/dr-ashu",
    "https://www.gdgoenkauniversity.com/faculty/dr-aslesha-bodavula",
    "https://www.gdgoenkauniversity.com/faculty/dr-azad-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-bhagat-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-chandravilash-rai",
    "https://www.gdgoenkauniversity.com/faculty/dr-dakshita-sangwan",
    "https://www.gdgoenkauniversity.com/faculty/dr-deepika-garg",
    "https://www.gdgoenkauniversity.com/faculty/dr-dinkar-verma",
    "https://www.gdgoenkauniversity.com/faculty/dr-divya",
    "https://www.gdgoenkauniversity.com/faculty/dr-divya-goyal",
    "https://www.gdgoenkauniversity.com/faculty/dr-gajendra-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-jyoti-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-kamna-chibber",
    "https://www.gdgoenkauniversity.com/faculty/dr-kanika-wadhwa",
    "https://www.gdgoenkauniversity.com/faculty/dr-khushbu-parik",
    "https://www.gdgoenkauniversity.com/faculty/dr-kirti-amresh-gautam",
    "https://www.gdgoenkauniversity.com/faculty/dr-kriti-kaushik",
    "https://www.gdgoenkauniversity.com/faculty/dr-kritika-sharma",
    "https://www.gdgoenkauniversity.com/faculty/dr-leena-chhabra",
    "https://www.gdgoenkauniversity.com/faculty/dr-mainak-basu",
    "https://www.gdgoenkauniversity.com/faculty/dr-manish-kumar",
    "https://www.gdgoenkauniversity.com/faculty/dr-manish-yadav",
    "https://www.gdgoenkauniversity.com/faculty/dr-manpreet-ola",
    "https://www.gdgoenkauniversity.com/faculty/dr-manu-banga",
    "https://www.gdgoenkauniversity.com/faculty/dr-megha-gupta",
    "https://www.gdgoenkauniversity.com/faculty/dr-mir-mohsin",
    "https://www.gdgoenkauniversity.com/faculty/dr-mohammad-kamran-ahsan",
    "https://www.gdgoenkauniversity.com/faculty/dr-mohit-mangla",
    "https://www.gdgoenkauniversity.com/faculty/dr-monica-rose",
    "https://www.gdgoenkauniversity.com/faculty/dr-nancy-arya",
    "https://www.gdgoenkauniversity.com/faculty/dr-naresh-sharma",
    "https://www.gdgoenkauniversity.com/faculty/dr-neetu",
    "https://www.gdgoenkauniversity.com/faculty/dr-neetu-ahmed",
    "https://www.gdgoenkauniversity.com/faculty/dr-nidhi-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-parul-mishra",
    "https://www.gdgoenkauniversity.com/faculty/dr-parvesh-lata",
    "https://www.gdgoenkauniversity.com/faculty/dr-pinaz-nasim",
    "https://www.gdgoenkauniversity.com/faculty/dr-pooja-mathur",
    "https://www.gdgoenkauniversity.com/faculty/dr-prabha-arya",
    "https://www.gdgoenkauniversity.com/faculty/dr-preeti-malhotra",
    "https://www.gdgoenkauniversity.com/faculty/dr-prerna-sharma",
    "https://www.gdgoenkauniversity.com/faculty/dr-priya",
    "https://www.gdgoenkauniversity.com/faculty/dr-priyanka-rohatgi",
    "https://www.gdgoenkauniversity.com/faculty/dr-priyanka-sharma",
    "https://www.gdgoenkauniversity.com/faculty/dr-promil-pande",
    "https://www.gdgoenkauniversity.com/faculty/dr-raj-kumar",
    "https://www.gdgoenkauniversity.com/faculty/dr-rajesh-yadav",
    "https://www.gdgoenkauniversity.com/faculty/dr-raunak-dhanker",
    "https://www.gdgoenkauniversity.com/faculty/dr-rekha-kaushal",
    "https://www.gdgoenkauniversity.com/faculty/dr-rinkal-chaudhary",
    "https://www.gdgoenkauniversity.com/faculty/dr-ritu-malik",
    "https://www.gdgoenkauniversity.com/faculty/dr-rubina-bhutani",
    "https://www.gdgoenkauniversity.com/faculty/dr-sangita-shrama",
    "https://www.gdgoenkauniversity.com/faculty/dr-sapna-bansal",
    "https://www.gdgoenkauniversity.com/faculty/dr-sarita-devi",
    "https://www.gdgoenkauniversity.com/faculty/dr-satya-prakash",
    "https://www.gdgoenkauniversity.com/faculty/dr-shalini-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-shashi-bala",
    "https://www.gdgoenkauniversity.com/faculty/dr-shashikant-gupta",
    "https://www.gdgoenkauniversity.com/faculty/dr-sheetal-malhan",
    "https://www.gdgoenkauniversity.com/faculty/dr-shilpa",
    "https://www.gdgoenkauniversity.com/faculty/dr-shipra-kataria",
    "https://www.gdgoenkauniversity.com/faculty/dr-shraddha-agrawal",
    "https://www.gdgoenkauniversity.com/faculty/dr-shraddha-oberoi",
    "https://www.gdgoenkauniversity.com/faculty/dr-shradhey-gupta",
    "https://www.gdgoenkauniversity.com/faculty/dr-shweta-gehlout",
    "https://www.gdgoenkauniversity.com/faculty/dr-smita-kumari",
    "https://www.gdgoenkauniversity.com/faculty/dr-soumita-talukdar",
    "https://www.gdgoenkauniversity.com/faculty/dr-sudesh-kumar-aryan",
    "https://www.gdgoenkauniversity.com/faculty/dr-sudipta-k-mishra",
    "https://www.gdgoenkauniversity.com/faculty/dr-sunanda-vashisth",
    "https://www.gdgoenkauniversity.com/faculty/dr-sunrita-chaudhari",
    "https://www.gdgoenkauniversity.com/faculty/dr-swati-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-swati-sisodia",
    "https://www.gdgoenkauniversity.com/faculty/dr-syed-hameedur",
    "https://www.gdgoenkauniversity.com/faculty/dr-syed-mohd-jamal-mahmood",
    "https://www.gdgoenkauniversity.com/faculty/dr-tabish-fahim",
    "https://www.gdgoenkauniversity.com/faculty/dr-teena",
    "https://www.gdgoenkauniversity.com/faculty/dr-ugur-guven",
    "https://www.gdgoenkauniversity.com/faculty/dr-uzma-rukhsar",
    "https://www.gdgoenkauniversity.com/faculty/dr-vikas-jhawat",
    "https://www.gdgoenkauniversity.com/faculty/dr-vinod-kumar",
    "https://www.gdgoenkauniversity.com/faculty/dr-virendra-kumar",
    "https://www.gdgoenkauniversity.com/faculty/dr-vishakha",
    "https://www.gdgoenkauniversity.com/faculty/dr-vishwaieet-trivedi",
    "https://www.gdgoenkauniversity.com/faculty/dr-vrushali-pathak",
    "https://www.gdgoenkauniversity.com/faculty/drpriya-goyal",
    "https://www.gdgoenkauniversity.com/faculty/ena-goel",
    "https://www.gdgoenkauniversity.com/faculty/felnunmoi-gangte",
    "https://www.gdgoenkauniversity.com/faculty/hitesh-kumar",
    "https://www.gdgoenkauniversity.com/faculty/lakshmi-surya",
    "https://www.gdgoenkauniversity.com/faculty/manohar-karmakar",
    "https://www.gdgoenkauniversity.com/faculty/mayank-tiwari",
    "https://www.gdgoenkauniversity.com/faculty/md-aamir-nayyar",
    "https://www.gdgoenkauniversity.com/faculty/mr-afeef-abdul-kader",
    "https://www.gdgoenkauniversity.com/faculty/mr-amit-kumar",
    "https://www.gdgoenkauniversity.com/faculty/mr-amit-singh-slathia",
    "https://www.gdgoenkauniversity.com/faculty/mr-ashish-parihar",
    "https://www.gdgoenkauniversity.com/faculty/mr-daksh-mehta",
    "https://www.gdgoenkauniversity.com/faculty/mr-devansh-sagar-mall",
    "https://www.gdgoenkauniversity.com/faculty/mr-fakhruddin-ali-ahmad",
    "https://www.gdgoenkauniversity.com/faculty/mr-girish-ahuja",
    "https://www.gdgoenkauniversity.com/faculty/mr-harbir-singh",
    "https://www.gdgoenkauniversity.com/faculty/mr-jai-prabhat-ranjan",
    "https://www.gdgoenkauniversity.com/faculty/mr-krishna-kumar-gupta",
    "https://www.gdgoenkauniversity.com/faculty/mr-kunal-dahiya",
    "https://www.gdgoenkauniversity.com/faculty/mr-lalji-maurya",
    "https://www.gdgoenkauniversity.com/faculty/mr-mahesh-kumar",
    "https://www.gdgoenkauniversity.com/faculty/mr-mohd-naveed-khan",
    "https://www.gdgoenkauniversity.com/faculty/mr-mudit-jain",
    "https://www.gdgoenkauniversity.com/faculty/mr-mukul-sharma",
    "https://www.gdgoenkauniversity.com/faculty/mr-narender-sharma",
    "https://www.gdgoenkauniversity.com/faculty/mr-neeraj-singh",
    "https://www.gdgoenkauniversity.com/faculty/mr-pankaj-avasthi",
    "https://www.gdgoenkauniversity.com/faculty/mr-prakash-moorthy",
    "https://www.gdgoenkauniversity.com/faculty/mr-rahul-kumar",
    "https://www.gdgoenkauniversity.com/faculty/mr-saurabh-shekhar",
    "https://www.gdgoenkauniversity.com/faculty/mr-sunny-saxena",
    "https://www.gdgoenkauniversity.com/faculty/mr-triloki-singh",
    "https://www.gdgoenkauniversity.com/faculty/mr-tushaar-sonkar",
    "https://www.gdgoenkauniversity.com/faculty/mr-yogindra-ashok-vaidya",
    "https://www.gdgoenkauniversity.com/faculty/ms-aakansha-singh",
    "https://www.gdgoenkauniversity.com/faculty/ms-amali",
    "https://www.gdgoenkauniversity.com/faculty/ms-anjali-khantal",
    "https://www.gdgoenkauniversity.com/faculty/ms-bhawna-goel",
    "https://www.gdgoenkauniversity.com/faculty/ms-chanda",
    "https://www.gdgoenkauniversity.com/faculty/ms-chintakayal-purnima",
    "https://www.gdgoenkauniversity.com/faculty/ms-deepali-bedi",
    "https://www.gdgoenkauniversity.com/faculty/ms-diksha-dubey",
    "https://www.gdgoenkauniversity.com/faculty/ms-heena-parveen",
    "https://www.gdgoenkauniversity.com/faculty/ms-jagriti-gaba",
    "https://www.gdgoenkauniversity.com/faculty/ms-jaya-sharma",
    "https://www.gdgoenkauniversity.com/faculty/ms-jesbin-johnson",
    "https://www.gdgoenkauniversity.com/faculty/ms-judith-kuiur",
    "https://www.gdgoenkauniversity.com/faculty/ms-jyoti-ahlawat",
    "https://www.gdgoenkauniversity.com/faculty/ms-kamna-sarin",
    "https://www.gdgoenkauniversity.com/faculty/ms-kavita-pandey",
    "https://www.gdgoenkauniversity.com/faculty/ms-komal-rani",
    "https://www.gdgoenkauniversity.com/faculty/ms-laxmi-chauhan",
    "https://www.gdgoenkauniversity.com/faculty/ms-laxmi-rani",
    "https://www.gdgoenkauniversity.com/faculty/ms-mahi-khare",
    "https://www.gdgoenkauniversity.com/faculty/ms-maliha-sultan-chaudhry",
    "https://www.gdgoenkauniversity.com/faculty/ms-mandeep-grewal",
    "https://www.gdgoenkauniversity.com/faculty/ms-mimansa-singh-tanwar",
    "https://www.gdgoenkauniversity.com/faculty/ms-monika",
    "https://www.gdgoenkauniversity.com/faculty/ms-nainika-makhija",
    "https://www.gdgoenkauniversity.com/faculty/ms-neha-jaiswal",
    "https://www.gdgoenkauniversity.com/faculty/ms-nidhi-sharma",
    "https://www.gdgoenkauniversity.com/faculty/ms-palak-khurana",
    "https://www.gdgoenkauniversity.com/faculty/ms-parisha",
    "https://www.gdgoenkauniversity.com/faculty/ms-pooja",
    "https://www.gdgoenkauniversity.com/faculty/ms-pooja-trehan",
    "https://www.gdgoenkauniversity.com/faculty/ms-poonam-yadav",
    "https://www.gdgoenkauniversity.com/faculty/ms-pragya-sachdeva",
    "https://www.gdgoenkauniversity.com/faculty/ms-priya-arora",
    "https://www.gdgoenkauniversity.com/faculty/ms-priyanka",
    "https://www.gdgoenkauniversity.com/faculty/ms-priyanka-nair",
    "https://www.gdgoenkauniversity.com/faculty/ms-riddhima-singh",
    "https://www.gdgoenkauniversity.com/faculty/ms-roshni-sengupta",
    "https://www.gdgoenkauniversity.com/faculty/ms-rosy-gehlaut",
    "https://www.gdgoenkauniversity.com/faculty/ms-sakshi-singhal",
    "https://www.gdgoenkauniversity.com/faculty/ms-shefali-shelat",
    "https://www.gdgoenkauniversity.com/faculty/ms-shipra-khanna",
    "https://www.gdgoenkauniversity.com/faculty/ms-shreya-goswami",
    "https://www.gdgoenkauniversity.com/faculty/ms-shweta-kumari",
    "https://www.gdgoenkauniversity.com/faculty/ms-sonam-chaudhary",
    "https://www.gdgoenkauniversity.com/faculty/ms-sunita-sahu",
    "https://www.gdgoenkauniversity.com/faculty/ms-tasleem-khanam",
    "https://www.gdgoenkauniversity.com/faculty/ms-tasnim-jahan",
    "https://www.gdgoenkauniversity.com/faculty/ms-tejaswi",
    "https://www.gdgoenkauniversity.com/faculty/ms-vaishali-tyagi",
    "https://www.gdgoenkauniversity.com/faculty/ms-yukti-yadav",
    "https://www.gdgoenkauniversity.com/faculty/onkar-gayakwad",
    "https://www.gdgoenkauniversity.com/faculty/pragya-jain",
    "https://www.gdgoenkauniversity.com/faculty/dr-aashish-sharma",
    "https://www.gdgoenkauniversity.com/faculty/dr-anjali-midha-sharan",
    "https://www.gdgoenkauniversity.com/faculty/dr-deepti-wadera",
    "https://www.gdgoenkauniversity.com/faculty/dr-jyoti-shrivastava",
    "https://www.gdgoenkauniversity.com/faculty/dr-k-thammi-reddy",
    "https://www.gdgoenkauniversity.com/faculty/dr-rahul-pratap-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-rahul-singh",
    "https://www.gdgoenkauniversity.com/faculty/dr-saahil-arora",
    "https://www.gdgoenkauniversity.com/faculty/dr-vandana-mehrotra",
    "https://www.gdgoenkauniversity.com/faculty/ankur-gulati",
    "https://www.gdgoenkauniversity.com/faculty/manish-iyer",
    "https://www.gdgoenkauniversity.com/faculty/rahman-zaini",
    "https://www.gdgoenkauniversity.com/faculty/saloni-agarwal",
    "https://www.gdgoenkauniversity.com/faculty/shrey-subodh",
    "https://www.gdgoenkauniversity.com/faculty/shreyas-kalia",
    "https://www.gdgoenkauniversity.com/faculty/sumedha-garg",
    "https://www.gdgoenkauniversity.com/faculty/sunil-verma",
    "https://www.gdgoenkauniversity.com/faculty/tarun-saish-sampathirao",
    "https://www.gdgoenkauniversity.com/faculty/vedang-kaushik",
    "https://www.gdgoenkauniversity.com/faculty/yashika-munjal"
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
