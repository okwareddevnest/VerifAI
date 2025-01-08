import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
import re
from urllib.parse import urlparse
import nltk
from nltk.tokenize import sent_tokenize

class ArticleProcessor:
    def __init__(self):
        # Download required NLTK data
        nltk.download('punkt')
        
        # Common headers to avoid being blocked
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """Extract the main article content from HTML"""
        # Try common article content patterns
        article_tags = soup.find_all(['article', 'main'])
        if article_tags:
            return ' '.join(tag.get_text() for tag in article_tags)
        
        # Look for content in paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            return ' '.join(p.get_text() for p in paragraphs)
        
        # Fallback to body content
        return soup.get_text()
    
    def _get_domain(self, url: str) -> str:
        """Extract the domain name from URL"""
        parsed = urlparse(url)
        return parsed.netloc
    
    def _summarize_text(self, text: str, max_sentences: int = 3) -> str:
        """Create a brief summary of the article text"""
        sentences = sent_tokenize(text)
        
        if len(sentences) <= max_sentences:
            return text
        
        # Simple extractive summarization
        # In a real implementation, you might want to use a more sophisticated approach
        return ' '.join(sentences[:max_sentences])
    
    def process_url(self, url: str) -> Dict:
        """Process an article from URL"""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = soup.title.string if soup.title else ''
            
            # Extract main content
            content = self._extract_main_content(soup)
            cleaned_content = self._clean_text(content)
            
            # Create article data
            article_data = {
                'url': url,
                'domain': self._get_domain(url),
                'title': title,
                'text': cleaned_content,
                'summary': self._summarize_text(cleaned_content),
                'html': response.text  # Keep original HTML for potential further processing
            }
            
            return article_data
            
        except Exception as e:
            print(f"Error processing URL {url}: {str(e)}")
            return None
    
    def process_text(self, text: str) -> Dict:
        """Process raw article text"""
        cleaned_text = self._clean_text(text)
        
        return {
            'url': None,
            'domain': None,
            'title': None,
            'text': cleaned_text,
            'summary': self._summarize_text(cleaned_text),
            'html': None
        }
