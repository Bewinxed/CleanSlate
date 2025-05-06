#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
X Post Cleaner
--------------
A tool to automatically scan and remove problematic posts from your X/Twitter archive
using a local multimodal LLM.

Install dependencies with UV:
    uv pip install playwright requests tqdm pillow base64io
    python -m playwright install chromium

Usage:
    python x_post_cleaner.py --tweets-file "path/to/your/tweets.js"
"""

# PEP 723 dependency metadata for UV
# @dependencies = [
#   "playwright>=1.36.0",
#   "requests>=2.28.0",
#   "tqdm>=4.65.0",
#   "pillow>=9.5.0",
#   "base64io>=1.0.0"
# ]


import json
import os
import re
import time
import logging
import pickle
import requests
import base64
from tqdm import tqdm
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError, Page

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("x_post_cleaner.log", mode='a', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class XPostCleaner:
    def __init__(self, tweets_file, progress_file="progress.pkl", llm_studio_url="http://localhost:1234/v1/chat/completions"):
        self.tweets_file = tweets_file
        self.progress_file = progress_file
        self.llm_studio_url = llm_studio_url
        self.browser = None
        self.page: Page | None = None
        self.processed_tweets = self._load_progress()
        self.tweets = self._load_tweets()
        self.image_dir = "downloaded_images"
        self.context_image_dir = "context_images" # For images from replied-to/quoted tweets
        
        os.makedirs(self.image_dir, exist_ok=True)
        os.makedirs(self.context_image_dir, exist_ok=True)

    def _load_tweets(self):
        """Load and parse the tweets.js file."""
        logger.info(f"Loading tweets from {self.tweets_file}")
        try:
            with open(self.tweets_file, 'r', encoding='utf-8') as f: content = f.read()
            # Remove the JavaScript variable assignment (e.g., "window.YTD.tweets.part0 = [...]")
            json_content = re.sub(r'^window\.YTD\.tweets\.part\d+\s*=\s*', '', content.strip())
            tweets_data = json.loads(json_content)
            # Ensure tweets are a list of objects, each containing a 'tweet' key
            valid_tweets = [item for item in tweets_data if isinstance(item, dict) and "tweet" in item]
            if len(valid_tweets) != len(tweets_data):
                logger.warning(f"Filtered {len(tweets_data) - len(valid_tweets)} invalid tweet entries.")
            logger.info(f"Loaded {len(valid_tweets)} valid tweets")
            return valid_tweets
        except FileNotFoundError: logger.error(f"Tweets file not found: {self.tweets_file}"); raise
        except json.JSONDecodeError as e: logger.error(f"Error decoding JSON from tweets: {e}"); raise
        except Exception as e: logger.error(f"Error loading tweets: {e}", exc_info=True); raise
    
    def _load_progress(self):
        """Load progress from previous run, if available."""
        if os.path.exists(self.progress_file):
            try:
                with open(self.progress_file, 'rb') as f: progress = pickle.load(f)
                if not isinstance(progress, set): # Basic validation
                    logger.warning("Progress file corrupted. Starting fresh."); return set()
                logger.info(f"Loaded progress: {len(progress)} tweets already processed")
                return progress
            except Exception as e:
                logger.error(f"Error loading progress (corrupted/incompatible): {e}. Starting fresh."); return set()
        return set()
    
    def _save_progress(self):
        """Save current progress."""
        try:
            with open(self.progress_file, 'wb') as f: pickle.dump(self.processed_tweets, f)
            logger.debug(f"Progress saved: {len(self.processed_tweets)} tweets processed")
        except Exception as e: logger.error(f"Error saving progress: {e}", exc_info=True)

    def extract_image_urls(self, tweet_data):
        """Extract image URLs from a tweet data structure."""
        # This method needs to be part of the class to be called with self.
        image_urls = set() # Use a set to automatically handle duplicates
        tweet_id_str = tweet_data.get("tweet", {}).get("id_str", "UNKNOWN_ID")
        try:
            # Ensure we are looking inside the 'tweet' object within the tweet_data item
            tweet_content = tweet_data.get("tweet", {}) 
            if not tweet_content: # Handle cases where tweet_data might not have a 'tweet' key
                 logger.warning(f"No 'tweet' key found in tweet_data for ID {tweet_id_str} during image extraction.")
                 return []

            entities = tweet_content.get("entities", {})
            if "media" in entities:
                for media_item in entities["media"]: 
                    if media_item.get("type") == "photo":
                        url = media_item.get("media_url_https") or media_item.get("media_url")
                        if url: image_urls.add(url)
            
            extended_entities = tweet_content.get("extended_entities", {})
            if "media" in extended_entities:
                for media_item in extended_entities["media"]: 
                    if media_item.get("type") == "photo":
                        url = media_item.get("media_url_https") or media_item.get("media_url")
                        if url: image_urls.add(url)
        except Exception as e:
            logger.error(f"Error extracting image URLs for tweet {tweet_id_str}: {e}", exc_info=True)
        
        extracted_list = list(image_urls)
        logger.debug(f"Extracted {len(extracted_list)} image URLs for tweet {tweet_id_str}.")
        return extracted_list

    def _get_llm_image_description(self, image_paths, context_tweet_id):
        """Gets a simple description of images using a multimodal LLM."""
        if not image_paths:
            return []

        descriptions = []
        for image_path in image_paths:
            try:
                with open(image_path, "rb") as img_file: image_data = img_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
                image_format = os.path.splitext(image_path)[1].lower().replace('.', '') or 'jpeg'
                if image_format not in ['jpeg', 'jpg', 'png', 'gif', 'webp']: image_format = 'jpeg'

                system_prompt = "You are an image analysis assistant. Describe the content of the image factually and concisely in one sentence."
                user_prompt_content = [
                    {"type": "text", "text": "Describe this image."},
                    {"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}}
                ]
                
                json_schema_desc = {
                    "name": "image_description_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"description": {"type": "string"}},
                        "required": ["description"]
                    }
                }

                payload = {
                    "model": "default", 
                    "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_content}],
                    "temperature": 0.2, "max_tokens": 100,
                    "response_format": {"type": "json_schema", "json_schema": json_schema_desc}
                }
                response = requests.post(self.llm_studio_url, headers={"Content-Type": "application/json"}, json=payload, timeout=60)

                if response.status_code == 200:
                    resp_json = response.json()
                    content = resp_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    if content:
                        try:
                            desc_data = json.loads(content)
                            descriptions.append(desc_data.get("description", "Could not describe image."))
                        except json.JSONDecodeError:
                            logger.warning(f"LLM image description for {context_tweet_id} was not JSON: {content}")
                            descriptions.append(content) 
                    else: descriptions.append("Empty description from LLM.")
                else:
                    logger.error(f"LLM error describing image {image_path} for context tweet {context_tweet_id}: {response.status_code} - {response.text}")
                    descriptions.append("Error describing image.")
            except Exception as e:
                logger.error(f"Error getting LLM description for image {image_path} (context {context_tweet_id}): {e}", exc_info=True)
                descriptions.append("Exception while describing image.")
        return descriptions

    def _fetch_external_tweet_content_data(self, tweet_id_to_fetch):
        """Fetches text and image content of an external tweet using Playwright and describes images with LLM."""
        if not self.page:
            logger.error("Playwright page not available for fetching external tweet content.")
            return None 
        
        tweet_url = f"https://x.com/i/status/{tweet_id_to_fetch}" 
        logger.info(f"Fetching external context from: {tweet_url}")
        
        scraped_text = "Could not fetch external tweet text."
        image_descriptions = []

        try:
            self.page.goto(tweet_url, timeout=60000, wait_until="domcontentloaded")
            self.page.wait_for_selector('article[data-testid="tweet"]', timeout=30000)
            time.sleep(2) 

            text_selectors = [
                'article[data-testid="tweet"] div[data-testid="tweetText"]', 
                'article[data-testid="tweet"] div[lang][dir="auto"]', 
            ]
            tweet_text_element = None
            for selector in text_selectors:
                if self.page.locator(selector).first.is_visible(timeout=5000):
                    tweet_text_element = self.page.locator(selector).first
                    break
            
            if tweet_text_element:
                scraped_text = tweet_text_element.inner_text()
                logger.info(f"Scraped external text for {tweet_id_to_fetch}: \"{scraped_text[:100]}...\"")
            else:
                logger.warning(f"Could not find tweet text for external tweet {tweet_id_to_fetch} using selectors.")

            # Use locator().all() which is the newer Playwright syntax
            img_elements = self.page.locator('article[data-testid="tweet"] div[data-testid="photos"] img[alt="Image"], article[data-testid="tweet"] div[data-testid="videoPlayer"] img').all()
            
            external_image_urls = []
            for img_el in img_elements:
                src = img_el.get_attribute("src")
                # Filter more strictly for URLs that look like tweet images
                if src and ('twimg.com/media/' in src or 'pbs.twimg.com/media/' in src) and not src.startswith("data:"): 
                    external_image_urls.append(src)
            
            if external_image_urls:
                logger.info(f"Found {len(external_image_urls)} potential images in external tweet {tweet_id_to_fetch}.")
                downloaded_context_image_paths_for_llm = [] 
                for i, img_url in enumerate(external_image_urls):
                    # Pass the correct directory for context images
                    img_path = self.download_image(img_url, f"context_{tweet_id_to_fetch}_{i}", base_dir=self.context_image_dir)
                    if img_path:
                        downloaded_context_image_paths_for_llm.append(img_path)
                
                if downloaded_context_image_paths_for_llm:
                    image_descriptions = self._get_llm_image_description(downloaded_context_image_paths_for_llm, tweet_id_to_fetch)
            
            return {"text": scraped_text, "image_descriptions": image_descriptions}

        except PlaywrightTimeoutError:
            logger.error(f"Timeout fetching external tweet {tweet_id_to_fetch} at {tweet_url}")
        except Exception as e:
            logger.error(f"Error fetching or processing external tweet {tweet_id_to_fetch}: {e}", exc_info=True)
        
        # Return default structure even on failure
        return {"text": scraped_text, "image_descriptions": ["Failed to fetch or analyze context images."]}


    def classify_tweet(self, tweet_data, external_context=None):
        """Classify a post (text-only focus) with optional external context, using JSON schema."""
        tweet_id_str = tweet_data.get("tweet", {}).get("id_str", "UNKNOWN_ID")
        try:
            tweet_obj = tweet_data.get("tweet", {})
            full_text = tweet_obj.get("full_text", "")
            if not full_text:
                logger.warning(f"Tweet {tweet_id_str} has no text. Skipping LLM classification.")
                return {"is_problematic": False, "reason": "No text content"}

            is_reply = "in_reply_to_status_id_str" in tweet_obj and tweet_obj["in_reply_to_status_id_str"] is not None
            is_quote = "quoted_status_id_str" in tweet_obj and tweet_obj.get("quoted_status_id_str") is not None \
                       or "quoted_status" in tweet_obj 

            base_context = "This is a standalone post."
            if is_reply and is_quote: base_context = "This post is a reply and quotes another tweet."
            elif is_reply: base_context = "This post is a reply."
            elif is_quote: base_context = "This post quotes another tweet."

            system_prompt = """You are analyzing User's Post. Your goal is to determine if the User's Post is offensive, racist, homophobic, or otherwise inappropriate for a job background check.
            Flag only clearly problematic content in the User's Post.
            ADDITIONAL CONTEXT from a tweet the User's Post replies to or quotes may be provided. Use this to understand the User's Post better, but your final judgment (is_problematic, reason) MUST be about the User's Post ONLY.
            Return JSON: {"is_problematic": boolean, "reason": "explanation for User's Post or N/A"}."""
            
            user_prompt = f"User's Post: {full_text}\nUser's Post Context: {base_context}\n"
            if external_context:
                user_prompt += f"--- External Context (Replied-to/Quoted Tweet) ---\n"
                user_prompt += f"Text: {external_context.get('text', 'N/A')}\n"
                if external_context.get('image_descriptions'):
                    user_prompt += f"Image Descriptions: {'; '.join(external_context['image_descriptions'])}\n"
                user_prompt += "--- End External Context ---\n"
            user_prompt += "Analyze User's Post and return JSON."
            
            json_schema_definition = {
                "name": "tweet_classification_response", "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_problematic": {"type": "boolean"}, "reason": {"type": "string"}
                    }, "required": ["is_problematic", "reason"]
                }
            }
            payload = {
                "model": "default", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "temperature": 0.1, "max_tokens": 350, 
                "response_format": {"type": "json_schema", "json_schema": json_schema_definition}
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.llm_studio_url, headers=headers, json=payload, timeout=60) 
            
            if response.status_code != 200:
                logger.error(f"LLM Error for {tweet_id_str}: {response.status_code} - {response.text}"); return {"is_problematic": False, "reason": f"LLM API error {response.status_code}"}
            
            llm_response_json = response.json()
            if not llm_response_json.get("choices") or not llm_response_json["choices"][0].get("message") or "content" not in llm_response_json["choices"][0]["message"]:
                logger.error(f"Malformed LLM response for {tweet_id_str}: {llm_response_json}"); return {"is_problematic": False, "reason": "Malformed LLM response"}
                
            assistant_message_content = llm_response_json["choices"][0]["message"]["content"]
            classification = {}
            try: classification = json.loads(assistant_message_content) 
            except json.JSONDecodeError:
                logger.warning(f"LLM response for {tweet_id_str} not direct JSON. Extracting. Raw: {assistant_message_content}")
                match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', assistant_message_content, re.DOTALL) or re.search(r'(\{[\s\S]*?\})', assistant_message_content, re.DOTALL)
                if match:
                    try: classification = json.loads(match.group(1))
                    except json.JSONDecodeError as e: logger.error(f"Failed extracted JSON for {tweet_id_str}: {e}"); classification = {"is_problematic": False, "reason": "JSON parsing failed"}
                else: logger.error(f"No JSON in LLM response for {tweet_id_str}"); classification = {"is_problematic": False, "reason": "No JSON in response"}

            is_problematic_val = classification.get("is_problematic")
            # Ensure boolean conversion
            if isinstance(is_problematic_val, str):
                classification["is_problematic"] = is_problematic_val.lower() == "true"
            elif not isinstance(is_problematic_val, bool):
                classification["is_problematic"] = False 
            
            if not isinstance(classification.get("reason"), str): classification["reason"] = "N/A" if not classification.get("is_problematic") else "Reason not specified"
            
            logger.info(f"Classified post {tweet_id_str}: Problematic={classification['is_problematic']}, Reason='{classification.get('reason', 'N/A')}'")
            return classification
            
        except Exception as e:
            logger.error(f"Unexpected error in classify_tweet for {tweet_id_str}: {e}", exc_info=True)
            return {"is_problematic": False, "reason": f"Unexpected classification error: {str(e)}"}

    def classify_tweet_with_image(self, tweet_data, image_paths, external_context=None):
        """Classify a post with its own images and optional external context, using JSON schema."""
        tweet_id_str = tweet_data.get("tweet", {}).get("id_str", "UNKNOWN_ID")
        try:
            tweet_obj = tweet_data.get("tweet", {})
            full_text = tweet_obj.get("full_text", "")

            is_reply = "in_reply_to_status_id_str" in tweet_obj and tweet_obj["in_reply_to_status_id_str"] is not None
            is_quote = "quoted_status_id_str" in tweet_obj and tweet_obj.get("quoted_status_id_str") is not None \
                       or "quoted_status" in tweet_obj
            
            base_context = "This is a standalone post."
            if is_reply and is_quote: base_context = "This post is a reply and quotes another tweet."
            elif is_reply: base_context = "This post is a reply."
            elif is_quote: base_context = "This post quotes another tweet."

            system_prompt = """You are analyzing User's Post (which includes text and potentially images). Your goal is to determine if the User's Post is offensive, racist, homophobic, or otherwise inappropriate for a job background check.
            Flag only clearly problematic content in the User's Post (text or its images).
            ADDITIONAL CONTEXT from a tweet the User's Post replies to or quotes may be provided (text and image descriptions). Use this to understand the User's Post better, but your final judgment (is_problematic, reason, image_analysis) MUST be about the User's Post ONLY.
            Return JSON: {"is_problematic": boolean, "reason": "explanation for User's Post or N/A", "image_analysis": "analysis of User's Post's images or N/A"}."""
            
            user_prompt_content = [{"type": "text", "text": f"User's Post Text: {full_text}\nUser's Post Context: {base_context}\n"}]

            if external_context:
                ext_context_str = f"--- External Context (Replied-to/Quoted Tweet) ---\n"
                ext_context_str += f"Text: {external_context.get('text', 'N/A')}\n"
                if external_context.get('image_descriptions'):
                    ext_context_str += f"Image Descriptions: {'; '.join(external_context['image_descriptions'])}\n"
                ext_context_str += "--- End External Context ---\n"
                user_prompt_content[0]["text"] += ext_context_str
            
            user_prompt_content[0]["text"] += "Analyze User's Post (including its own images if any attached below) and return JSON."

            valid_user_images_added = 0
            if not image_paths: # If user's post has no images, use the text-only classification
                logger.info(f"User's post {tweet_id_str} has no images. Using text-focused classification (with external context if any).")
                return self.classify_tweet(tweet_data, external_context=external_context)

            for image_path in image_paths: # Images belonging to the user's own tweet
                try:
                    with open(image_path, "rb") as img_file: image_data = img_file.read()
                    base64_image = base64.b64encode(image_data).decode('utf-8')
                    image_format = os.path.splitext(image_path)[1].lower().replace('.', '') or 'jpeg'
                    if image_format not in ['jpeg', 'jpg', 'png', 'gif', 'webp']: image_format = 'jpeg'
                    user_prompt_content.append({"type": "image_url", "image_url": {"url": f"data:image/{image_format};base64,{base64_image}"}})
                    valid_user_images_added +=1
                except Exception as img_err: logger.error(f"Error processing user's image {image_path} for {tweet_id_str}: {img_err}", exc_info=True)
            
            if image_paths and not valid_user_images_added: # Had paths but failed to process any
                 logger.warning(f"User's post {tweet_id_str} had image paths, but none processed. Using text-focused classification.")
                 return self.classify_tweet(tweet_data, external_context=external_context)

            json_schema_definition = {
                "name": "tweet_multimodal_classification_response", "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "is_problematic": {"type": "boolean"}, "reason": {"type": "string"},
                        "image_analysis": {"type": "string", "description": "Analysis of the User's Post's own images, or 'N/A'."}
                    }, "required": ["is_problematic", "reason", "image_analysis"]
                }
            }
            payload = {
                "model": "default", "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_content}],
                "temperature": 0.1, "max_tokens": 550, 
                "response_format": {"type": "json_schema", "json_schema": json_schema_definition}
            }
            headers = {"Content-Type": "application/json"}
            response = requests.post(self.llm_studio_url, headers=headers, json=payload, timeout=150) 
            
            if response.status_code != 200:
                logger.error(f"LLM Error (multimodal) for {tweet_id_str}: {response.status_code} - {response.text}"); return {"is_problematic": False, "reason": f"LLM API error {response.status_code}", "image_analysis": "N/A"}
            
            llm_response_json = response.json()
            if not llm_response_json.get("choices") or not llm_response_json["choices"][0].get("message") or "content" not in llm_response_json["choices"][0]["message"]:
                logger.error(f"Malformed LLM response (multimodal) for {tweet_id_str}: {llm_response_json}"); return {"is_problematic": False, "reason": "Malformed LLM response", "image_analysis": "N/A"}

            assistant_message_content = llm_response_json["choices"][0]["message"]["content"]
            classification = {}
            try: classification = json.loads(assistant_message_content)
            except json.JSONDecodeError:
                logger.warning(f"LLM response (multimodal) for {tweet_id_str} not direct JSON. Extracting. Raw: {assistant_message_content}")
                match = re.search(r'```json\s*(\{[\s\S]*?\})\s*```', assistant_message_content, re.DOTALL) or re.search(r'(\{[\s\S]*?\})', assistant_message_content, re.DOTALL)
                if match:
                    try: classification = json.loads(match.group(1))
                    except json.JSONDecodeError as e: logger.error(f"Failed extracted JSON (multimodal) for {tweet_id_str}: {e}"); classification = {"is_problematic": False, "reason": "JSON parsing failed", "image_analysis": "N/A"}
                else: logger.error(f"No JSON in LLM response (multimodal) for {tweet_id_str}"); classification = {"is_problematic": False, "reason": "No JSON in response", "image_analysis": "N/A"}

            is_problematic_val = classification.get("is_problematic")
            if isinstance(is_problematic_val, str): classification["is_problematic"] = is_problematic_val.lower() == "true"
            elif not isinstance(is_problematic_val, bool): classification["is_problematic"] = False
            
            if not isinstance(classification.get("reason"), str): classification["reason"] = "N/A" if not classification.get("is_problematic") else "Reason not specified"
            if not isinstance(classification.get("image_analysis"), str): classification["image_analysis"] = "N/A"

            logger.info(f"Classified post {tweet_id_str} w/ images: Problematic={classification['is_problematic']}, Reason='{classification.get('reason')}', ImgAnalysis='{classification.get('image_analysis')}'")
            return classification

        except Exception as e:
            logger.error(f"Unexpected error in classify_tweet_with_image for {tweet_id_str}: {e}", exc_info=True)
            return {"is_problematic": False, "reason": f"Unexpected classification error (multimodal): {str(e)}", "image_analysis": "N/A"}
    
    def download_image(self, image_url, image_file_prefix, base_dir=None):
        """Download an image from the given URL."""
        if not image_url:
            logger.warning(f"Empty image URL for {image_file_prefix}")
            return None
        
        target_dir = base_dir if base_dir else self.image_dir

        try:
            parsed_url = urlparse(image_url)
            filename_from_url = os.path.basename(parsed_url.path)
            filename = re.sub(r'[?&=:]', '_', filename_from_url) # Sanitize
            if not filename or len(filename) > 100: 
                extension_match = re.search(r'\.(jpg|jpeg|png|gif|webp)(\?|$)', image_url, re.IGNORECASE)
                extension = extension_match.group(1) if extension_match else 'jpg'
                filename = f"image.{extension}"

            unique_filename = f"{image_file_prefix}_{filename}"
            file_path = os.path.join(target_dir, unique_filename)
            
            response = requests.get(image_url, timeout=30) 
            response.raise_for_status() 
            
            with open(file_path, 'wb') as f: f.write(response.content)
            logger.info(f"Downloaded image for {image_file_prefix} from {image_url} to {file_path}")
            return file_path
        
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"HTTP error {http_err.response.status_code} downloading {image_url} for {image_file_prefix}: {http_err}")
        except requests.exceptions.RequestException as req_err:
            logger.error(f"Request error downloading {image_url} for {image_file_prefix}: {req_err}")
        except Exception as e:
            logger.error(f"Unexpected error downloading image {image_url} for {image_file_prefix}: {e}", exc_info=True)
        return None

    def start_browser(self):
        """Initialize the browser with Playwright."""
        try:
            playwright_instance = sync_playwright().start()
            self.browser = playwright_instance.chromium.launch(headless=False) 
            self.page = self.browser.new_page()
            self.page.set_default_timeout(60000) 
            logger.info("Browser started in non-headless mode with 60s default timeout.")
        except Exception as e: logger.error(f"Error starting browser: {e}", exc_info=True); raise
    
    def login_to_x(self):
        """Handle login to X."""
        if not self.page: raise Exception("Browser page not initialized.")
        try:
            self.page.goto('https://x.com/login')
            logger.info("MANUAL LOGIN REQUIRED in browser. Waiting up to 5 mins...")
            self.page.wait_for_selector('a[aria-label="Profile"], nav[aria-label="Primary"] a[href="/home"], div[data-testid="SideNav_AccountSwitcher_Button"]') 
            logger.info("Login successful.")
            with open('x_cookies.json', 'w') as f: json.dump(self.page.context.cookies(), f)
            logger.info("Saved X cookies.")
        except Exception as e: logger.error(f"Error during X login: {e}", exc_info=True); raise
    
    def restore_login_session(self):
        """Try to restore a previous login session."""
        if not self.page: return False
        try:
            if os.path.exists('x_cookies.json'):
                with open('x_cookies.json', 'r') as f: cookies = json.load(f)
                if not cookies: logger.info("x_cookies.json empty."); return False
                self.page.context.add_cookies(cookies)
                logger.info("Restored X cookies.")
                self.page.goto('https://x.com/home', wait_until="domcontentloaded")
                try:
                    self.page.wait_for_selector('a[aria-label="Profile"], nav[aria-label="Primary"] a[href="/home"]', timeout=45000) 
                    logger.info("Session restored and verified.")
                    return True
                except PlaywrightTimeoutError:
                    logger.warning("Failed to verify restored session. Manual login needed.")
                    self.page.context.clear_cookies(); return False
            else: logger.info("No saved X cookies. Manual login needed."); return False
        except Exception as e: logger.error(f"Error restoring session: {e}", exc_info=True); return False
    
    def delete_tweet(self, tweet_id):
        """Navigate to a post on X and delete it using specific CSS selectors."""
        if not self.page: 
            logger.critical("Browser page not initialized for delete_tweet.")
            raise Exception("Browser page not initialized.")
        post_url = f"https://x.com/i/status/{tweet_id}" 
        logger.info(f"Attempting to delete post: {post_url}")

        try:
            self.page.goto(post_url, wait_until="domcontentloaded")
            
            # Check if tweet is already gone
            unavailable_texts = ["This post is unavailable", "Hmm...this page doesn't exist", "This account doesn't exist"]
            if any(self.page.locator(f"text=/{text}/i").first.is_visible(timeout=3000) for text in unavailable_texts):
                logger.info(f"Post {tweet_id} already deleted/inaccessible."); return True 

            # Find the main article containing our tweet - this is the critical change
            main_tweet = self.page.locator(f'article:has([href*="{tweet_id}"])').first
            
            # Click "More" button within the main tweet only
            logger.debug(f"Attempting to click 'More' button for {tweet_id} using specific selector.")
            more_button = main_tweet.locator('button[data-testid="caret"]')
            more_button.wait_for(state="visible", timeout=15000) 
            more_button.click(timeout=5000)
            logger.info(f"Clicked 'More' button for {tweet_id} using specific selector.")

            # Click "Delete" option from menu
            logger.debug(f"Attempting to click 'Delete option' for {tweet_id} using specific selector.")
            delete_option = self.page.locator("div[role='menuitem'] span:has-text('Delete')")
            delete_option.wait_for(state="visible", timeout=10000)
            delete_option.click(timeout=5000)
            logger.info(f"Clicked 'Delete option' for {tweet_id} using specific selector.")
            
            # Click "Confirm Delete" button
            logger.debug(f"Attempting to click 'Confirm Delete' for {tweet_id} using specific selector.")
            confirm_button = self.page.locator("button:has-text('Delete')")
            confirm_button.wait_for(state="visible", timeout=10000)
            confirm_button.click(timeout=5000)
            logger.info(f"Clicked 'Confirm Delete' for {tweet_id} using specific selector.")
            
            # Fix for the toast notification check - use locator() instead of wait_for_selector()
            try: 
                toast = self.page.locator('[data-testid="toast"]').filter(has_text=re.compile(r'Your post was deleted|Post deleted', re.IGNORECASE))
                toast.wait_for(state="visible", timeout=15000)
                logger.info(f"Post {tweet_id} deleted (toast confirmed).")
            except PlaywrightTimeoutError:
                logger.info(f"Post {tweet_id} deletion initiated. No toast confirmation found, assuming success or page changed.")
                time.sleep(2) 
                if self.page.url == post_url and not any(self.page.locator(f"text=/{text}/i").first.is_visible(timeout=1000) for text in unavailable_texts):
                    logger.warning(f"Post {tweet_id} might still be present after deletion attempt without toast.")
            return True
        
        except PlaywrightTimeoutError as pte:
            logger.warning(f"Playwright timeout during deletion of post {tweet_id}: {pte}. It might be already deleted, restricted, or selectors are now invalid.")
            if self.page and (self.page.url != post_url or any(self.page.locator(f"text=/{text}/i").first.is_visible(timeout=1000) for text in unavailable_texts)):
                logger.info(f"Post {tweet_id} seems to be gone despite earlier timeout.")
                return True
            return False
        except Exception as e:
            logger.error(f"Error deleting post {tweet_id}: {e}", exc_info=True)
            return False
    
    def process_tweets(self, use_multimodal_processing=True):
        """Main processing loop for tweets."""
        try:
            self.start_browser()
            if not self.page: logger.critical("Browser page not initialized. Aborting."); return
            if not self.restore_login_session(): self.login_to_x()
            
            tweets_to_process = [t for t in self.tweets if t["tweet"]["id_str"] not in self.processed_tweets]
            logger.info(f"Processing {len(tweets_to_process)} remaining posts. Multimodal default: {use_multimodal_processing}")
            
            for tweet_data in tqdm(tweets_to_process, desc="Processing Posts"):
                tweet_obj = tweet_data.get("tweet", {})
                tweet_id = tweet_obj.get("id_str", "UNKNOWN_ID")
                tweet_text_full = tweet_obj.get("full_text", "[NO TEXT CONTENT]")
                tweet_text_preview = tweet_text_full[:70].replace("\n", " ")
                logger.info(f"Processing post ID: {tweet_id}, Text: \"{tweet_text_preview}...\"")
                
                external_context_data = None
                # 1. Check for Quoted Tweet (data likely in archive)
                quoted_status_data = tweet_obj.get("quoted_status") 
                if quoted_status_data: 
                    logger.info(f"Post {tweet_id} quotes another tweet. Extracting context from embedded archive data.")
                    quoted_text = quoted_status_data.get("full_text", "Quoted text not available in archive.")
                    quoted_image_urls = self.extract_image_urls({"tweet": quoted_status_data}) 
                    quoted_image_descriptions = []
                    if quoted_image_urls:
                        downloaded_quoted_imgs = []
                        for i, img_url in enumerate(quoted_image_urls):
                            img_path = self.download_image(img_url, f"context_quoted_{tweet_id}_{i}", base_dir=self.context_image_dir)
                            if img_path: downloaded_quoted_imgs.append(img_path)
                        if downloaded_quoted_imgs:
                            quoted_image_descriptions = self._get_llm_image_description(downloaded_quoted_imgs, f"quoted_{tweet_id}")
                    external_context_data = {"text": quoted_text, "image_descriptions": quoted_image_descriptions}
                
                # 2. Check for Reply (data needs fetching if not already covered by quote)
                elif tweet_obj.get("in_reply_to_status_id_str"):
                    replied_to_id = tweet_obj["in_reply_to_status_id_str"]
                    logger.info(f"Post {tweet_id} is a reply to {replied_to_id}. Fetching external context via Playwright.")
                    external_context_data = self._fetch_external_tweet_content_data(replied_to_id)
                
                classification_result = None
                user_image_urls = self.extract_image_urls(tweet_data) # Images of the user's own tweet
                downloaded_user_image_paths = []

                if use_multimodal_processing and user_image_urls:
                    logger.info(f"User's post {tweet_id} has {len(user_image_urls)} image(s). Preparing for multimodal.")
                    for i, img_url in enumerate(user_image_urls):
                        img_path = self.download_image(img_url, f"{tweet_id}_userimg_{i}", base_dir=self.image_dir)
                        if img_path: downloaded_user_image_paths.append(img_path)
                    
                    if downloaded_user_image_paths: 
                        classification_result = self.classify_tweet_with_image(tweet_data, downloaded_user_image_paths, external_context=external_context_data)
                    else: 
                        logger.warning(f"Failed to download user's images for multimodal post {tweet_id}. Using text-focused classification.")
                        classification_result = self.classify_tweet(tweet_data, external_context=external_context_data)
                else: 
                    if user_image_urls: logger.info(f"User's post {tweet_id} has images, but multimodal is off or no images downloaded. Text-focused classification.")
                    classification_result = self.classify_tweet(tweet_data, external_context=external_context_data)
                
                if classification_result and classification_result.get("is_problematic"):
                    logger.warning(f"PROBLEM POST ID: {tweet_id} | Reason: {classification_result.get('reason')} | Content: \"{tweet_text_preview}...\"")
                    if downloaded_user_image_paths and "image_analysis" in classification_result: # Check key exists
                        logger.warning(f"Analysis of User's Image(s) for {tweet_id}: {classification_result.get('image_analysis')}")
                    
                    deleted = self.delete_tweet(tweet_id) 
                    if deleted: logger.info(f"Successfully deleted problematic post {tweet_id}.")
                    else: logger.warning(f"FAILED to delete problematic post {tweet_id}.")
                elif not classification_result:
                     logger.error(f"Classification result was None for tweet {tweet_id}. Skipping deletion.")

                self.processed_tweets.add(tweet_id)
                if len(self.processed_tweets) % 5 == 0: self._save_progress()
                # Use environment variable for delay, default 3s, minimum 2s
                time.sleep(max(2, int(os.environ.get("XPC_DELAY", "3")))) 
            
            self._save_progress(); logger.info(f"Completed processing all {len(tweets_to_process)} targeted posts.")
        except KeyboardInterrupt: logger.warning("Process interrupted. Saving progress..."); self._save_progress()
        except Exception as e: logger.error(f"FATAL ERROR in process_tweets: {e}", exc_info=True); self._save_progress()
        finally:
            if self.browser: self.browser.close(); logger.info("Browser closed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Clean problematic posts from X archive.")
    parser.add_argument("--tweets-file", type=str, required=True, help="Path to tweets.js")
    parser.add_argument("--llm-studio-url", type=str, default="http://localhost:1234/v1/chat/completions", help="LLM Studio API URL")
    parser.add_argument("--progress-file", type=str, default="progress.pkl", help="Progress file")
    parser.add_argument("--no-multimodal", action="store_true", help="Disable multimodal LLM analysis (ON by default)")
    parser.add_argument("--delay", type=int, default=3, help="Delay in seconds between processing tweets (min 2)")

    args = parser.parse_args()
    # Set delay via environment variable for access within the class if needed, ensure minimum
    os.environ["XPC_DELAY"] = str(max(2, args.delay)) 

    use_multimodal_on_default = not args.no_multimodal
    logger.info(f"Starting XPostCleaner. Multimodal: {use_multimodal_on_default}. Delay: {os.environ['XPC_DELAY']}s")

    cleaner = XPostCleaner(
        tweets_file=args.tweets_file,
        progress_file=args.progress_file,
        llm_studio_url=args.llm_studio_url
    )
    cleaner.process_tweets(use_multimodal_processing=use_multimodal_on_default)
