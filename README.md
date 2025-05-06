# X Post Cleaner

A tool to automatically scan and remove problematic posts from your X (formerly Twitter) archive using a local multimodal LLM.

## Overview

X Post Cleaner is designed to help you clean your X account history by automatically analyzing and removing potentially problematic posts. It uses a multimodal LLM (Large Language Model) to analyze both text and images in your posts, including context from replies and quoted tweets, to identify content that might be inappropriate for professional settings.

## Features

- **Smart Content Analysis**: Uses a multimodal LLM to analyze post text and images
- **Context-Aware**: Fetches and analyzes reply/quote context for more accurate classification
- **Automated Deletion**: Removes problematic posts automatically via browser automation
- **Progress Tracking**: Maintains progress across sessions with checkpointing
- **Image Analysis**: Downloads and analyzes images in posts when using multimodal mode
- **Detailed Logging**: Comprehensive logging of all actions and decisions

## Prerequisites

- Python 3.8+
- A multimodal LLM accessible via an API compatible with OpenAI's API (local or remote)
- Downloaded X archive containing your tweets (tweets.js file)
- The following Python packages (automatically installed with UV):
  - playwright
  - requests
  - tqdm
  - others as listed in requirements

## Installation

You can install X Post Cleaner using Astral's UV tool for fast dependency installation:

```bash
# Install UV (if not already installed)
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/astral-sh/uv/releases/download/0.7.0/uv-installer.sh | sh

# Clone the repository
git clone https://github.com/yourusername/x-post-cleaner.git
cd x-post-cleaner

# Create a virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt

# Install Playwright browsers
python -m playwright install chromium
```

## Usage

To run X Post Cleaner:

```bash
python x_post_cleaner.py --tweets-file "path/to/your/tweets.js" --llm-studio-url "http://localhost:1234/v1/chat/completions"
```

### Command Line Arguments

- `--tweets-file`: Path to your tweets.js file from the X data archive (required)
- `--llm-studio-url`: URL of the LLM API service (default: "http://localhost:1234/v1/chat/completions")
- `--progress-file`: Path to save progress (default: "progress.pkl")
- `--no-multimodal`: Disable multimodal LLM analysis (images processing)
- `--delay`: Delay in seconds between processing tweets (min 2, default 3)

## How It Works

1. **Load Tweets**: Parses your X archive's tweets.js file
2. **Browser Automation**: Launches a browser session for X.com
3. **Login**: Helps you log into your X account (manual login required the first time)
4. **Process Posts**: For each post:
   - Extracts content (text and images)
   - Gets context from replies/quotes if necessary
   - Uses an LLM to analyze the content
   - If problematic, deletes the post automatically
5. **Track Progress**: Saves progress to continue from where you left off

## Notes

- The first run requires manual login to X.com
- Subsequent runs attempt to use saved cookies
- Processing speed is controlled by the `--delay` parameter (adjust as needed)
- Deleted posts cannot be recovered, use with caution

## Project Structure

- `x_post_cleaner.py`: Main script
- `requirements.txt`: Required Python packages
- `x_post_cleaner.log`: Log file of all actions
- `progress.pkl`: Progress tracking file
- `downloaded_images/`: Downloaded post images
- `context_images/`: Images from replied-to/quoted tweets

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for personal use to help clean your own social media history. Use responsibly and at your own risk. The accuracy of problematic content detection depends on the LLM model used.