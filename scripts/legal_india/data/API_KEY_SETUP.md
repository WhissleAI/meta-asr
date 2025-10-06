# YouTube Data API v3 Setup Guide

## How to Get a YouTube API Key

### Step 1: Go to Google Cloud Console
1. Visit [Google Cloud Console](https://console.cloud.google.com/)
2. Sign in with your Google account

### Step 2: Create a New Project (or select existing)
1. Click on the project dropdown at the top
2. Click "New Project"
3. Enter a project name (e.g., "YouTube Downloader")
4. Click "Create"

### Step 3: Enable YouTube Data API v3
1. In the search bar, type "YouTube Data API v3"
2. Click on "YouTube Data API v3"
3. Click "Enable"

### Step 4: Create Credentials
1. Go to "Credentials" in the left sidebar
2. Click "Create Credentials" → "API Key"
3. Copy the generated API key

### Step 5: (Optional) Restrict the API Key
1. Click on the API key you just created
2. Under "API restrictions", select "Restrict key"
3. Choose "YouTube Data API v3"
4. Click "Save"

## Usage

Once you have your API key, use it with the script:

```bash
python3 youtube_api_downloader.py "live supreme court hearing india" --api-key YOUR_API_KEY --require-transcript en --max-results 100 --max-workers 10 --downloads-dir /external4/datasets/legal_data/live_supreme_court_hearing_today_india
```

## API Quotas

- **Free tier**: 10,000 units per day
- **Search request**: 100 units
- **Video details request**: 1 unit per video
- **Estimated capacity**: ~100 videos per day (free tier)

## Benefits of Using API vs Scraping

1. **Reliable**: No bot detection issues
2. **Legitimate**: Official Google service
3. **Structured data**: Clean, consistent metadata
4. **Rate limiting**: Built-in, predictable limits
5. **Future-proof**: Less likely to break with YouTube changes
