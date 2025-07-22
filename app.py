import os
import re
import string
import random
import tempfile
import base64
from dotenv import load_dotenv
import streamlit as st
import json
import boto3
import requests
import concurrent.futures
import pandas as pd
from jinja2 import Template
from openai import AzureOpenAI
from datetime import datetime, timezone
from json.decoder import JSONDecodeError

# ── Load .env ─────────────────────────────────────────────────────
load_dotenv()

# ── Configuration ─────────────────────────────────────────────────
# Azure OpenAI
AZURE_ENDPOINT = st.secrets["azure_openai"]["endpoint"]
AZURE_API_KEY  = st.secrets["azure_openai"]["api_key"]

# AWS
AWS_ACCESS_KEY = st.secrets["aws"]["access_key"]
AWS_SECRET_KEY = st.secrets["aws"]["secret_key"]
AWS_REGION     = st.secrets["aws"]["region"]
AWS_BUCKET     = st.secrets["aws"]["bucket"]
S3_PREFIX      = st.secrets["aws"]["s3_prefix"]
CDN_BASE       = st.secrets["aws"]["cdn_base"]
IMAGE_FOLDER   = st.secrets["aws"]["image_folder"]
CLOUDFRONT_BASE   = st.secrets["aws"]["CLOUDFRONT_BASE"]
# Pexels
PEXELS_API_KEY = st.secrets["pexels"]["api_key"]

# ── Clients ────────────────────────────────────────────────────────
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2025-01-01-preview"
)

s3 = boto3.client(
    "s3",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# ── Load Templates ─────────────────────────────────────────────────
prompt_template = Template(open("prompt_template.txt", "r", encoding="utf-8").read())
html_template   = Template(open("templates/master_template_org_updated.html", "r", encoding="utf-8").read())

# ── Helpers ────────────────────────────────────────────────────────
def search_pexels_image(query, index):
    url = "https://api.pexels.com/v1/search"
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": index + 1, "orientation": "portrait", "size": "medium"}
    photos = requests.get(url, headers=headers, params=params).json().get("photos", [])
    return photos[index]["src"]["original"] if len(photos) > index else None

def generate_resized_url(bucket: str, key: str, width: int, height: int, fit: str = "cover") -> str:
    instructions = {
        "bucket": bucket,
        "key": key,
        "edits": {
            "resize": {
                "width": width,
                "height": height,
                "fit": fit
            }
        }
    }
    payload = json.dumps(instructions).encode("utf-8")
    b64 = base64.urlsafe_b64encode(payload).decode("utf-8")
    return f"{CLOUDFRONT_BASE}/{b64}"

def generate_slug_and_urls(title):
    if not title or not isinstance(title, str):
        raise ValueError("Invalid title")
    slug = ''.join(
        c for c in title.lower().replace(" ", "-").replace("_", "-")
        if c in string.ascii_lowercase + string.digits + '-'
    ).strip('-')
    nano = ''.join(random.choices(string.ascii_letters + string.digits + '_-', k=10)) + '_G'
    slug_nano = f"{slug}_{nano}"
    return (
        nano,
        slug_nano,
        f"https://suvichaar.org/stories/{slug_nano}",
        f"{slug_nano}.html"
    )

def clean_filter_tags(data):
    """Extract & normalize filter tags from model output."""
    raw_tags = (
        data.get("filterTags")
        or data.get("filtertags")
        or data.get("tags")
        or data.get("metakeywords", "")
    )
    if isinstance(raw_tags, str):
        raw_tags = [t.strip() for t in re.split(r"[,|;]", raw_tags) if t.strip()]
    if not raw_tags:
        return []
    seen = set()
    cleaned = []
    for t in raw_tags:
        key = t.lower()
        if key not in seen:
            seen.add(key)
            cleaned.append(t)
    return cleaned

# ── Streamlit App ─────────────────────────────────────────────────
def main():
    st.title("Dynamic Story & Video Page Generator")
    topic = st.text_input("Enter your topic:")
    language = st.selectbox("Select your Language", ["en-US", "hi"])

    if st.button("Generate Story + Video Page") and topic:
        with st.spinner("Generating story and picking a video..."):
            # 1️⃣ Generate content via Azure OpenAI
            prompt = prompt_template.render(topic=topic)
            resp = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            )

            # Parse JSON safely
            raw_content = resp.choices[0].message.content
            try:
                data = json.loads(raw_content)
            except JSONDecodeError:
                # naive fallback: grab first {...} block
                match = re.search(r"\{.*\}", raw_content, flags=re.S)
                if not match:
                    st.error("Model did not return valid JSON.")
                    return
                data = json.loads(match.group(0))

            # ensure keys exist
            for i in range(1, 10):
                data.setdefault(f"s{i}paragraph1", "")
                data.setdefault(f"s{i}alt1", "")

            # 2️⃣ Randomly pick a video row from CSV
            df_v = pd.read_csv("Video_sheets.csv")
            video_row = df_v.sample(n=1).iloc[0]
            video_context = {
                "s10video1":       video_row["{{s10video1}}"],
                "hookline":        video_row["{{hookline}}"],
                "s10alt1":         video_row["{{s10alt1}}"],
                "videoscreenshot": video_row["{{videoscreenshot}}"],
                "s10caption1":     video_row["{{s10caption1}}"],
            }

            # 3️⃣ Fetch, upload & generate resized URLs
            images = {f"s{i}image1": "" for i in range(1, 10)}
            with concurrent.futures.ThreadPoolExecutor() as pool:
                futures = {pool.submit(search_pexels_image, topic, i): i for i in range(9)}
                for fut, idx in futures.items():
                    url_img = fut.result()
                    if not url_img:
                        continue

                    # download to temp, upload raw to S3
                    fd, tmp_path = tempfile.mkstemp(prefix=f"tmp_{idx+1}_", suffix=".jpg")
                    os.close(fd)
                    with open(tmp_path, "wb") as f:
                        f.write(requests.get(url_img).content)

                    s3_key = f"{IMAGE_FOLDER}/{topic.replace(' ', '_')}_{idx+1}.jpg"
                    with open(tmp_path, "rb") as f_in:
                        s3.upload_fileobj(f_in, AWS_BUCKET, s3_key)
                    os.remove(tmp_path)

                    # generate a 720×1280 resized URL via CloudFront
                    images[f"s{idx+1}image1"] = generate_resized_url(
                        bucket=AWS_BUCKET,
                        key=s3_key,
                        width=720,
                        height=1280,
                        fit="cover"
                    )

            # 4️⃣ Compute timestamps, slug, and URLs
            now_iso = datetime.now(timezone.utc).isoformat(timespec='seconds')
            nano, slug_nano, canurl, html_filename = generate_slug_and_urls(data["storytitle"])
            raw_url   = images.get("s1image1", "")
            pot_image = raw_url  # already resized via CloudFront
            storyhtmlurl = f"https://stories.suvichaar.org/{html_filename}"

            # 5️⃣ Merge everything and render HTML
            html_vars = {
                **data,
                **images,
                **video_context,
                "Topic":              topic,
                "lang":               language,
                "publishedtime":      now_iso,
                "modifiedtime":       now_iso,
                "canurl":             canurl,
                "potraightcoverurl":  pot_image
            }
            html_content = html_template.render(**html_vars)

            # ── Upload final HTML ─────────────────────────────────────────────
            s3.put_object(
                Bucket=HTML_BUCKET,
                Key=html_filename,
                Body=html_content.encode("utf-8"),
                ContentType="text/html",
            )

            # 5.5️⃣ Build & upload JSON metadata
            filter_tags = clean_filter_tags(data)
            story_json = {
                "story_title":      data.get("storytitle", ""),
                "categories":       1,
                "filterTags":       filter_tags,
                "story_uid":        nano,
                "story_link":       canurl,
                "storyhtmlurl":     storyhtmlurl,
                "urlslug":          slug_nano,
                "cover_image_link": images.get("s1image1", ""),
                "publisher_id":     1,
                "story_logo_link":  f"{CDN_BASE}/filters:resize/96x96/media/brandasset/suvichaariconblack.png",
                "keywords":         data.get("metakeywords", ""),
                "metadescription":  data.get("metadescription", ""),
                "lang":             language
            }
            story_json_str = json.dumps(story_json, indent=2, ensure_ascii=False)


            # 6️⃣ Preview & Download
            st.markdown("### Preview HTML")
            st.components.v1.html(html_content, height=600)

            st.markdown(f"🔗 **Live Story URL:** [View your story →]({canurl})")

            st.download_button(
                "Download HTML",
                html_content,
                file_name=html_filename,
                mime="text/html"
            )

            st.download_button(
                "Download JSON",
                story_json_str,
                file_name=f"{slug_nano}.json",
                mime="application/json"
            )

if __name__ == "__main__":
    main()
