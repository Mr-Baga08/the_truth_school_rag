"""
URL Document Fetcher for RAG-Anything

Fetches and processes documents from URLs for ingestion into the RAG system.

Features:
- Web page scraping and parsing
- PDF download from URLs
- Markdown conversion
- Content cleaning and preprocessing
- Advanced parsing with text and image extraction
- Integration with RAG pipeline

Author: RAG-Anything Team
Version: 2.0.0
"""

import os
import asyncio
import logging
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import hashlib
import base64

logger = logging.getLogger(__name__)

try:
    import requests
    from bs4 import BeautifulSoup
    import markdownify
    from urllib.parse import urljoin
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False
    logger.warning("URL fetcher dependencies not installed. Install with: pip install requests beautifulsoup4 markdownify")


class URLFetcher:
    """Fetch and process documents from URLs"""

    def __init__(
        self,
        download_dir: Optional[str] = None,
        timeout: int = 30,
        user_agent: str = "RAG-Anything/1.0"
    ):
        """
        Initialize URL fetcher

        Args:
            download_dir: Directory to save downloaded files
            timeout: Request timeout in seconds
            user_agent: User agent string for requests
        """
        if not DEPS_AVAILABLE:
            raise ImportError("Required dependencies not installed. Run: pip install requests beautifulsoup4 markdownify")

        self.download_dir = download_dir or tempfile.gettempdir()
        self.timeout = timeout
        self.headers = {"User-Agent": user_agent}

        Path(self.download_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"URLFetcher initialized (download_dir={self.download_dir})")

    def _create_content_list(self, title: str, text_content: str, images: List[Dict]) -> List[Dict[str, Any]]:
        """
        Create a structured content list compatible with RAG pipeline

        Args:
            title: Document title
            text_content: Extracted text content
            images: List of extracted images with metadata

        Returns:
            List of content blocks for RAG processing
        """
        content_list = []

        # Add title as first text block
        if title:
            content_list.append({
                "type": "text",
                "text": f"# {title}",
                "page_idx": 0
            })

        # Split text into paragraphs and add as text blocks
        paragraphs = [p.strip() for p in text_content.split("\n\n") if p.strip()]
        for idx, paragraph in enumerate(paragraphs[:50]):  # Limit to first 50 paragraphs
            if paragraph:
                content_list.append({
                    "type": "text",
                    "text": paragraph,
                    "page_idx": idx // 10  # Group every 10 paragraphs as a "page"
                })

        # Add images as image blocks
        for idx, img_info in enumerate(images):
            content_list.append({
                "type": "image",
                "img_path": img_info["path"],
                "image_caption": img_info.get("alt", "") or img_info.get("title", ""),
                "page_idx": (len(paragraphs) + idx) // 10
            })

        return content_list

    async def fetch_url(
        self,
        url: str,
        save_as_pdf: bool = False,
        convert_to_markdown: bool = True
    ) -> Dict[str, Any]:
        """
        Fetch and process content from URL

        Args:
            url: URL to fetch
            save_as_pdf: Whether to save as PDF (for PDF URLs)
            convert_to_markdown: Convert HTML to markdown

        Returns:
            Dictionary with file_path, content, metadata
        """
        try:
            logger.info(f"Fetching URL: {url}")

            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                raise ValueError(f"Invalid URL: {url}")

            # Determine content type
            response = await asyncio.to_thread(
                requests.head, url, headers=self.headers, timeout=self.timeout, allow_redirects=True
            )
            content_type = response.headers.get("Content-Type", "").lower()

            # Handle PDF files
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                return await self._fetch_pdf(url)

            # Handle HTML/web pages
            elif "html" in content_type or not content_type:
                return await self._fetch_html(url, convert_to_markdown)

            # Handle other file types
            else:
                return await self._fetch_generic(url, content_type)

        except Exception as e:
            logger.error(f"Error fetching URL {url}: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "url": url,
            }

    async def _fetch_pdf(self, url: str) -> Dict[str, Any]:
        """Fetch PDF from URL"""
        try:
            response = await asyncio.to_thread(
                requests.get, url, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()

            # Generate filename from URL
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"url_{url_hash}.pdf"
            file_path = Path(self.download_dir) / filename

            # Save PDF
            with open(file_path, "wb") as f:
                f.write(response.content)

            logger.info(f"PDF downloaded: {file_path}")

            return {
                "success": True,
                "file_path": str(file_path),
                "url": url,
                "content_type": "pdf",
                "size_bytes": len(response.content),
            }

        except Exception as e:
            logger.error(f"Error fetching PDF: {e}")
            raise

    async def _fetch_html(self, url: str, convert_to_markdown: bool = True) -> Dict[str, Any]:
        """Fetch and parse HTML page with advanced content extraction"""
        try:
            response = await asyncio.to_thread(
                requests.get, url, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.content, "html.parser")

            # Remove unwanted elements
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "iframe", "noscript"]):
                tag.decompose()

            # Extract title
            title = soup.find("title")
            title_text = title.get_text().strip() if title else "Untitled"

            # Extract main content
            main_content = soup.find("main") or soup.find("article") or soup.find("body")

            # Extract images before converting to markdown (limit to first 10 images)
            images = []
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            images_dir = Path(self.download_dir) / f"url_{url_hash}_images"
            images_dir.mkdir(parents=True, exist_ok=True)

            all_images = main_content.find_all("img")
            max_images = min(10, len(all_images))  # Limit to 10 images
            logger.info(f"Found {len(all_images)} images, downloading first {max_images}")

            for idx, img in enumerate(all_images[:max_images]):
                try:
                    img_url = img.get("src")
                    if not img_url:
                        continue

                    # Skip data URIs and very small images
                    if img_url.startswith("data:"):
                        continue

                    # Handle relative URLs
                    if img_url.startswith("//"):
                        img_url = "https:" + img_url
                    elif img_url.startswith("/"):
                        parsed_base = urlparse(url)
                        img_url = f"{parsed_base.scheme}://{parsed_base.netloc}{img_url}"
                    elif not img_url.startswith("http"):
                        img_url = urljoin(url, img_url)

                    # Download image with timeout
                    img_response = await asyncio.to_thread(
                        requests.get, img_url, headers=self.headers, timeout=5, stream=True
                    )

                    if img_response.status_code == 200:
                        # Check content size (skip if too large > 10MB)
                        content_length = img_response.headers.get('content-length')
                        if content_length and int(content_length) > 10 * 1024 * 1024:
                            logger.debug(f"Skipping large image {idx}: {content_length} bytes")
                            continue

                        # Determine file extension
                        content_type = img_response.headers.get("Content-Type", "")
                        ext = ".jpg"
                        if "png" in content_type:
                            ext = ".png"
                        elif "gif" in content_type:
                            ext = ".gif"
                        elif "webp" in content_type:
                            ext = ".webp"

                        img_path = images_dir / f"image_{idx}{ext}"
                        with open(img_path, "wb") as f:
                            f.write(img_response.content)

                        images.append({
                            "path": str(img_path),
                            "alt": img.get("alt", ""),
                            "title": img.get("title", ""),
                            "url": img_url
                        })
                        logger.debug(f"Downloaded image {idx+1}/{max_images}: {img_path.name}")
                except Exception as img_error:
                    logger.debug(f"Failed to download image {idx}: {img_error}")
                    continue

            if convert_to_markdown:
                # Convert to markdown
                content = markdownify.markdownify(
                    str(main_content),
                    heading_style="ATX",
                    bullets="-"
                )
            else:
                # Extract plain text
                content = main_content.get_text(separator="\n", strip=True)

            # Create content list with structured data
            content_list = self._create_content_list(title_text, content, images)

            # Save to file
            ext = ".md" if convert_to_markdown else ".txt"
            filename = f"url_{url_hash}{ext}"
            file_path = Path(self.download_dir) / filename

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(f"# {title_text}\n\n")
                f.write(f"Source: {url}\n\n")
                f.write(content)

            # Save content list as JSON for RAG processing
            import json
            json_path = Path(self.download_dir) / f"url_{url_hash}_content_list.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(content_list, f, indent=2, ensure_ascii=False)

            logger.info(f"HTML content saved: {file_path}")
            logger.info(f"Extracted {len(images)} images from web page")

            return {
                "success": True,
                "file_path": str(file_path),
                "content_list_path": str(json_path),
                "url": url,
                "content_type": "html",
                "title": title_text,
                "content_preview": content[:500],
                "images_count": len(images),
                "content_list": content_list
            }

        except Exception as e:
            logger.error(f"Error fetching HTML: {e}")
            raise

    async def _fetch_generic(self, url: str, content_type: str) -> Dict[str, Any]:
        """Fetch generic file"""
        try:
            response = await asyncio.to_thread(
                requests.get, url, headers=self.headers, timeout=self.timeout
            )
            response.raise_for_status()

            # Determine extension from content type
            ext_map = {
                "text/plain": ".txt",
                "text/markdown": ".md",
                "application/msword": ".doc",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            }
            ext = ext_map.get(content_type, ".bin")

            # Save file
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            filename = f"url_{url_hash}{ext}"
            file_path = Path(self.download_dir) / filename

            with open(file_path, "wb") as f:
                f.write(response.content)

            logger.info(f"File downloaded: {file_path}")

            return {
                "success": True,
                "file_path": str(file_path),
                "url": url,
                "content_type": content_type,
                "size_bytes": len(response.content),
            }

        except Exception as e:
            logger.error(f"Error fetching file: {e}")
            raise


def create_url_fetcher(download_dir: Optional[str] = None, **kwargs) -> URLFetcher:
    """
    Factory function to create a URL fetcher

    Args:
        download_dir: Directory to save downloaded files
        **kwargs: Additional URLFetcher parameters

    Returns:
        Configured URLFetcher instance
    """
    return URLFetcher(download_dir=download_dir, **kwargs)
