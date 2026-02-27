import argparse
import html
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import feedparser
import requests
import yaml
from dateutil import parser as date_parser
from openai import OpenAI


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

UTC = timezone.utc
CHINA_TZ = timezone(timedelta(hours=8))
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Focus tracks:
# 1) AI coding and agents
# 2) New models/tech (including open source)
# 3) Product iteration of model companies
# 4) Model architecture and training efficiency
INCLUDE_KEYWORDS = [
    "ai 编程", "ai coding", "code agent", "coding agent", "agent", "智能体", "多智能体", "copilot",
    "cursor", "devin", "claude code", "代码生成", "代码助手", "编程助手",
    "大模型", "foundation model", "llm", "vlm", "multimodal", "多模态",
    "模型发布", "模型开源", "open source", "开源模型", "推理优化", "蒸馏", "微调", "对齐",
    "训练效率", "训练成本", "模型架构", "transformer", "moe", "long context", "rag",
    "openai", "anthropic", "google deepmind", "gemini", "claude", "kimi", "deepseek", "qwen", "通义", "豆包",
    "产品更新", "版本更新", "api 更新", "功能发布", "发布会",
]

EXCLUDE_KEYWORDS = [
    "ai 芯片", "芯片", "gpu", "npu", "算力卡", "半导体",
    "ai 金融", "金融", "证券", "基金", "股市", "银行", "保险", "理财", "投顾", "ipo",
]


@dataclass
class NewsItem:
    source_name: str
    title: str
    link: str
    published_utc: datetime
    summary: str


def load_sources(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    sources = data.get("sources", [])
    if not sources:
        raise ValueError("sources.yaml 中没有配置任何 RSS 源")

    return sources


def parse_entry_datetime(entry: Any) -> datetime | None:
    candidates = [
        entry.get("published"),
        entry.get("updated"),
        entry.get("created"),
    ]
    for val in candidates:
        if not val:
            continue
        try:
            dt = date_parser.parse(val)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except Exception:
            continue

    return None


def get_yesterday_range_utc(now_utc: datetime) -> tuple[datetime, datetime, str]:
    now_cn = now_utc.astimezone(CHINA_TZ)
    yesterday_cn_date = now_cn.date() - timedelta(days=1)
    start_cn = datetime.combine(yesterday_cn_date, datetime.min.time(), tzinfo=CHINA_TZ)
    end_cn = start_cn + timedelta(days=1)
    return start_cn.astimezone(UTC), end_cn.astimezone(UTC), yesterday_cn_date.isoformat()


def _extract_next_data_json(html_text: str) -> dict[str, Any] | None:
    match = re.search(
        r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
        html_text,
        re.S,
    )
    if not match:
        return None

    try:
        return json.loads(match.group(1))
    except Exception:
        return None


def fetch_chinastarmarket_for_yesterday(
    source_name: str, source_url: str, start_utc: datetime, end_utc: datetime
) -> list[NewsItem]:
    target_cn_date = start_utc.astimezone(CHINA_TZ).date()
    month_str = target_cn_date.strftime("%Y%m")
    sitemap_url = f"https://rss.chinastarmarket.cn/kcb/baidu/{month_str}/sitemap.xml"

    subject_id: str | None = None
    subject_match = re.search(r"/subject/(\d+)", source_url)
    if subject_match:
        subject_id = subject_match.group(1)

    try:
        resp = requests.get(sitemap_url, headers=DEFAULT_HEADERS, timeout=20)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
    except Exception as exc:
        logging.warning("中国星市场 sitemap 获取失败: %s - %s", source_name, exc)
        return []

    candidate_links: list[str] = []
    for url_node in root.findall("./url"):
        loc = (url_node.findtext("loc") or "").strip()
        lastmod = (url_node.findtext("lastmod") or "").strip()
        if not loc or "/detail/" not in loc:
            continue
        if lastmod[:10] != target_cn_date.isoformat():
            continue
        candidate_links.append(loc)

    # Cap detail-page requests to control runtime in CI.
    max_detail_fetch = 120
    if len(candidate_links) > max_detail_fetch:
        candidate_links = candidate_links[:max_detail_fetch]

    result: list[NewsItem] = []
    for link in candidate_links:
        try:
            detail_resp = requests.get(link, headers=DEFAULT_HEADERS, timeout=20)
            detail_resp.raise_for_status()
            next_data = _extract_next_data_json(detail_resp.text)
            if not next_data:
                continue

            article = (
                next_data.get("props", {})
                .get("pageProps", {})
                .get("data", {})
            )
            if not isinstance(article, dict):
                continue

            subjects = article.get("subject") or []
            if subject_id:
                if not any(
                    isinstance(s, dict) and str(s.get("id", "")) == subject_id
                    for s in subjects
                ):
                    continue

            ctime = article.get("ctime")
            if not isinstance(ctime, (int, float)):
                continue

            dt = datetime.fromtimestamp(ctime, tz=CHINA_TZ).astimezone(UTC)
            if not (start_utc <= dt < end_utc):
                continue

            title = str(article.get("title") or "").strip()
            summary = str(article.get("brief") or article.get("content") or "").strip()
            if not title:
                continue

            result.append(
                NewsItem(
                    source_name=source_name,
                    title=title,
                    link=link,
                    published_utc=dt,
                    summary=summary,
                )
            )
        except Exception:
            continue

    logging.info("ChinaStarMarket HTML collected: %s -> %d items", source_name, len(result))
    return result


def fetch_news_for_yesterday(sources: list[dict[str, str]], start_utc: datetime, end_utc: datetime) -> list[NewsItem]:
    items: list[NewsItem] = []

    for source in sources:
        name = source.get("name", "Unknown")
        url = source.get("url", "").strip()
        if not url:
            continue

        parsed = urlparse(url)
        if "chinastarmarket.cn" in parsed.netloc.lower() and "/subject/" in parsed.path:
            logging.info("Fetching ChinaStarMarket HTML: %s (%s)", name, url)
            items.extend(fetch_chinastarmarket_for_yesterday(name, url, start_utc, end_utc))
            continue

        logging.info("Fetching RSS: %s (%s)", name, url)
        feed = feedparser.parse(url)

        if getattr(feed, "bozo", 0):
            logging.warning("RSS 解析异常: %s - %s", name, getattr(feed, "bozo_exception", "unknown"))

        for entry in feed.entries:
            dt = parse_entry_datetime(entry)
            if not dt:
                continue
            if not (start_utc <= dt < end_utc):
                continue

            title = (entry.get("title") or "").strip()
            link = (entry.get("link") or "").strip()
            summary = (entry.get("summary") or entry.get("description") or "").strip()

            if not title or not link:
                continue

            items.append(
                NewsItem(
                    source_name=name,
                    title=title,
                    link=link,
                    published_utc=dt,
                    summary=summary,
                )
            )

    dedup: dict[str, NewsItem] = {}
    for item in items:
        key = item.link.lower() if item.link else item.title.lower()
        if key not in dedup:
            dedup[key] = item

    result = list(dedup.values())
    result.sort(key=lambda x: x.published_utc)
    return result


def clean_text(raw: str) -> str:
    if not raw:
        return ""
    text = html.unescape(raw)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    text = text.replace("□", "")
    return text


def is_relevant_ai_news(item: NewsItem) -> bool:
    hay = f"{item.title} {item.summary}".lower()

    if any(k.lower() in hay for k in EXCLUDE_KEYWORDS):
        return False

    return any(k.lower() in hay for k in INCLUDE_KEYWORDS)


def normalize_and_filter_items(news_items: list[NewsItem]) -> list[NewsItem]:
    filtered: list[NewsItem] = []
    for item in news_items:
        normalized = NewsItem(
            source_name=item.source_name,
            title=clean_text(item.title),
            link=item.link,
            published_utc=item.published_utc,
            summary=clean_text(item.summary),
        )
        if is_relevant_ai_news(normalized):
            filtered.append(normalized)
    return filtered


def serialize_items(news_items: list[NewsItem]) -> list[dict[str, str]]:
    return [
        {
            "source": item.source_name,
            "title": item.title,
            "link": item.link,
            "published_utc": item.published_utc.isoformat(),
            "summary": item.summary,
        }
        for item in news_items
    ]


def write_collected_news(news_items: list[NewsItem], output_path: str, target_date_cn: str) -> None:
    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    payload = {
        "date_cn": target_date_cn,
        "count": len(news_items),
        "items": serialize_items(news_items),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def build_prompt(news_items: list[NewsItem], target_date_cn: str) -> str:
    payload = []
    for idx, item in enumerate(news_items, start=1):
        payload.append(
            {
                "id": idx,
                "source": item.source_name,
                "title": item.title,
                "link": item.link,
                "published_utc": item.published_utc.isoformat(),
                "raw_summary": item.summary,
            }
        )

    return (
        f"你是 AI 行业新闻编辑。请基于给定新闻，输出 {target_date_cn}（北京时间）的 AI 新闻中文简报。\\n"
        "要求：\\n"
        "0) 只保留以下赛道：\\n"
        "- 国内外 AI 编程与智能体（Agent）\\n"
        "- 国内外新模型/新技术（含开源）\\n"
        "- 国内外大模型公司产品迭代\\n"
        "- 模型架构与训练效率突破\\n"
        "明确排除：AI 芯片、AI 金融。\\n"
        "1) 仅根据输入内容，不要编造。\\n"
        "2) 选择最重要的 5-10 条（不足则按实际数量）。\\n"
        "3) 每条严格使用以下格式：\\n"
        "标题：...\\n"
        "3句摘要：\\n"
        "- ...\\n"
        "- ...\\n"
        "- ...\\n"
        "来源链接：...\\n"
        "影响点评：...\\n"
        "4) 最后补充一段“今日总体观察”（2-3 句）。\\n"
        "5) 输出语言：简体中文。\\n\\n"
        "新闻输入(JSON)：\\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
    )


def summarize_with_openai(client: OpenAI, model_name: str, news_items: list[NewsItem], target_date_cn: str) -> str:
    if not news_items:
        return f"{target_date_cn} 未抓取到可用的 AI 新闻。"

    prompt = build_prompt(news_items, target_date_cn)

    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": "你是严谨的科技媒体编辑，擅长将多来源新闻整合成简明中文早报。",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    return response.choices[0].message.content.strip()


def push_to_feishu(webhook: str, content: str, target_date_cn: str) -> None:
    body = {
        "msg_type": "text",
        "content": {
            "text": f"AI 新闻日报（{target_date_cn}）\\n\\n{content}"
        },
    }

    resp = requests.post(webhook, json=body, timeout=20)
    if resp.status_code != 200:
        raise RuntimeError(f"飞书推送失败: HTTP {resp.status_code} - {resp.text}")

    data = resp.json()
    code = data.get("code")
    if code not in (0, "0", None):
        raise RuntimeError(f"飞书推送失败: {data}")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI News Agent")
    parser.add_argument(
        "--mode",
        choices=["collect", "full"],
        default="collect",
        help="collect: 仅抓取新闻并写入本地；full: 抓取+OpenAI总结+飞书发送",
    )
    parser.add_argument(
        "--output",
        default="ai-news-agent/output/collected_news.json",
        help="collect 模式下本地输出文件路径",
    )
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sources_path = os.path.join(base_dir, "sources.yaml")
    sources = load_sources(sources_path)

    now_utc = datetime.now(tz=UTC)
    start_utc, end_utc, target_date_cn = get_yesterday_range_utc(now_utc)

    logging.info("Collecting news in UTC range: [%s, %s)", start_utc.isoformat(), end_utc.isoformat())
    items = fetch_news_for_yesterday(sources, start_utc, end_utc)
    logging.info("Collected %d deduplicated items", len(items))

    items = normalize_and_filter_items(items)
    logging.info("After focus filter: %d items", len(items))

    if args.mode == "collect":
        output_path = args.output
        if not os.path.isabs(output_path):
            output_path = os.path.join(os.path.dirname(base_dir), output_path)
        write_collected_news(items, output_path, target_date_cn)
        logging.info("Collect mode done. Output written to: %s", output_path)
        return

    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    openai_base_url = os.getenv("OPENAI_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "gpt-4.1-mini").strip()
    feishu_webhook = os.getenv("FEISHU_WEBHOOK", "").strip()

    if not openai_api_key:
        raise EnvironmentError("full 模式缺少环境变量 OPENAI_API_KEY")
    if not feishu_webhook:
        raise EnvironmentError("full 模式缺少环境变量 FEISHU_WEBHOOK")

    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url or None)
    summary = summarize_with_openai(client, model_name, items, target_date_cn)
    push_to_feishu(feishu_webhook, summary, target_date_cn)

    logging.info("Full mode done. Feishu message sent successfully.")


if __name__ == "__main__":
    main()

