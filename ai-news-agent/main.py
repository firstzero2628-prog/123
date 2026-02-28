import argparse
import html
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlparse, parse_qs
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

# User preference filter: keep only items that match these focus keywords.
# Set to empty list to disable this stricter filter.
PREFERRED_KEYWORDS = [
    "智能体", "agent", "编程平台", "编程", "coding",
    "deepseek", "v4", "推理性能", "视频模型", "多模态",
    "榜", "排行", "第一梯队", "DAU", "日活", "千问",
    "nano banana", "图像生成", "gemini", "手机助手",
    "codex", "figma", "健康", "医疗", "健康助手",
    "openclaw", "围城", "生态",
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


def _collect_chinastarmarket_detail_links(
    month_str: str, target_cn_date_str: str
) -> list[str]:
    sitemap_candidates: list[str] = []
    index_url = "https://www.chinastarmarket.cn/sitemap.xml"
    fallback_url = f"https://rss.chinastarmarket.cn/kcb/baidu/{month_str}/sitemap.xml"

    try:
        idx_resp = requests.get(index_url, headers=DEFAULT_HEADERS, timeout=20)
        idx_resp.raise_for_status()
        idx_root = ET.fromstring(idx_resp.text)
        for node in idx_root.findall("./sitemap"):
            loc = (node.findtext("loc") or "").strip()
            if f"/{month_str}/sitemap.xml" in loc:
                sitemap_candidates.append(loc)
                break
    except Exception as exc:
        logging.warning("中国星市场 sitemap 索引获取失败: %s", exc)

    if fallback_url not in sitemap_candidates:
        sitemap_candidates.append(fallback_url)

    candidate_links: list[str] = []
    for sitemap_url in sitemap_candidates:
        try:
            resp = requests.get(sitemap_url, headers=DEFAULT_HEADERS, timeout=20)
            resp.raise_for_status()
            root = ET.fromstring(resp.text)

            for url_node in root.findall("./url"):
                loc = (url_node.findtext("loc") or "").strip()
                lastmod = (url_node.findtext("lastmod") or "").strip()
                if not loc or "/detail/" not in loc:
                    continue
                if lastmod[:10] != target_cn_date_str:
                    continue
                candidate_links.append(loc)

            if candidate_links:
                return candidate_links
        except Exception as exc:
            logging.warning("中国星市场 sitemap 获取失败: %s - %s", sitemap_url, exc)

    return candidate_links


def fetch_chinastarmarket_for_yesterday(
    source_name: str, source_url: str, start_utc: datetime, end_utc: datetime
) -> list[NewsItem]:
    target_cn_date = start_utc.astimezone(CHINA_TZ).date()
    month_str = target_cn_date.strftime("%Y%m")

    subject_id: str | None = None
    subject_match = re.search(r"/subject/(\d+)", source_url)
    if subject_match:
        subject_id = subject_match.group(1)

    subject_names: list[str] = []
    parsed = urlparse(source_url)
    q = parse_qs(parsed.query)
    for key in ("subject_name", "tag", "tags"):
        vals = q.get(key, [])
        for v in vals:
            subject_names.extend([s.strip() for s in v.split(",") if s.strip()])
    subject_names = [s.lower() for s in subject_names]

    candidate_links = _collect_chinastarmarket_detail_links(
        month_str=month_str,
        target_cn_date_str=target_cn_date.isoformat(),
    )
    if not candidate_links:
        logging.warning("中国星市场无可用 sitemap 数据: %s", source_name)
        return []

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

            if subject_names:
                if not any(
                    isinstance(s, dict)
                    and str(s.get("name", "")).strip().lower() in subject_names
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

    if not any(k.lower() in hay for k in INCLUDE_KEYWORDS):
        return False

    if PREFERRED_KEYWORDS:
        return any(k.lower() in hay for k in PREFERRED_KEYWORDS)

    return True


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
        "用户定位：作为一个 AIGC 爱好者，关注 AI 编码、智能体、工作流、AI 新模型。\\n"
        "输出风格：先整体总结，再分点事件。\\n"
        "要求：\\n"
        "0) 只保留以下赛道：\\n"
        "- 国内外 AI 编程与智能体（Agent）\\n"
        "- 国内外新模型/新技术（含开源）\\n"
        "- 国内外大模型公司产品迭代\\n"
        "- 模型架构与训练效率突破\\n"
        "明确排除：AI 芯片、AI 金融。\\n"
        "1) 仅根据输入内容，不要编造。\\n"
        "2) 选择最重要的 5-10 条（不足则按实际数量）。\\n"
        "3) 输出必须为简体中文；如果新闻原文是英文，请先准确翻译再写摘要。\\n"
        "4) 国外新闻最多保留 3 条，其余优先中文来源新闻。\\n"
        "5) 标题固定为：全球AI产业要闻简报 | 24小时内\\n"
        "6) 在标题下一行单独写日期：日期：YYYY-MM-DD（北京时间，使用给定日期）。\\n"
        "7) 整体总结：2-4 句，概括本日最大趋势与方向，避免空话。\\n"
        "8) 分点事件：用 1..N 编号。每条格式如下：\\n"
        "1. 简短标题（不超过20字，聚焦动作/突破）\\n"
        "• 要点1（核心事实）\\n"
        "• 要点2（关键影响/数据）\\n"
        "• 要点3（可选：与竞品/生态的关系）\\n"
        "信源：来源名（链接）\\n"
        "9) 若出现英文术语，首次出现需括号补一个简短中文解释。\\n"
        "10) 末尾补充“今日总体观察”：2-3 句。\\n"
        "11) 保持整体可读性，少用口号式表述。\\n\\n"
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


def summarize_with_baidu_search(
    api_key: str,
    target_date_cn: str,
    client: OpenAI | None = None,
    model_name: str | None = None,
) -> str | None:
    url = "https://qianfan.baidubce.com/v2/ai_search/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    prompt = (
        f"你是 AI 行业新闻编辑。请基于全网搜索结果，输出 {target_date_cn}（北京时间）"
        "AI 新闻热点简报。"
        "用户定位：AIGC 爱好者，关注 AI 编码、智能体、工作流、AI 新模型。"
        "输出格式：先整体总结（2-4句），再分点事件（1..N）。"
        "每条需附上原始链接。"
    )
    body = {
        "messages": [
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        # Try multiple common response shapes.
        if isinstance(data, dict):
            if "result" in data:
                if isinstance(data["result"], str):
                    return data["result"]
                if isinstance(data["result"], dict):
                    text = data["result"].get("text") or data["result"].get("content")
                    if text:
                        return text
            if "response" in data and isinstance(data["response"], str):
                return data["response"]
            if "output" in data:
                if isinstance(data["output"], str):
                    return data["output"]
                if isinstance(data["output"], dict):
                    text = data["output"].get("text") or data["output"].get("content")
                    if text:
                        return text
            if "choices" in data and data["choices"]:
                msg = (data.get("choices", [{}])[0].get("message", {}) or {})
                content = msg.get("content")
                if content:
                    return content
        references = []
        if isinstance(data, dict):
            references = data.get("references") or data.get("data", {}).get("references") or []
        if references:
            logging.info("百度智能搜索生成仅返回 references，count=%d", len(references))
            if client and model_name:
                ref_lines = []
                for ref in references[:10]:
                    title = (ref.get("title") or "").strip()
                    url = (ref.get("url") or "").strip()
                    snippet = (ref.get("snippet") or ref.get("summary") or "").strip()
                    if title and url:
                        ref_lines.append(f"- {title}\\n  链接：{url}\\n  摘要：{snippet}")
                prompt2 = (
                    f"请基于以下检索参考，生成 {target_date_cn}（北京时间）"
                    "AI 新闻热点简报。"
                    "用户定位：AIGC 爱好者，关注 AI 编码、智能体、工作流、AI 新模型。"
                    "输出格式：先整体总结（2-4句），再分点事件（1..N）。"
                    "每条需附上原始链接。\\n\\n"
                    "参考：\\n" + "\\n".join(ref_lines)
                )
                try:
                    response = client.chat.completions.create(
                        model=model_name,
                        temperature=0.2,
                        messages=[
                            {"role": "system", "content": "你是严谨的科技媒体编辑。"},
                            {"role": "user", "content": prompt2},
                        ],
                    )
                    return response.choices[0].message.content.strip()
                except Exception as exc:
                    logging.warning("基于 references 生成简报失败: %s", exc)
            # Fallback: list references only.
            fallback = ["未返回生成文本，以下为检索链接："]
            for ref in references[:10]:
                title = (ref.get("title") or "").strip()
                url = (ref.get("url") or "").strip()
                if title and url:
                    fallback.append(f"- {title} {url}")
            return "\\n".join(fallback)

        logging.warning(
            "百度智能搜索生成返回结构无法解析，keys=%s",
            list(data.keys()) if isinstance(data, dict) else type(data),
        )
        return None
    except Exception as exc:
        logging.warning("百度智能搜索生成调用失败: %s", exc)
        return None


def compose_final_digest(
    client: OpenAI,
    model_name: str,
    target_date_cn: str,
    source_digest: str,
    web_digest: str | None,
) -> str:
    prompt = (
        f"你是 AI 行业新闻编辑。请将下方两段简报合并重排，输出 {target_date_cn}（北京时间）的一份最终简报。\\n"
        "目标：保留最重要的 10 条（不足则按实际数量）。\\n"
        "优先级从高到低：AI 编程、AI 智能体、AI 工作流、AI 新模型、AIGC。\\n"
        "去重、合并相同事件，避免重复。\\n"
        "输出格式要清晰易读：\\n"
        "标题：全球AI产业要闻简报 | 24小时内\\n"
        "日期：YYYY-MM-DD（北京时间，使用给定日期）\\n"
        "整体总结：2-4 句\\n"
        "分点事件（1..N）：\\n"
        "1. 简短标题（<=20字）\\n"
        "• 要点1\\n"
        "• 要点2\\n"
        "信源：来源名（链接）\\n"
        "末尾：今日总体观察 2-3 句\\n\\n"
        "简报A（抓取源）：\\n"
        f"{source_digest}\\n\\n"
        "简报B（全网检索）：\\n"
        f"{web_digest or '无'}"
    )
    response = client.chat.completions.create(
        model=model_name,
        temperature=0.2,
        messages=[
            {"role": "system", "content": "你是严谨的科技媒体编辑。"},
            {"role": "user", "content": prompt},
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
    baidu_api_key = os.getenv("BAIDU_API_KEY", "").strip()

    if not openai_api_key:
        raise EnvironmentError("full 模式缺少环境变量 OPENAI_API_KEY")
    if not feishu_webhook:
        raise EnvironmentError("full 模式缺少环境变量 FEISHU_WEBHOOK")

    client = OpenAI(api_key=openai_api_key, base_url=openai_base_url or None)
    summary = summarize_with_openai(client, model_name, items, target_date_cn)
    if baidu_api_key:
        baidu_summary = summarize_with_baidu_search(
            baidu_api_key, target_date_cn, client=client, model_name=model_name
        )
        if baidu_summary:
            summary = compose_final_digest(
                client,
                model_name,
                target_date_cn,
                summary,
                baidu_summary,
            )
    push_to_feishu(feishu_webhook, summary, target_date_cn)

    logging.info("Full mode done. Feishu message sent successfully.")


if __name__ == "__main__":
    main()

