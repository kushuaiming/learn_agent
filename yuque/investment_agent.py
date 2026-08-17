"""
宏观投资日报 Agent

功能:
1. 读取 .env 中的 Access ID (YU_QUE_ID / TU_SHARE_ID / LLM_*).
2. 通过 Tushare 拉取中国宏观数据 (CPI/PPI/货币供应/GDP/PMI/SHIBOR).
3. 使用 LLM 对宏观数据进行分析.
4. 在语雀知识库 "200 - 社会科学" 的 "投资报告" 目录下新建一个文档,
   文件名形如 "2026/08/16 日报", 并把宏观数据分析写入该文档.
"""

import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
import tushare as ts
from dotenv import load_dotenv
from openai import OpenAI

# --- 0. 加载 .env (脚本位于 yuque/ 目录, .env 位于上一级目录) ---
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(os.path.dirname(_BASE_DIR), ".env")
load_dotenv(_ENV_PATH)

# --- 1. 读取 Access ID ---
YU_QUE_ID = os.getenv("YU_QUE_ID", "").strip()
TU_SHARE_ID = os.getenv("TU_SHARE_ID", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com").strip()
LLM_MODEL_ID = os.getenv("LLM_MODEL_ID", "deepseek-v4-flash").strip()

YUQUE_API_BASE = "https://www.yuque.com/api/v2"
BOOK_NAME = "200 - 社会科学"  # 目标知识库
DIR_TITLE = "投资报告"  # 目标目录 (知识库目录树中的分组节点)


# =====================================================================
# 语雀客户端
# =====================================================================
class YuqueClient:
    """封装语雀开放 API v2 的最小客户端."""

    def __init__(self, token: str):
        if not token:
            raise ValueError("缺少 YU_QUE_ID, 请检查 .env 文件.")
        self.headers = {
            "X-Auth-Token": token,
            "User-Agent": "investment-daily-agent",
        }

    def _request(self, method: str, path: str, **kwargs):
        url = f"{YUQUE_API_BASE}{path}"
        resp = requests.request(method, url, headers=self.headers, timeout=30, **kwargs)
        if resp.status_code >= 400:
            raise RuntimeError(
                f"语雀 API 请求失败 {resp.status_code}: {resp.text[:300]}"
            )
        payload = resp.json()
        if not isinstance(payload, dict) or "data" not in payload:
            raise RuntimeError(f"语雀 API 返回异常: {payload}")
        return payload["data"]

    def get_user(self) -> dict:
        return self._request("GET", "/user")

    def list_repos(self, login: str) -> List[dict]:
        return self._request("GET", f"/users/{login}/repos", params={"type": "Book"})

    def find_repo(self, name: str) -> dict:
        user = self.get_user()
        for repo in self.list_repos(user["login"]):
            if repo.get("name") == name:
                return repo
        raise RuntimeError(f"未找到知识库: {name}")

    def get_toc(self, namespace: str) -> List[dict]:
        return self._request("GET", f"/repos/{namespace}/toc")

    def create_doc(self, namespace: str, title: str, body: str) -> dict:
        return self._request(
            "POST",
            f"/repos/{namespace}/docs",
            json={
                "title": title,
                "body": body,
                "format": "markdown",
                "public": 0,
            },
        )

    def add_doc_to_toc(self, namespace: str, doc_id: int, parent_uuid: str):
        """把已创建的文档挂到目录树的指定节点 (parent_uuid) 下."""
        return self._request(
            "PUT",
            f"/repos/{namespace}/toc",
            json={
                "action": "appendNode",
                "action_mode": "child",
                "target_uuid": parent_uuid,
                "type": "DOC",
                "doc_ids": [doc_id],
            },
        )


# =====================================================================
# Tushare 宏观数据
# =====================================================================
def _prev_month(dt: datetime) -> str:
    """上一个完整月份, 形如 202607."""
    first = dt.replace(day=1)
    return (first - timedelta(days=1)).strftime("%Y%m")


def _month_range(end_month: str, months_back: int) -> str:
    """返回 start_month, 往前推 months_back 个月 (含 end_month 共 months_back+1 个月)."""
    y, m = int(end_month[:4]), int(end_month[4:6])
    m -= months_back
    while m <= 0:
        y -= 1
        m += 12
    return f"{y:04d}{m:02d}"


def _prev_quarter(dt: datetime) -> str:
    """最近一个已结束的季度, 形如 2026Q2."""
    q = (dt.month - 1) // 3
    if q == 0:
        return f"{dt.year - 1}Q4"
    return f"{dt.year}Q{q}"


def _quarter_range(end_quarter: str, quarters_back: int) -> str:
    """返回 start_quarter, 往前推 quarters_back 个季度."""
    year, q = int(end_quarter[:4]), int(end_quarter[-1])
    q -= quarters_back
    while q <= 0:
        year -= 1
        q += 4
    return f"{year}Q{q}"


def fetch_macro_data(now: datetime) -> Dict[str, pd.DataFrame]:
    """拉取各类宏观数据, 某接口失败时不影响其它接口."""
    ts.set_token(TU_SHARE_ID)
    pro = ts.pro_api()

    end_m = _prev_month(now)  # 最近一个完整月份
    start_m = _month_range(end_m, 11)  # 近 12 个月
    end_q = _prev_quarter(now)  # 最近一个完整季度
    start_q = _quarter_range(end_q, 7)  # 近 8 个季度
    start_d = (now - timedelta(days=180)).strftime("%Y%m%d")
    end_d = now.strftime("%Y%m%d")

    result: Dict[str, pd.DataFrame] = {}

    def _safe(key: str, fn, *args, **kwargs):
        try:
            df = fn(*args, **kwargs)
            if df is not None and not df.empty:
                result[key] = df
        except Exception as exc:  # noqa: BLE001
            print(f"Tushare fail to get {key} exception: {exc}")

    _safe("cpi", pro.cn_cpi, start_m=start_m, end_m=end_m)
    _safe("ppi", pro.cn_ppi, start_m=start_m, end_m=end_m)
    _safe("money", pro.cn_m, start_m=start_m, end_m=end_m)
    _safe("gdp", pro.cn_gdp, start_q=start_q, end_q=end_q)
    _safe("pmi", pro.cn_pmi, start_m=start_m, end_m=end_m)
    _safe("shibor", pro.shibor, start_date=start_d, end_date=end_d)

    if not result:
        raise RuntimeError(
            "未能从 Tushare 获取到任何宏观数据, 请检查 TU_SHARE_ID 与积分权限."
        )
    return result


# =====================================================================
# 数据格式化
# =====================================================================
def _fmt(v) -> str:
    """把单元格数值格式化为字符串."""
    try:
        if pd.isna(v):
            return "-"
    except (TypeError, ValueError):
        pass
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def df_to_markdown(df: pd.DataFrame, columns: List[str], max_rows: int = 8) -> str:
    """把 DataFrame 的指定列渲染为 Markdown 表格 (按时间倒序)."""
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return "(无数据)"
    sub = df[cols].head(max_rows)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = [header, sep]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in cols) + " |")
    return "\n".join(lines)


# 各数据集的展示列 (列名与含义对照)
_SECTION_COLUMNS = {
    "cpi": ["month", "nt_yoy", "nt_mom", "nt_accu"],
    "ppi": ["month", "ppi_yoy", "ppi_mom", "ppi_accu"],
    "money": ["month", "m0", "m0_yoy", "m1", "m1_yoy", "m2", "m2_yoy"],
    "gdp": ["quarter", "gdp_yoy", "pi_yoy", "si_yoy", "ti_yoy"],
    "pmi": ["MONTH", "PMI010000"],
    "shibor": ["date", "on", "1w", "1m", "3m", "6m", "1y"],
}

_SECTION_TITLES = {
    "cpi": "居民消费价格指数 (CPI)",
    "ppi": "工业生产者出厂价格指数 (PPI)",
    "money": "货币供应量 (M0/M1/M2)",
    "gdp": "国内生产总值 (GDP)",
    "pmi": "制造业采购经理指数 (PMI)",
    "shibor": "上海银行间同业拆放利率 (SHIBOR)",
}


def build_report_body(
    date_title: str, data: Dict[str, pd.DataFrame], analysis: str
) -> str:
    """组装最终写入语雀的 Markdown 报告."""
    parts: List[str] = []
    parts.append(f"# {date_title}")
    parts.append("")
    parts.append(
        f"> 数据来源: Tushare  |  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    parts.append("")

    # 数据速览: 每个数据集取最新一条的关键指标
    parts.append("## 一、核心数据速览")
    parts.append("")
    for key in _SECTION_COLUMNS:
        if key not in data:
            continue
        df = data[key]
        cols = _SECTION_COLUMNS[key]
        if df.empty:
            continue
        latest = df.iloc[0]
        overview = "、".join(
            f"{c}={_fmt(latest[c])}" for c in cols[1:] if c in df.columns
        )
        parts.append(
            f"- **{_SECTION_TITLES[key]}** (最新: {_fmt(latest[cols[0]])}): {overview}"
        )
    parts.append("")

    # 详细数据表
    parts.append("## 二、详细数据")
    parts.append("")
    for key in _SECTION_COLUMNS:
        if key not in data:
            continue
        parts.append(f"### {_SECTION_TITLES[key]}")
        parts.append("")
        parts.append(df_to_markdown(data[key], _SECTION_COLUMNS[key]))
        parts.append("")

    # 分析
    parts.append("## 三、宏观分析与展望")
    parts.append("")
    parts.append(analysis)
    parts.append("")
    parts.append("---")
    parts.append("*本报告由程序自动生成, 数据仅供参考, 不构成任何投资建议.*")
    return "\n".join(parts)


# =====================================================================
# LLM 分析
# =====================================================================
def build_digest(data: Dict[str, pd.DataFrame]) -> str:
    """把宏观数据压缩成一段纯文本摘要, 供 LLM 分析."""
    lines = []
    for key in _SECTION_COLUMNS:
        if key not in data:
            continue
        df = data[key]
        cols = _SECTION_COLUMNS[key]
        if df.empty:
            continue
        latest = df.iloc[0]
        lines.append(
            f"[{_SECTION_TITLES[key]}] 最新({_fmt(latest[cols[0]])}): "
            + "; ".join(f"{c}={_fmt(latest[c])}" for c in cols[1:] if c in df.columns)
        )
        # 附上最近 6 期原始值
        lines.append(
            "  近6期: " + df_to_markdown(df, cols, max_rows=6).replace("\n", " | ")
        )
    return "\n".join(lines)


def analyze_with_llm(data: Dict[str, pd.DataFrame]) -> Optional[str]:
    """调用 LLM 生成宏观分析, 失败时返回 None."""
    if not all([LLM_API_KEY, LLM_BASE_URL, LLM_MODEL_ID]):
        print("[LLM] 未配置 LLM_API_KEY/LLM_BASE_URL/LLM_MODEL_ID, 跳过 LLM 分析.")
        return None

    digest = build_digest(data)
    prompt = (
        "你是一名资深宏观分析师。以下是从 Tushare 获取的中国宏观数据摘要:\n\n"
        f"{digest}\n\n"
        "请基于以上数据撰写一份简明的「每日宏观投资日报」分析(300~500 字), 包含:\n"
        "1. 增长端 (GDP / PMI) 的边际变化;\n"
        "2. 通胀端 (CPI / PPI) 的变化趋势;\n"
        "3. 流动性 (货币供应 / SHIBOR) 的松紧情况;\n"
        "4. 对大类资产 (股票、债券、商品、黄金) 的简要启示.\n"
        "要求: 客观、数据驱动、不过度解读, 结尾附一句风险提示."
    )

    try:
        client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
        print(f"[LLM] 正在调用 {LLM_MODEL_ID} 生成分析...")
        resp = client.chat.completions.create(
            model=LLM_MODEL_ID,
            messages=[
                {"role": "system", "content": "你是一名专业、审慎的宏观分析师."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            stream=False,
        )
        text = resp.choices[0].message.content.strip()
        print("[LLM] 分析生成成功.")
        return text
    except Exception as exc:  # noqa: BLE001
        print(f"[LLM] 调用失败, 将使用程序化摘要替代: {exc}")
        return None


def fallback_analysis(data: Dict[str, pd.DataFrame]) -> str:
    """LLM 不可用时的程序化摘要分析."""
    lines = ["(以下为程序化摘要, LLM 分析暂不可用)", ""]
    for key in _SECTION_COLUMNS:
        if key not in data:
            continue
        df = data[key]
        cols = _SECTION_COLUMNS[key]
        if df.empty:
            continue
        latest = df.iloc[0]
        lines.append(
            f"- {_SECTION_TITLES[key]}: "
            + "; ".join(f"{c}={_fmt(latest[c])}" for c in cols[1:] if c in df.columns)
        )
    return "\n".join(lines)


def main():
    now = datetime.now()
    date_title = f"{now.strftime('%Y/%m/%d')} 日报"
    print(f"date_title: {date_title}")

    # 1. 拉取宏观数据
    print("=" * 50)
    print("[1/4] 通过 Tushare 拉取宏观数据...")
    data = fetch_macro_data(now)
    print(f"      成功获取 {len(data)} 类数据: {list(data.keys())}")

    # 2. LLM 分析
    print("=" * 50)
    print("[2/4] 生成宏观分析...")
    analysis = analyze_with_llm(data) or fallback_analysis(data)

    # 3. 组装报告
    print("=" * 50)
    print("[3/4] 组装 Markdown 报告...")
    body = build_report_body(date_title, data, analysis)

    # 4. 写入语雀
    print("=" * 50)
    print("[4/4] 写入语雀知识库...")
    yuque = YuqueClient(YU_QUE_ID)
    repo = yuque.find_repo(BOOK_NAME)
    namespace = repo["namespace"]
    print(f"      knowledge Base: {repo['name']} ({namespace})")

    # 目录树 (同时用于幂等检查: 若同名文档已存在则不再重复创建)
    toc = yuque.get_toc(namespace) # toc means table of contents
    dir_node = next((n for n in toc if n.get("title") == DIR_TITLE), None)
    if dir_node is None:
        raise RuntimeError(f"知识库中未找到目录: {DIR_TITLE}")
    parent_uuid = dir_node.get("uuid")
    print(f"      目标目录: {DIR_TITLE} (uuid={parent_uuid})")

    existing = next((n for n in toc if n.get("title") == date_title), None)
    if existing:
        print(f"      文档已存在, 跳过创建: {date_title} (id={existing.get('id')})")
        return

    doc = yuque.create_doc(namespace, title=date_title, body=body)
    doc_id = doc["id"]
    print(f"      文档已创建: id={doc_id}, slug={doc.get('slug')}")

    yuque.add_doc_to_toc(namespace, doc_id=doc_id, parent_uuid=parent_uuid)
    print(f"      文档已挂载到目录「{DIR_TITLE}」下.")

    print("=" * 50)
    print(f"完成! 已生成《{date_title}》并写入语雀.")


if __name__ == "__main__":
    main()
