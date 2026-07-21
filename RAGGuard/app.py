"""RAGGuard Streamlit Dashboard — frontend for RAGGuard REST API."""

import json
import os
import sys
import time

# ── Chinese font setup (must run before any pyplot import) ──────
import matplotlib
import matplotlib.font_manager as fm

_CHINESE_FONTS = [
    "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Noto Sans SC", "Noto Sans CJK",
    "WenQuanYi Micro Hei", "WenQuanYi Zen Hei",
]
_font_family = None
for _name in _CHINESE_FONTS:
    for _f in fm.fontManager.ttflist:
        if _name.lower() in _f.name.lower():
            _font_family = _f.name
            break
    if _font_family:
        break
if _font_family:
    matplotlib.rcParams["font.family"] = _font_family
    matplotlib.rcParams["font.sans-serif"] = [_font_family]
matplotlib.rcParams["axes.unicode_minus"] = False

import requests
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

st.set_page_config(
    page_title="RAGGuard - 幻觉检测",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ RAGGuard - 客服回复幻觉检测 Agent")
st.caption("Multi-Agent Hallucination Detection Framework for Customer Service LLMs")

# ── Session state defaults ───────────────────────────────────

for key, default in [
    ("backend_url", "http://localhost:8000"),
    ("results", None),
    ("metrics", None),
    ("run_id", None),
    ("single_result", None),
    ("single_case", None),
    ("runs", []),
    ("loaded_run_id", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


def api_url(path: str) -> str:
    return f"{st.session_state.backend_url}{path}"


def check_backend() -> bool:
    try:
        r = requests.get(api_url("/health"), timeout=3)
        return r.status_code == 200
    except Exception:
        return False


# ── Sidebar ──────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ 配置")

    backend_url = st.text_input("Backend URL", value=st.session_state.backend_url)
    if backend_url != st.session_state.backend_url:
        st.session_state.backend_url = backend_url

    backend_ok = check_backend()
    if backend_ok:
        st.success("Backend 已连接")
    else:
        st.error("Backend 未连接 — 请先启动 server.py")

    st.divider()

    mode = st.selectbox("检测模式", ["mock", "llm"], index=0)

    if mode == "llm":
        env_key = os.getenv("RAGGUARD_API_KEY") or os.getenv("OPENAI_API_KEY") or ""
        env_url = os.getenv("RAGGUARD_BASE_URL") or os.getenv("OPENAI_BASE_URL") or "https://api.deepseek.com"
        env_model = os.getenv("RAGGUARD_MODEL") or "deepseek-v4-pro"

        api_key = st.text_input("API Key", value=env_key, type="password")
        base_url = st.text_input("Base URL", value=env_url)
        model = st.text_input("Model", value=env_model)

    st.divider()

    # Load case list
    try:
        with open("data/replies.json", "r", encoding="utf-8") as f:
            all_cases = json.load(f)
        case_ids = [c["id"] for c in all_cases]
    except FileNotFoundError:
        all_cases = []
        case_ids = []

    st.header("📋 批量评估")
    if st.button("🔄 运行检测", use_container_width=True, type="primary"):
        if not backend_ok:
            st.error("请先启动 Backend: python server.py")
        else:
            payload = {"mode": mode, "concurrency": 8}
            if mode == "llm":
                payload["api_key"] = api_key
                payload["base_url"] = base_url
                payload["model"] = model
            try:
                # Start async evaluation
                resp = requests.post(
                    api_url("/evaluate/start"),
                    json=payload,
                    timeout=30,
                )
                if resp.status_code != 200:
                    st.error(f"启动失败: {resp.text}")
                else:
                    task_id = resp.json()["task_id"]
                    total = resp.json()["total"]

                    # Poll progress with a progress bar
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    while True:
                        time.sleep(0.5)
                        pr = requests.get(
                            api_url(f"/evaluate/{task_id}"),
                            timeout=10,
                        )
                        if pr.status_code != 200:
                            continue
                        task = pr.json()
                        done, total = task["done"], task["total"]

                        progress_bar.progress(done / max(total, 1))
                        status_text.text(
                            f"检测中... {done}/{total} "
                            f"({task['status']})"
                        )

                        if task["status"] == "done":
                            progress_bar.progress(1.0)
                            status_text.text("检测完成！加载结果中...")
                            st.session_state["run_id"] = task["run_id"]
                            st.session_state["results"] = task.get("results")
                            st.session_state["metrics"] = task.get("metrics_data")
                            st.session_state["loaded_run_id"] = task["run_id"]
                            st.success(
                                f"完成！Run: {task['run_id']} | "
                                f"幻觉: {task['hallucination_count']}/{task['total_cases']}"
                            )
                            status_text.empty()
                            break
                        elif task["status"] == "error":
                            st.error(f"检测失败: {task.get('error', '未知错误')}")
                            status_text.empty()
                            break
            except requests.exceptions.ConnectionError:
                st.error("无法连接 Backend")
            except requests.exceptions.Timeout:
                st.error("请求超时")

    st.divider()

    # Run history
    st.header("📂 历史记录")
    if st.button("🔄 刷新记录", use_container_width=True):
        if backend_ok:
            try:
                resp = requests.get(api_url("/runs"), timeout=10)
                if resp.status_code == 200:
                    st.session_state["runs"] = resp.json()
            except Exception:
                pass

    if st.session_state["runs"]:
        run_ids = [r["run_id"] for r in st.session_state["runs"]]
        selected_run = st.selectbox("选择历史 Run", run_ids)
        if st.button("📂 加载此 Run", use_container_width=True):
            if backend_ok:
                try:
                    resp = requests.get(api_url(f"/runs/{selected_run}"), timeout=30)
                    if resp.status_code == 200:
                        rd = resp.json()
                        st.session_state["results"] = rd["results"]
                        st.session_state["metrics"] = rd["metrics"]
                        st.session_state["loaded_run_id"] = selected_run
                        st.success(f"已加载 {selected_run}: {len(rd['results'])} 条")
                except Exception as e:
                    st.error(f"加载失败: {e}")

    st.divider()
    st.header("🔍 单条检测")
    selected_case = st.selectbox("选择 Case", case_ids if case_ids else ["h01"])
    run_single = st.button("🔎 检测此条", use_container_width=True)

    if run_single and all_cases and backend_ok:
        case = next((c for c in all_cases if c["id"] == selected_case), None)
        if case:
            with st.spinner(f"检测 {selected_case}..."):
                try:
                    resp = requests.post(
                        api_url("/detect"),
                        json={
                            "id": case["id"],
                            "user_question": case["user_question"],
                            "system_reply": case["system_reply"],
                            "knowledge_base": case["knowledge_base"],
                        },
                        timeout=120,
                    )
                    if resp.status_code == 200:
                        from src.state import HallucinationResult
                        st.session_state["single_result"] = HallucinationResult(**resp.json())
                        st.session_state["single_case"] = case
                    else:
                        st.error(f"检测失败: {resp.text}")
                except Exception as e:
                    st.error(f"请求失败: {e}")


# ── Main Area ───────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["📊 评估总览", "📋 Case 详情", "🔍 单条分析"])

# Tab 1: Evaluation Overview
with tab1:
    results = st.session_state.get("results")
    metrics = st.session_state.get("metrics")
    run_id = st.session_state.get("loaded_run_id")

    if results:
        if run_id:
            st.caption(f"Run: {run_id}")

        if metrics:
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precision", f"{metrics['precision']:.2%}")
            with col2:
                st.metric("Recall", f"{metrics['recall']:.2%}")
            with col3:
                st.metric("F1-Score", f"{metrics['f1']:.4f}")
            with col4:
                st.metric("Type Accuracy", f"{metrics['type_accuracy']:.2%}")

            # Confusion matrix
            st.subheader("混淆矩阵")
            y_true_vals = []
            y_pred_vals = []
            gt_path = "data/ground_truth.json"
            if os.path.exists(gt_path):
                with open(gt_path, "r", encoding="utf-8") as f:
                    gt_dict = {item["id"]: item for item in json.load(f)}
                for pred in results:
                    pid = pred.get("id", pred.get("id", ""))
                    if pid in gt_dict:
                        y_true_vals.append(1 if gt_dict[pid].get("is_hallucination") else 0)
                        y_pred_vals.append(1 if pred.get("is_hallucination") else 0)

            if len(set(y_true_vals)) >= 2:
                cm = confusion_matrix(y_true_vals, y_pred_vals)
                fig, ax = plt.subplots(figsize=(4, 3))
                ConfusionMatrixDisplay(cm, display_labels=["No Hallucination", "Hallucination"]).plot(ax=ax)
                st.pyplot(fig)
            elif y_true_vals:
                st.info("所有样本标签相同，无法绘制混淆矩阵")

        # Type distribution
        st.subheader("幻觉类型分布")
        type_counts = {}
        for r in results:
            t = r.get("hallucination_type", "无")
            type_counts[t] = type_counts.get(t, 0) + 1

        if type_counts:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            labels = list(type_counts.keys())
            sizes = list(type_counts.values())
            colors = ['#EF5350', '#FF7043', '#FFA726', '#FFCA28',
                      '#66BB6A', '#42A5F5', '#AB47BC', '#8D6E63', '#78909C']

            wedges, texts, autotexts = ax1.pie(
                sizes, labels=labels, autopct='%1.1f%%',
                colors=colors[:len(labels)], startangle=140
            )
            ax1.set_title("幻觉类型占比")

            bars = ax2.bar(labels, sizes, color=colors[:len(labels)])
            ax2.set_title("幻觉类型计数")
            ax2.set_ylabel("数量")
            ax2.tick_params(axis='x', rotation=45)
            for bar, v in zip(bars, sizes):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                         str(v), ha='center', fontsize=9)

            plt.tight_layout()
            st.pyplot(fig)

        # Error cases
        if metrics:
            st.subheader("误判分析")
            fp_cases = metrics.get("fp_cases", [])
            fn_cases = metrics.get("fn_cases", [])
            type_mismatches = metrics.get("type_mismatches", [])

            if fp_cases or fn_cases:
                err_col1, err_col2 = st.columns(2)
                with err_col1:
                    st.markdown("**误报 (FP)**")
                    for c in fp_cases:
                        st.text(c)
                    if not fp_cases:
                        st.text("无")
                with err_col2:
                    st.markdown("**漏检 (FN)**")
                    for c in fn_cases:
                        st.text(c)
                    if not fn_cases:
                        st.text("无")

            if type_mismatches:
                st.markdown("**类型误判**")
                for m in type_mismatches:
                    st.text(f"{m['id']}: 预测={m['predicted']} vs 实际={m['actual']}")

        # Worst cases
        st.subheader("最差 3 条 Case")
        results_with_errors = sorted(
            [r for r in results if r.get("is_hallucination")],
            key=lambda r: sum(1 for c in r.get("claims", []) if c.get("nli_status") == "CONTRADICTED"),
            reverse=True
        )
        for i, case in enumerate(results_with_errors[:3]):
            n_c = sum(1 for c in case.get("claims", []) if c.get("nli_status") == "CONTRADICTED")
            with st.expander(
                f"#{i+1} {case['id']} — {case.get('hallucination_type')} "
                f"({case.get('severity')}) [{n_c} 矛盾声明]"
            ):
                st.write(case.get("detail", ""))

        # Charts from backend
        if run_id and backend_ok:
            try:
                rd = requests.get(api_url(f"/runs/{run_id}"), timeout=10)
                if rd.status_code == 200:
                    charts = rd.json().get("charts", {})
                    if charts:
                        st.subheader("图表")
                        chart_cols = st.columns(len(charts))
                        for i, (name, url) in enumerate(charts.items()):
                            with chart_cols[i % len(charts)]:
                                st.image(f"{st.session_state.backend_url}{url}",
                                         caption=name, use_container_width=True)
            except Exception:
                pass
    else:
        st.info("👈 请先在左侧边栏运行批量检测或加载历史记录")

# Tab 2: Case Details
with tab2:
    results = st.session_state.get("results")
    if results:
        filter_type = st.multiselect(
            "幻觉类型筛选",
            ["能力越界", "安全误导", "参数编造", "信息编造", "政策编造", "优惠编造", "政策偏差", "信息遗漏", "无"],
            default=[]
        )
        filter_severity = st.multiselect(
            "严重度筛选",
            ["Critical", "High", "Medium", "None"],
            default=[]
        )

        filtered = results
        if filter_type:
            filtered = [r for r in filtered if r.get("hallucination_type") in filter_type]
        if filter_severity:
            filtered = [r for r in filtered if r.get("severity") in filter_severity]

        st.write(f"显示 {len(filtered)} / {len(results)} 条")

        for case in filtered:
            claims = case.get("claims", [])
            n_contradicted = sum(1 for c in claims if c.get("nli_status") == "CONTRADICTED")
            n_entailed = sum(1 for c in claims if c.get("nli_status") == "ENTAILED")
            n_unmentioned = sum(1 for c in claims if c.get("nli_status") == "UNMENTIONED")

            status_color = ":red[幻觉]" if case.get("is_hallucination") else ":green[正常]"
            with st.expander(
                f"{case['id']} {status_color} — {case.get('hallucination_type', 'N/A')} "
                f"({case.get('severity', 'N/A')}) | Claims: {n_entailed}✓ {n_contradicted}✗ {n_unmentioned}?"
            ):
                col_a, col_b = st.columns([1, 2])
                with col_a:
                    st.markdown(f"**类型**: {case.get('hallucination_type')}")
                    st.markdown(f"**严重度**: {case.get('severity')}")
                    st.markdown(f"**结果**: {n_contradicted} 矛盾 / {n_entailed} 一致 / {n_unmentioned} 未提及")
                with col_b:
                    st.markdown("**分析**:")
                    st.text(case.get("detail", ""))

                st.divider()
                st.markdown("**Claim 验证详情**")
                for j, claim in enumerate(claims):
                    status = claim.get("nli_status", "PENDING")
                    icon = {"ENTAILED": "✅", "CONTRADICTED": "❌", "UNMENTIONED": "❓"}.get(status, "⏳")
                    st.caption(
                        f"{icon} [{claim.get('claim_type', '?')}] {claim.get('claim_text', '')} "
                        f"→ {status}"
                    )
                    if claim.get("reasoning"):
                        st.caption(f"   理由: {claim['reasoning']}")
    else:
        st.info("👈 请先在左侧边栏运行批量检测或加载历史记录")

# Tab 3: Single Case Analysis
with tab3:
    single_result = st.session_state.get("single_result")
    single_case = st.session_state.get("single_case")

    if single_result and single_case:
        st.subheader(f"Case: {single_case['id']}")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**用户问题**")
            st.info(single_case["user_question"])
        with col2:
            st.markdown("**系统回复**")
            st.warning(single_case["system_reply"])
        with col3:
            st.markdown("**知识库**")
            st.success(single_case["knowledge_base"])

        st.divider()

        if single_result.is_hallucination:
            st.error(f"🚨 检测到幻觉 — 类型: {single_result.hallucination_type} | 严重度: {single_result.severity}")
        else:
            st.success(f"✅ 无幻觉 — {single_result.detail}")

        st.markdown(f"**分析**: {single_result.detail}")

        st.subheader("Claim 验证过程")
        for i, claim in enumerate(single_result.claims):
            status = claim.nli_status or "PENDING"
            icon = {"ENTAILED": "✅", "CONTRADICTED": "❌", "UNMENTIONED": "❓"}.get(status, "⏳")
            with st.container():
                st.markdown(f"**Claim {i+1}**: {icon} `[{claim.claim_type}]` {claim.claim_text}")
                st.caption(f"  NLI: {status} | {claim.reasoning or ''}")
    else:
        st.info("👈 请在左侧边栏选择 Case 并点击检测")
