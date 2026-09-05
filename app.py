"""
app.py — LedgerGuard: AI Finance Controller

Streamlit presentation/orchestration layer. All actual logic lives in the
backend packages (data/, matching/, agent/, anomaly/, audit/) — this file
wires them together and renders them, it does not reimplement them.

Run with:  streamlit run app.py   (from the project root — see README.md)
"""

import time
import io
import csv
import pandas as pd
import altair as alt
import streamlit as st

from data import generate_data
from matching import rule_matcher as matcher
from matching import build_training_data
from matching import ml_matcher
from agent import investigate
from anomaly import detector as anomaly
from audit import audit_log as audit
from evaluation import evaluate as pipeline_evaluation

AUTO_MATCH_TYPES = ("exact", "fuzzy", "fee_adjusted", "refund_adjusted")

st.set_page_config(page_title="LedgerGuard — AI Finance Controller", layout="wide", page_icon="🛡️")



# animations

st.markdown("""
<style>
    :root {
        --lg-accent: #22d3ee;
        --lg-accent-dim: #0e7490;
        --lg-bg: #0a0e14;
        --lg-panel: #121822;
        --lg-border: #1f2937;
    }
    .stApp { background: radial-gradient(circle at 20% 0%, #0d1420 0%, var(--lg-bg) 45%); }

    /* Top accent line across the whole app */
    .stApp::before {
        content: ""; position: fixed; top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, var(--lg-accent), transparent 70%);
        z-index: 999;
    }

    div[data-testid="stMetricValue"] { font-size: 1.65rem; font-weight: 700; color: #f0f6fc; }
    div[data-testid="stMetricLabel"] { color: #8b949e; letter-spacing: 0.02em; text-transform: uppercase; font-size: 0.72rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, var(--lg-panel) 0%, #0d131c 100%);
        border: 1px solid var(--lg-border); border-radius: 10px; padding: 14px 18px;
        box-shadow: 0 0 0 1px rgba(34,211,238,0.03), 0 4px 14px rgba(0,0,0,0.35);
        transition: border-color 0.15s ease;
    }
    div[data-testid="stMetric"]:hover { border-color: var(--lg-accent-dim); }

    .status-badge {
        display: inline-block; padding: 3px 12px; border-radius: 999px;
        font-size: 0.76rem; font-weight: 650; letter-spacing: 0.01em;
    }
    .badge-matched { background-color: rgba(74,222,128,0.12); color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }
    .badge-unresolved { background-color: rgba(248,113,113,0.12); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }
    .badge-review { background-color: rgba(251,191,36,0.12); color: #fbbf24; border: 1px solid rgba(251,191,36,0.25); }

    h1, h2, h3 { font-weight: 700; letter-spacing: -0.01em; }
    h1 { background: linear-gradient(90deg, #f0f6fc, #7dd3fc 120%);
         -webkit-background-clip: text; background-clip: text; color: transparent; }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1420 0%, var(--lg-bg) 100%);
        border-right: 1px solid var(--lg-border);
    }
    .lg-brand { font-size: 1.5rem; font-weight: 800; letter-spacing: -0.02em;
                background: linear-gradient(90deg, #22d3ee, #818cf8);
                -webkit-background-clip: text; background-clip: text; color: transparent; }
    .lg-tagline { color: #8b949e; font-size: 0.82rem; font-style: italic; line-height: 1.4; }

    .lg-hero { text-align: center; padding: 56px 20px 40px; }
    .lg-hero-title { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.02em;
                      background: linear-gradient(90deg, #22d3ee, #818cf8);
                      -webkit-background-clip: text; background-clip: text; color: transparent; }
    .lg-hero-sub { color: #8b949e; font-size: 1.05rem; margin-top: 8px; max-width: 640px; margin-left: auto; margin-right: auto; }
    .lg-pill-row { display: flex; gap: 10px; justify-content: center; margin-top: 22px; flex-wrap: wrap; }
    .lg-pill { border: 1px solid var(--lg-border); background: var(--lg-panel); color: #c9d1d9;
               padding: 6px 14px; border-radius: 999px; font-size: 0.82rem; }
</style>
""", unsafe_allow_html=True)



# Session state initialization

def init_state():
    defaults = {
        "gateway_records": None, "bank_records": None, "ground_truth_records": None,
        "match_results": None, "processing_time": None, "data_source": None,
        "ml_result": None, "pipeline_metrics": None, "human_overrides": {}, "dataset_size": 500,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()



# Pipeline helpers

def run_reconciliation():
    t0 = time.time()
    results = matcher.run_matcher(
        gateway_records=st.session_state.gateway_records,
        bank_records=st.session_state.bank_records,
    )
    st.session_state.match_results = results
    st.session_state.processing_time = time.time() - t0
    st.session_state.human_overrides = {}
    st.session_state.pipeline_metrics = None
    if st.session_state.ground_truth_records is not None:
        truth = {r["gateway_txn_id"]: {"bank_txn_id": r["bank_txn_id"], "match_type": r["match_type"]}
                 for r in st.session_state.ground_truth_records if r["gateway_txn_id"]}
        st.session_state.pipeline_metrics = pipeline_evaluation.evaluate(results, truth)[0]
    audit.clear_audit_log()

    # Log every automated decision to the audit trail immediately
    for r in results:
        candidates_considered = []  # full candidate list logged lazily in Investigator to avoid recompute cost here
        audit.log_decision(
            gateway_txn_id=r["gateway_txn_id"], bank_txn_id=r["bank_txn_id"],
            candidates_considered=candidates_considered,
            selected_candidate=r["bank_txn_id"] if r["match_type"] in AUTO_MATCH_TYPES else None,
            decision=r["match_type"], confidence=r["confidence"],
            model_version="rule_v1", threshold=matcher.FUZZY_CONFIDENCE_THRESHOLD,
            human_override=False, reason=r.get("reason", ""),
        )


def train_ml_if_possible():
    """ML training/evaluation requires ground truth labels, which only exist
    for demo-generated data. Uploaded real data has no labels, so this is
    honestly skipped rather than faked."""
    if st.session_state.ground_truth_records is None:
        st.session_state.ml_result = None
        return
    rows = build_training_data.build_labeled_pairs(
        gateway_records=st.session_state.gateway_records,
        bank_records=st.session_state.bank_records,
        ground_truth_records=st.session_state.ground_truth_records,
    )
    n_pos = sum(r["label"] for r in rows)
    if n_pos < 20:
        st.session_state.ml_result = None  # not enough positive examples to train/evaluate meaningfully
        return
    st.session_state.ml_result = ml_matcher.train_and_evaluate(rows)


def get_gateway_by_id():
    return {g["gateway_txn_id"]: g for g in st.session_state.gateway_records}


def get_bank_by_id():
    return {b["bank_txn_id"]: b for b in st.session_state.bank_records}


def results_df():
    if not st.session_state.match_results:
        return pd.DataFrame()
    gw_by_id = get_gateway_by_id()
    bk_by_id = get_bank_by_id()
    rows = []
    for r in st.session_state.match_results:
        gw = gw_by_id.get(r["gateway_txn_id"])
        bk = bk_by_id.get(r["bank_txn_id"])
        override = st.session_state.human_overrides.get(r["gateway_txn_id"] or r["bank_txn_id"])
        status = override["decision"] if override else r["match_type"]
        rows.append({
            "gateway_txn_id": r["gateway_txn_id"] or "—",
            "bank_txn_id": r["bank_txn_id"] or "—",
            "gateway_amount": gw["amount"] if gw else None,
            "bank_amount": bk["amount"] if bk else None,
            "amount_diff": round((gw["amount"] - bk["amount"]), 2) if gw and bk else None,
            "gateway_date": gw["date"] if gw else None,
            "bank_date": bk["date"] if bk else None,
            "match_method": r["match_type"],
            "confidence": r["confidence"],
            "status": status,
            "reason": r.get("reason", ""),
            "human_reviewed": override is not None,
        })
    return pd.DataFrame(rows)



# Sidebar navigation

st.sidebar.markdown(
    '<div class="lg-brand">🛡️ LedgerGuard</div>'
    '<div class="lg-tagline">Reconcile money movement.<br>Explain every decision.<br>Escalate uncertainty.</div>',
    unsafe_allow_html=True,
)
st.sidebar.divider()

PAGES = ["Data / Run", "Command Center", "Reconciliation", "Transaction Investigator",
         "Exception Center", "Human Review", "Model / Evaluation", "Audit Log"]

has_data = st.session_state.match_results is not None
default_page = "Data / Run" if not has_data else "Command Center"
page = st.sidebar.radio("Navigate", PAGES, index=PAGES.index(default_page))

if has_data:
    st.sidebar.divider()
    st.sidebar.caption(f"Loaded: {len(st.session_state.gateway_records)} gateway / "
                        f"{len(st.session_state.bank_records)} bank records")
    st.sidebar.caption(f"Source: {st.session_state.data_source}")



# PAGE: Data / Run

if page == "Data / Run":
    st.title("Data / Run")
    st.caption("Load a dataset and run the reconciliation pipeline end to end.")

    tab_demo, tab_upload = st.tabs(["Demo Mode", "Upload CSVs"])

    with tab_demo:
        st.write("Generates a realistic synthetic batch (exact matches, fee deductions, "
                 "refunds, date lag, mangled references, duplicates, and deliberately "
                 "ambiguous decoy cases) and runs the full pipeline in one click.")
        n = st.number_input("Dataset size", min_value=50, max_value=5000, value=500, step=50)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        if st.button("Generate & Run Demo", type="primary"):
            with st.spinner("Generating synthetic data..."):
                gw, bk, gt = generate_data.build_dataset(n_records=int(n), seed=int(seed))
                st.session_state.gateway_records = gw
                st.session_state.bank_records = bk
                st.session_state.ground_truth_records = gt
                st.session_state.data_source = f"Demo (n={n}, seed={seed})"
            with st.spinner("Running reconciliation..."):
                run_reconciliation()
            with st.spinner("Training ML comparison model..."):
                train_ml_if_possible()
            st.success(f"Done — {len(gw)} gateway / {len(bk)} bank records processed "
                       f"in {st.session_state.processing_time:.3f}s")
            st.rerun()

    with tab_upload:
        st.write("Upload your own gateway transactions and bank statement CSVs. "
                 "Required columns: `gateway_txn_id, reference, amount, date` and "
                 "`bank_txn_id, reference, amount, date` respectively.")
        gw_file = st.file_uploader("Gateway transactions CSV", type="csv", key="gw_upload")
        bk_file = st.file_uploader("Bank statement CSV", type="csv", key="bk_upload")
        if st.button("Run Reconciliation on Uploaded Data", type="primary", disabled=not (gw_file and bk_file)):
            gw_records = list(csv.DictReader(io.StringIO(gw_file.getvalue().decode("utf-8"))))
            bk_records = list(csv.DictReader(io.StringIO(bk_file.getvalue().decode("utf-8"))))
            st.session_state.gateway_records = gw_records
            st.session_state.bank_records = bk_records
            st.session_state.ground_truth_records = None  # no labels for real uploaded data — honest, no ML eval
            st.session_state.data_source = f"Uploaded ({gw_file.name}, {bk_file.name})"
            with st.spinner("Running reconciliation..."):
                run_reconciliation()
            st.session_state.ml_result = None
            st.success(f"Done — {len(gw_records)} gateway / {len(bk_records)} bank records processed "
                       f"in {st.session_state.processing_time:.3f}s")
            st.info("No ground truth labels available for uploaded data, so ML comparison and "
                    "precision/recall metrics are not shown for this run — the Reconciliation "
                    "and Exception Center pages still work fully.")
            st.rerun()

    if st.session_state.match_results is not None:
        st.divider()
        st.subheader("Export")
        df = results_df()
        st.download_button("Download match_results.csv", df.to_csv(index=False),
                            file_name="match_results.csv", mime="text/csv")



# PAGE: Command Center

elif page == "Command Center":
    if st.session_state.match_results is None:
        st.markdown("""
        <div class="lg-hero">
            <div class="lg-hero-title">🛡️ LedgerGuard</div>
            <div class="lg-hero-sub">AI Finance Controller — reconcile money movement,
            explain every decision, escalate uncertainty. No data loaded yet.</div>
            <div class="lg-pill-row">
                <span class="lg-pill">Rule + ML matching, compared honestly</span>
                <span class="lg-pill">Full audit trail</span>
                <span class="lg-pill">Human-in-the-loop review</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.info("Go to **Data / Run** in the sidebar to generate demo data or upload CSVs.")
    else:
        st.title("Command Center")
        df = results_df()
        n_gateway = len(st.session_state.gateway_records)
        n_bank = len(st.session_state.bank_records)
        matched = df[df["match_method"].isin(AUTO_MATCH_TYPES)]
        unresolved = df[df["match_method"].isin(["unresolved", "human_review"])]
        total_value = sum(g["amount"] for g in st.session_state.gateway_records)
        matched_value = matched["gateway_amount"].sum()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Gateway records", n_gateway)
        c2.metric("Bank records", n_bank)
        c3.metric("Matched", len(matched))
        c4.metric("Exceptions / review", len(unresolved))

        c5, c6, c7, c8 = st.columns(4)
        automation_rate = len(matched) / n_gateway if n_gateway else 0
        c5.metric("Automation rate", f"{automation_rate:.1%}")
        c6.metric("Total value processed", f"₹{total_value:,.0f}")
        c7.metric("Matched value", f"₹{matched_value:,.0f}")
        c8.metric("Processing time", f"{st.session_state.processing_time:.3f}s")

        st.divider()
        if st.session_state.pipeline_metrics:
            pm = st.session_state.pipeline_metrics
            st.subheader("Full pipeline performance")
            a, b, c, d = st.columns(4)
            a.metric("Precision", f"{pm['precision']:.1%}")
            b.metric("Recall", f"{pm['recall']:.1%}")
            c.metric("F1", f"{pm['f1']:.3f}")
            d.metric("False-match rate", f"{pm['false_match_rate']:.1%}")
            st.caption("Actual record-assignment results. Pairwise ML diagnostics below use a separate held-out protocol.")

        st.divider()
        if st.session_state.ml_result:
            st.subheader("Measured accuracy (held-out test set)")
            rm = st.session_state.ml_result["rule_metrics"]
            mm = st.session_state.ml_result["ml_metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Rule precision", f"{rm['precision']:.1%}")
            c2.metric("Rule recall", f"{rm['recall']:.1%}")
            c3.metric("Rule F1", f"{rm['f1']:.3f}")
            c4.metric("Rule false-match rate", f"{rm['fp']/(rm['fp']+rm['tp']) if (rm['fp']+rm['tp']) else 0:.1%}")
            st.caption("Full rule-vs-ML comparison is on the Model / Evaluation page.")
        else:
            st.info("**Not evaluated.** Precision/recall/F1 require ground truth labels, only "
                    "available for demo-generated data. Run Demo Mode on the Data / Run page to see these.")

        st.divider()
        st.subheader("Reconciliation status")
        status_counts = df["status"].value_counts().reset_index()
        status_counts.columns = ["status", "count"]
        status_chart = alt.Chart(status_counts).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("status:N", sort="-y", title=None),
            y=alt.Y("count:Q", title="Records"),
            color=alt.Color("status:N", scale=alt.Scale(range=["#22d3ee", "#818cf8", "#f87171", "#fbbf24", "#4ade80"]), legend=None),
            tooltip=["status", "count"],
        ).properties(height=280)
        st.altair_chart(status_chart, use_container_width=True)



# PAGE: Reconciliation

elif page == "Reconciliation":
    st.title("Reconciliation")
    if st.session_state.match_results is None:
        st.info("No data loaded yet. Go to **Data / Run** first.")
    else:
        df = results_df()
        filter_options = ["All"] + sorted(df["match_method"].unique().tolist())
        selected = st.selectbox("Filter by match method", filter_options)
        filtered = df if selected == "All" else df[df["match_method"] == selected]

        def badge(row):
            m = row["match_method"]
            if m in AUTO_MATCH_TYPES:
                return "MATCHED"
            if m == "human_review":
                return "HUMAN REVIEW / AMBIGUOUS"
            return "UNRESOLVED"

        display_df = filtered.copy()
        display_df["status_display"] = display_df.apply(badge, axis=1)
        st.dataframe(
            display_df[["gateway_txn_id", "bank_txn_id", "status_display", "gateway_amount",
                        "bank_amount", "amount_diff", "gateway_date", "bank_date",
                        "match_method", "confidence", "reason"]],
            use_container_width=True, height=500,
        )
        st.caption(f"Showing {len(filtered)} of {len(df)} records")

# PAGE: Transaction Investigator

elif page == "Transaction Investigator":
    st.title("Transaction Investigator")
    if st.session_state.match_results is None:
        st.info("No data loaded yet. Go to **Data / Run** first.")
    else:
        gw_by_id = get_gateway_by_id()
        gw_ids = sorted(gw_by_id.keys())
        selected_id = st.selectbox("Select a gateway transaction", gw_ids)

        if selected_id:
            gw_record = gw_by_id[selected_id]
            model = st.session_state.ml_result["model"] if st.session_state.ml_result else None
            scaler = st.session_state.ml_result["scaler"] if st.session_state.ml_result else None

            result = investigate.investigate_transaction(
                gw_record, st.session_state.bank_records, model=model, scaler=scaler,
            )

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Gateway record")
                st.json(gw_record)
            with col2:
                st.subheader("Decision")
                st.markdown(f"**{result['decision']}**")
                st.write(result["reason"])

            st.divider()
            st.subheader(f"Candidate bank records ({len(result['candidates'])} found within blocking window)")
            if result["candidates"]:
                cand_df = pd.DataFrame(result["candidates"])
                st.dataframe(cand_df, use_container_width=True)
            else:
                st.warning("No bank records fall within the date/amount blocking window for this transaction.")



# PAGE: Exception Center

elif page == "Exception Center":
    st.title("Exception Center")
    if st.session_state.match_results is None:
        st.info("No data loaded yet. Go to **Data / Run** first.")
    else:
        unresolved = [r for r in st.session_state.match_results if r["match_type"] in ("unresolved", "human_review")]
        gw_by_id = get_gateway_by_id()
        bk_by_id = get_bank_by_id()

        # Duplicate detection for categorization
        dup_flags = {r["reference"] for r in anomaly.detect_duplicate_references(st.session_state.gateway_records)}

        def categorize(r):
            if r["gateway_txn_id"] and not r["bank_txn_id"]:
                gw = gw_by_id.get(r["gateway_txn_id"], {})
                if gw.get("reference") in dup_flags:
                    return "Duplicate"
                if r["match_type"] == "human_review":
                    return "Ambiguous candidate"
                if "refund" in r.get("reason", "").lower():
                    return "Refund"
                if "fee" in r.get("reason", "").lower():
                    return "Fee adjustment"
                if "below" in r.get("reason", ""):
                    return "Low confidence"
                return "Missing bank record"
            if r["bank_txn_id"] and not r["gateway_txn_id"]:
                return "Missing gateway record"
            return "Other"

        categories = {}
        for r in unresolved:
            cat = categorize(r)
            categories.setdefault(cat, []).append(r)

        cols = st.columns(len(categories) if categories else 1)
        for i, (cat, items) in enumerate(categories.items()):
            cols[i].metric(cat, len(items))

        st.divider()
        for cat, items in categories.items():
            with st.expander(f"{cat} ({len(items)})"):
                rows = []
                for r in items:
                    gw = gw_by_id.get(r["gateway_txn_id"], {})
                    bk = bk_by_id.get(r["bank_txn_id"], {})
                    rows.append({
                        "gateway_txn_id": r["gateway_txn_id"] or "—",
                        "bank_txn_id": r["bank_txn_id"] or "—",
                        "amount": gw.get("amount") or bk.get("amount"),
                        "date": gw.get("date") or bk.get("date"),
                        "reason": r.get("reason", ""),
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

        if not unresolved:
            st.success("No exceptions in this batch.")



# PAGE: Human Review

elif page == "Human Review":
    st.title("Human Review")
    if st.session_state.match_results is None:
        st.info("No data loaded yet. Go to **Data / Run** first.")
    else:
        st.caption("Medium-confidence cases: candidates exist but scored below the auto-match "
                   "threshold. Review the evidence and decide.")
        review_cases = [r for r in st.session_state.match_results
                        if r["match_type"] in ("unresolved", "human_review") and r["gateway_txn_id"]
                        and r["confidence"] > 0]

        if not review_cases:
            st.success("No cases currently need human review.")
        else:
            gw_by_id = get_gateway_by_id()
            case_ids = [r["gateway_txn_id"] for r in review_cases]
            selected = st.selectbox(f"Cases needing review ({len(review_cases)})", case_ids)

            gw_record = gw_by_id[selected]
            result = investigate.investigate_transaction(gw_record, st.session_state.bank_records)

            st.subheader("Gateway record")
            st.json(gw_record)
            st.subheader("Top candidates")
            if result["candidates"]:
                st.dataframe(pd.DataFrame(result["candidates"][:5]), use_container_width=True)

                top_candidate = result["candidates"][0]
                c1, c2, c3 = st.columns(3)
                if c1.button("✅ Approve top match", key=f"approve_{selected}"):
                    st.session_state.human_overrides[selected] = {
                        "decision": "human_approved", "bank_txn_id": top_candidate["bank_txn_id"],
                    }
                    audit.log_decision(
                        gateway_txn_id=selected, bank_txn_id=top_candidate["bank_txn_id"],
                        candidates_considered=[c["bank_txn_id"] for c in result["candidates"]],
                        selected_candidate=top_candidate["bank_txn_id"], decision="human_approved",
                        confidence=top_candidate["rule_score"], human_override=True,
                        reason="Human reviewer approved the top-scored candidate.",
                    )
                    st.success("Approved and logged to audit trail.")
                    st.rerun()
                if c2.button("❌ Reject match", key=f"reject_{selected}"):
                    st.session_state.human_overrides[selected] = {"decision": "human_rejected", "bank_txn_id": None}
                    audit.log_decision(
                        gateway_txn_id=selected, bank_txn_id=None,
                        candidates_considered=[c["bank_txn_id"] for c in result["candidates"]],
                        selected_candidate=None, decision="human_rejected",
                        confidence=top_candidate["rule_score"], human_override=True,
                        reason="Human reviewer rejected all candidates.",
                    )
                    st.success("Rejected and logged to audit trail.")
                    st.rerun()
                if c3.button("🚫 Mark unresolved", key=f"unresolved_{selected}"):
                    st.session_state.human_overrides[selected] = {"decision": "human_marked_unresolved", "bank_txn_id": None}
                    audit.log_decision(
                        gateway_txn_id=selected, bank_txn_id=None,
                        candidates_considered=[c["bank_txn_id"] for c in result["candidates"]],
                        selected_candidate=None, decision="human_marked_unresolved",
                        confidence=None, human_override=True,
                        reason="Human reviewer marked this case unresolved for later follow-up.",
                    )
                    st.success("Marked unresolved and logged to audit trail.")
                    st.rerun()
            else:
                st.warning("No candidates found for this record.")



# PAGE: Model / Evaluation

elif page == "Model / Evaluation":
    st.title("Model / Evaluation")
    if st.session_state.ml_result is None:
        st.info("**Not evaluated.** This requires ground truth labels, which only exist for "
                "demo-generated data (uploaded data has no labels, so nothing is fabricated here). "
                "Run Demo Mode on the Data / Run page.")
    else:
        res = st.session_state.ml_result
        st.subheader("Held-out test set comparison")
        st.caption("Both approaches evaluated on the identical test set, split by gateway record "
                   "(not by row) to avoid leakage, and touched exactly once.")

        comp_df = pd.DataFrame({
            "Rule-based baseline": res["rule_metrics"],
            "ML classifier": res["ml_metrics"],
        }).T
        st.dataframe(comp_df, use_container_width=True)

        st.divider()
        st.subheader("Feature importance (ML model)")
        fi_df = pd.DataFrame(res["feature_importance"], columns=["feature", "coefficient"])
        fi_df["direction"] = fi_df["coefficient"].apply(lambda x: "positive" if x >= 0 else "negative")
        fi_chart = alt.Chart(fi_df).mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4).encode(
            x=alt.X("coefficient:Q", title="Standardized coefficient"),
            y=alt.Y("feature:N", sort="-x", title=None),
            color=alt.Color("direction:N", scale=alt.Scale(domain=["positive", "negative"], range=["#22d3ee", "#f87171"]), legend=None),
            tooltip=["feature", "coefficient"],
        ).properties(height=260)
        st.altair_chart(fi_chart, use_container_width=True)

        st.divider()
        st.subheader("Split sizes")
        st.write(res["split_sizes"])
        st.caption(f"Positive rate — train: {res['positive_rates']['train']:.1%}, "
                   f"val: {res['positive_rates']['val']:.1%}, test: {res['positive_rates']['test']:.1%}")

        rm, mm = res["rule_metrics"], res["ml_metrics"]
        if rm["f1"] >= mm["f1"]:
            st.info("**Honest finding:** the rule-based baseline matches or beats the ML classifier "
                    "on this dataset. The 6 hand-picked features already capture most of the signal — "
                    "reference similarity dominates both approaches. This is reported as-is, not "
                    "papered over.")
        else:
            st.info("**Finding:** the ML classifier outperforms the rule-based baseline on this "
                    "held-out test set.")


# ===========================================================================
# PAGE: Audit Log
# ===========================================================================
elif page == "Audit Log":
    st.title("Audit Log")
    st.caption("Every automated decision and every human override, in order.")
    log = audit.load_audit_log()
    if not log:
        st.info("No audit entries yet. Run reconciliation from the Data / Run page.")
    else:
        df = pd.DataFrame(log)
        human_only = st.checkbox("Show human overrides only")
        if human_only:
            df = df[df["human_override"] == "True"]
        st.dataframe(df, use_container_width=True, height=500)
        st.caption(f"{len(df)} entries")
