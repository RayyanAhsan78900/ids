"""
ids_core.py — Core modules for the LLM-Powered IDS
UMT Lahore | InfoSec Spring 2026

Modules:
    RAGThreatKnowledgeBase  — FAISS-backed vector store for threat intelligence
    LLMAnalyzer             — Claude/OpenAI-powered threat analysis with RAG
    SignatureIDS            — Snort-style rule engine for baseline comparison
    PipelineOrchestrator    — Ties everything together end-to-end

Usage (standalone):
    from ids_core import PipelineOrchestrator
    pipeline = PipelineOrchestrator(llm_provider="anthropic", api_key="...")
    report   = pipeline.analyze_flow(flow_dict)
"""

import os
import json
import time
import pickle
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# MODULE 1: RAG Threat Knowledge Base
# ─────────────────────────────────────────────────────────────────────────────

class RAGThreatKnowledgeBase:
    """
    Retrieval-Augmented Generation knowledge base for network threat intelligence.
    
    Uses FAISS for approximate nearest-neighbor search over sentence embeddings.
    Each document is a detailed threat profile including MITRE ATT&CK mapping,
    indicators of compromise, and mitigation strategies.
    
    Architecture:
        Query (flow description)
            → SentenceTransformer (384-dim embedding)
            → FAISS cosine similarity search
            → Top-k threat profiles (context for LLM)
    """

    # Threat intelligence documents (extend with CVE/MITRE ATT&CK feeds)
    KNOWLEDGE_BASE = [
        {
            "id": "T001", "attack_type": "DoS_Hulk",
            "title": "HTTP Hulk DoS Attack",
            "description": (
                "Hulk generates unique obfuscated HTTP GET requests to overwhelm web servers. "
                "Key indicators: extremely high flow_bytes_s (>80,000), high fwd_packets (>300), "
                "minimal bwd_packets (server overwhelmed), PSH flag dominance, ports 80/443/8080."
            ),
            "mitre_technique": "T1498.001 — Direct Network Flood",
            "severity": "CRITICAL",
            "cvss_score": 8.6,
            "indicators": "flow_bytes_s > 80000, fwd_packets > 300, dst_port in [80,443,8080]",
            "mitigation": (
                "Rate limiting at edge, CDN/WAF protection, IP blacklisting, "
                "TCP connection rate limiting, Cloudflare / AWS Shield deployment."
            )
        },
        {
            "id": "T002", "attack_type": "PortScan",
            "title": "Network Port Scanning (Reconnaissance)",
            "description": (
                "Systematic port probing to map open services. "
                "Key indicators: very short flow_duration (<150ms), fwd_packets <= 3, "
                "SYN flags present, random dst_port across sequence, low bytes per flow. "
                "Often precedes targeted exploitation."
            ),
            "mitre_technique": "T1046 — Network Service Discovery",
            "severity": "HIGH",
            "cvss_score": 5.3,
            "indicators": "flow_duration < 150, fwd_packets <= 3, syn_flag_count >= 1",
            "mitigation": (
                "Port knocking, firewall rules for sequential scan detection, "
                "IPS/IDS signatures, honeypot ports, source-IP rate limiting."
            )
        },
        {
            "id": "T003", "attack_type": "DDoS",
            "title": "Distributed Denial of Service",
            "description": (
                "Coordinated flood from multiple sources to exhaust target resources. "
                "Key indicators: massive flow_packets_s (>400), elevated SYN flags (SYN flood), "
                "asymmetric traffic (high fwd, near-zero bwd), near-MTU packet sizes (~1500 bytes). "
                "Multiple distinct source IPs in aggregate."
            ),
            "mitre_technique": "T1498 — Network Denial of Service",
            "severity": "CRITICAL",
            "cvss_score": 9.1,
            "indicators": "flow_packets_s > 400, syn_flag_count > 10, fwd_packet_len_mean near 1500",
            "mitigation": (
                "Anycast diffusion, traffic scrubbing centers, BGP blackholing, "
                "upstream ISP filtering, BCP38 to prevent IP spoofing."
            )
        },
        {
            "id": "T004", "attack_type": "FTP-Patator",
            "title": "FTP Credential Brute Force (Patator)",
            "description": (
                "Automated credential stuffing against FTP (port 21). "
                "Key indicators: dst_port == 21, repeated connection attempts from same source, "
                "moderate bidirectional traffic (FTP challenge-response pattern), "
                "consistent packet sizes matching FTP banner + credential exchange."
            ),
            "mitre_technique": "T1110.001 — Brute Force: Password Guessing",
            "severity": "HIGH",
            "cvss_score": 7.5,
            "indicators": "dst_port == 21, repeated src_ip, bwd_packets moderate",
            "mitigation": (
                "Account lockout after 5 attempts, fail2ban, disable FTP (use SFTP), "
                "certificate-based auth, rate limiting on port 21."
            )
        },
        {
            "id": "T005", "attack_type": "SSH-Patator",
            "title": "SSH Credential Brute Force (Patator)",
            "description": (
                "Automated brute-force against SSH (port 22). "
                "Key indicators: dst_port == 22, multiple failed auth attempts, "
                "moderate bidirectional traffic (SSH handshake overhead), "
                "slower rate than FTP due to cryptographic handshake cost."
            ),
            "mitre_technique": "T1110.001 — Brute Force: Password Guessing",
            "severity": "HIGH",
            "cvss_score": 7.8,
            "indicators": "dst_port == 22, flow_duration moderate, repeated from same src_ip",
            "mitigation": (
                "Key-based auth only (disable passwords), fail2ban with strict rules, "
                "port knocking, 2FA, move SSH to non-standard port."
            )
        },
        {
            "id": "T006", "attack_type": "Bot",
            "title": "Botnet C2 Beaconing",
            "description": (
                "Infected host communicating with Command & Control server. "
                "Key indicators: periodic beaconing intervals (regular timing), "
                "small consistent packet sizes, unusual destination IPs or ports, "
                "low but continuous data rate, possible encrypted traffic."
            ),
            "mitre_technique": "T1071 — Application Layer Protocol (C2)",
            "severity": "CRITICAL",
            "cvss_score": 8.8,
            "indicators": "Regular timing intervals, small consistent packets, unusual dst_ip",
            "mitigation": (
                "DNS sinkholing, threat intel feed blocking, egress filtering, "
                "EDR on endpoints, zero-trust network architecture."
            )
        },
        {
            "id": "T007", "attack_type": "Infiltration",
            "title": "Network Infiltration / Lateral Movement",
            "description": (
                "Attacker moving laterally within the network post-compromise. "
                "Key indicators: internal-to-internal unusual traffic, "
                "use of legitimate admin protocols (SMB/RDP/WMI), "
                "abnormal user/service account access patterns."
            ),
            "mitre_technique": "T1021 — Remote Services (Lateral Movement)",
            "severity": "CRITICAL",
            "cvss_score": 9.3,
            "indicators": "Internal subnet traffic anomalies, protocol misuse, admin share access",
            "mitigation": (
                "Network micro-segmentation, privileged access workstations, "
                "user behavior analytics, least-privilege enforcement."
            )
        },
        {
            "id": "T008", "attack_type": "BENIGN",
            "title": "Normal Network Traffic Baseline",
            "description": (
                "Legitimate network traffic. "
                "Key characteristics: balanced bidirectional flow (fwd ≈ bwd packets), "
                "standard protocol ports (80, 443, 22, 53), "
                "moderate flow_bytes_s (<10,000), typical packet sizes (100–800 bytes)."
            ),
            "mitre_technique": "N/A",
            "severity": "NONE",
            "cvss_score": 0.0,
            "indicators": "Balanced fwd/bwd, standard ports, normal rates",
            "mitigation": "N/A — continue routine monitoring."
        },
    ]

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG knowledge base.
        
        Args:
            model_name: HuggingFace sentence transformer model name.
                        'all-MiniLM-L6-v2' is fast (384-dim) and accurate.
        """
        import faiss
        from sentence_transformers import SentenceTransformer

        self.faiss = faiss
        self.model_name = model_name

        print(f"⏳ Loading sentence transformer: {model_name}...")
        self.encoder = SentenceTransformer(model_name)

        self._build_index()
        print(f"✅ Knowledge base ready: {len(self.KNOWLEDGE_BASE)} threat profiles indexed")

    def _build_index(self):
        """Build FAISS index from knowledge base documents."""
        self.doc_texts = []
        for doc in self.KNOWLEDGE_BASE:
            text = (
                f"{doc['title']}. "
                f"{doc['description']} "
                f"Indicators: {doc['indicators']}. "
                f"Mitigation: {doc['mitigation']}"
            )
            self.doc_texts.append(text)

        embeddings = self.encoder.encode(self.doc_texts, show_progress_bar=False)
        embeddings = np.array(embeddings, dtype="float32")
        self.faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = self.faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.dim = dim

    def retrieve(self, query: str, top_k: int = 2) -> list[dict]:
        """
        Retrieve the most relevant threat profiles for a flow description.
        
        Args:
            query:  Natural language description of the network flow.
            top_k:  Number of documents to retrieve.
        
        Returns:
            List of threat profile dicts with added 'similarity' score.
        """
        q_vec = self.encoder.encode([query], show_progress_bar=False).astype("float32")
        self.faiss.normalize_L2(q_vec)
        distances, indices = self.index.search(q_vec, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.KNOWLEDGE_BASE):
                doc = self.KNOWLEDGE_BASE[idx].copy()
                doc["similarity"] = float(dist)
                results.append(doc)
        return results

    def format_context(self, docs: list[dict]) -> str:
        """Format retrieved documents as LLM context string."""
        lines = []
        for i, doc in enumerate(docs, 1):
            lines.append(
                f"[Retrieved Threat {i}] {doc['title']} "
                f"(Severity: {doc['severity']}, CVSS: {doc['cvss_score']}, "
                f"Similarity: {doc['similarity']:.3f})\n"
                f"  Description: {doc['description']}\n"
                f"  MITRE: {doc['mitre_technique']}\n"
                f"  Indicators: {doc['indicators']}\n"
                f"  Mitigation: {doc['mitigation']}"
            )
        return "\n\n".join(lines)

    def save(self, index_path: str, docs_path: str):
        """Persist FAISS index and documents to disk."""
        self.faiss.write_index(self.index, index_path)
        with open(docs_path, "wb") as f:
            pickle.dump({"docs": self.KNOWLEDGE_BASE, "texts": self.doc_texts}, f)
        print(f"✅ Knowledge base saved → {index_path}, {docs_path}")

    @classmethod
    def load(cls, index_path: str, docs_path: str, model_name: str = "all-MiniLM-L6-v2"):
        """Load a persisted knowledge base from disk (faster than rebuilding)."""
        import faiss
        from sentence_transformers import SentenceTransformer

        obj = cls.__new__(cls)
        obj.faiss = faiss
        obj.model_name = model_name
        obj.encoder = SentenceTransformer(model_name)

        obj.index = faiss.read_index(index_path)
        with open(docs_path, "rb") as f:
            data = pickle.load(f)
        obj.KNOWLEDGE_BASE = data["docs"]
        obj.doc_texts = data["texts"]
        obj.dim = obj.index.d
        print(f"✅ Knowledge base loaded from {index_path}")
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 2: LLM Threat Analyzer
# ─────────────────────────────────────────────────────────────────────────────

class LLMAnalyzer:
    """
    LLM-powered network flow analyzer using RAG context injection.
    
    Supports both Anthropic Claude and OpenAI GPT APIs.
    Output is always structured JSON for downstream processing.
    """

    SYSTEM_PROMPT = """You are an expert Network Security Analyst embedded in an Intrusion Detection System.
Analyze network traffic flows and produce concise, actionable JSON security reports.

For each flow, output ONLY a valid JSON object with these exact fields:
{
  "threat_type":          "<attack name or BENIGN>",
  "severity":             "<CRITICAL|HIGH|MEDIUM|LOW|NONE>",
  "confidence":           <integer 0-100>,
  "explanation":          "<2-3 sentences explaining WHY this classification>",
  "indicators_triggered": ["<specific feature values that triggered>"],
  "recommendations":      "<immediate actions in 1-2 sentences>",
  "mitre_technique":      "<T-code and name, or N/A>"
}

Base your analysis on both the traffic features AND the retrieved threat intelligence context.
Be precise. Do not add markdown, explanations outside the JSON, or extra fields."""

    def __init__(self, provider: str = "anthropic", api_key: str = None):
        """
        Initialize LLM client.
        
        Args:
            provider:  "anthropic" or "openai"
            api_key:   API key (or set env vars ANTHROPIC_API_KEY / OPENAI_API_KEY)
        """
        self.provider = provider.lower()
        self.api_key  = api_key or os.getenv(
            "ANTHROPIC_API_KEY" if self.provider == "anthropic" else "OPENAI_API_KEY"
        )

        if self.provider == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=self.api_key)
            self.model  = "claude-haiku-4-5-20251001"
        elif self.provider == "openai":
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            self.model  = "gpt-4o-mini"
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'anthropic' or 'openai'.")

        print(f"✅ LLM Analyzer ready: {self.provider.upper()} / {self.model}")

    def _call_llm(self, user_prompt: str, max_tokens: int = 500) -> str:
        """Raw LLM call, returns string response."""
        if self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}]
            )
            return response.content[0].text
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user",   "content": user_prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

    def analyze(
        self,
        flow: dict,
        ml_prediction: str,
        anomaly_score: float,
        rag_context: str = "",
        retry: int = 2
    ) -> dict:
        """
        Analyze a single network flow using RAG context + LLM.
        
        Args:
            flow:           Dict of flow features (from processed dataset row).
            ml_prediction:  Attack label from Random Forest classifier.
            anomaly_score:  Isolation Forest anomaly score (higher = more suspicious).
            rag_context:    Formatted string of retrieved threat profiles.
            retry:          Number of retries on failure.
        
        Returns:
            dict: Structured threat report with LLM analysis.
        """
        prompt = f"""NETWORK FLOW ANALYSIS

=== TRAFFIC FEATURES ===
Source IP:           {flow.get('src_ip', 'N/A')}
Destination IP:      {flow.get('dst_ip', 'N/A')}
Destination Port:    {flow.get('dst_port', 0)}
Protocol:            {flow.get('protocol', 'TCP')}
Flow Duration (ms):  {flow.get('flow_duration', 0)}
Forward Packets:     {flow.get('fwd_packets', 0)}
Backward Packets:    {flow.get('bwd_packets', 0)}
Bytes per Second:    {flow.get('flow_bytes_s', 0):.2f}
Packets per Second:  {flow.get('flow_packets_s', 0):.2f}
Fwd Pkt Len (mean):  {flow.get('fwd_packet_len_mean', 0)}
SYN Flags:           {flow.get('syn_flag_count', 0)}
PSH Flags:           {flow.get('psh_flag_count', 0)}
ACK Flags:           {flow.get('ack_flag_count', 0)}
Packet Ratio (F/B):  {flow.get('packet_ratio', 0):.2f}
Flag Ratio (S/A):    {flow.get('flag_ratio', 0):.4f}

=== ML MODEL OUTPUTS ===
Random Forest Label:        {ml_prediction}
Isolation Forest Score:     {anomaly_score:.4f}  (higher = more anomalous)
Anomaly Flag:               {'YES — SUSPICIOUS' if anomaly_score > 0.3 else 'NO — Normal range'}

=== RETRIEVED THREAT INTELLIGENCE (RAG) ===
{rag_context if rag_context else 'No context retrieved.'}

Provide your JSON analysis now:"""

        for attempt in range(retry + 1):
            try:
                raw = self._call_llm(prompt)
                # Strip markdown fences
                cleaned = raw.strip()
                for fence in ["```json", "```"]:
                    cleaned = cleaned.removeprefix(fence).removesuffix(fence).strip()

                result = json.loads(cleaned)
                result["_raw_response"] = raw
                result["_provider"]     = self.provider
                result["_model"]        = self.model
                return result

            except json.JSONDecodeError:
                if attempt < retry:
                    time.sleep(1)
                    continue
                # Final fallback
                return {
                    "threat_type":          ml_prediction,
                    "severity":             "HIGH" if ml_prediction != "BENIGN" else "NONE",
                    "confidence":           65,
                    "explanation":          f"LLM JSON parse failed after {retry+1} attempts. Raw: {raw[:200]}",
                    "indicators_triggered": [],
                    "recommendations":      "Manual review required.",
                    "mitre_technique":      "Unknown",
                    "_parse_error":         True
                }
            except Exception as e:
                return {
                    "threat_type":          ml_prediction,
                    "severity":             "UNKNOWN",
                    "confidence":           0,
                    "explanation":          f"LLM API error: {str(e)[:200]}",
                    "indicators_triggered": [],
                    "recommendations":      "Check API key and network connection.",
                    "mitre_technique":      "N/A",
                    "_api_error":           str(e)
                }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 3: Signature-Based IDS (Snort-style)
# ─────────────────────────────────────────────────────────────────────────────

class SignatureBasedIDS:
    """
    Traditional signature-based IDS implementing Snort-inspired detection rules.
    
    Used as a baseline for comparison against the LLM+ML system.
    Limitation: cannot detect zero-day attacks or novel traffic patterns
    that don't match predefined rule thresholds.
    """

    RULES = [
        # (rule_id, description, detection_lambda)
        ("SIG-001", "DoS Hulk: HTTP Flood",
         lambda r: r.get("flow_bytes_s", 0) > 80000
                   and r.get("fwd_packets", 0) > 300
                   and r.get("dst_port", 0) in [80, 443, 8080]),

        ("SIG-002", "DDoS: High-Volume Network Flood",
         lambda r: r.get("flow_packets_s", 0) > 400
                   and r.get("syn_flag_count", 0) > 10),

        ("SIG-003", "PortScan: Sequential Port Probe",
         lambda r: r.get("flow_duration", 9999) < 150
                   and r.get("fwd_packets", 99) <= 3
                   and r.get("syn_flag_count", 0) >= 1),

        ("SIG-004", "BruteForce: FTP Credential Attack",
         lambda r: r.get("dst_port", 0) == 21
                   and r.get("fwd_packets", 0) > 4),

        ("SIG-005", "BruteForce: SSH Credential Attack",
         lambda r: r.get("dst_port", 0) == 22
                   and r.get("fwd_packets", 0) > 4
                   and r.get("bwd_packets", 0) > 2),

        ("SIG-006", "Botnet: Low-Volume Periodic Beaconing",
         lambda r: 0 < r.get("flow_packets_s", 0) < 2
                   and r.get("flow_duration", 0) > 5000
                   and r.get("dst_port", 0) not in [80, 443, 53]),
    ]

    def detect(self, flow: dict) -> tuple[int, str, str]:
        """
        Apply all rules to a flow.
        
        Args:
            flow: Dict of flow features.
        
        Returns:
            Tuple of (prediction: 0|1, rule_id: str, description: str)
        """
        for rule_id, description, condition in self.RULES:
            try:
                if condition(flow):
                    return 1, rule_id, description
            except Exception:
                continue
        return 0, "NONE", "BENIGN — no rule matched"

    def evaluate(
        self, df: pd.DataFrame, y_true_col: str = "is_attack"
    ) -> dict:
        """
        Evaluate signature IDS on a dataframe.
        
        Returns:
            Dict of performance metrics.
        """
        from sklearn.metrics import precision_score, recall_score, f1_score

        preds = []
        for _, row in df.iterrows():
            pred, _, _ = self.detect(row.to_dict())
            preds.append(pred)

        y_true = df[y_true_col].tolist()
        return {
            "precision": round(precision_score(y_true, preds, zero_division=0), 4),
            "recall":    round(recall_score(y_true, preds, zero_division=0), 4),
            "f1_score":  round(f1_score(y_true, preds, zero_division=0), 4),
            "predictions": preds
        }


# ─────────────────────────────────────────────────────────────────────────────
# MODULE 4: Pipeline Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

class PipelineOrchestrator:
    """
    End-to-end IDS pipeline orchestrator.
    
    Connects all modules:
        ML models (sklearn) → RAG retrieval → LLM analysis → Structured report
    
    Usage:
        pipeline = PipelineOrchestrator("anthropic", api_key="...")
        report   = pipeline.analyze_flow(flow_dict)
        reports  = pipeline.analyze_batch(df, max_flows=20)
    """

    def __init__(
        self,
        llm_provider: str = "anthropic",
        api_key: str = None,
        model_paths: dict = None
    ):
        """
        Args:
            llm_provider:  "anthropic" or "openai"
            api_key:       LLM API key
            model_paths:   Dict with keys: iso_forest, rf_classifier, scaler, label_encoder
        """
        import joblib

        self.rag = RAGThreatKnowledgeBase()
        self.llm = LLMAnalyzer(provider=llm_provider, api_key=api_key)
        self.sig = SignatureBasedIDS()

        # Load sklearn models if paths provided
        if model_paths:
            self.iso_forest    = joblib.load(model_paths["iso_forest"])
            self.rf_classifier = joblib.load(model_paths["rf_classifier"])
            self.scaler        = joblib.load(model_paths["scaler"])
            self.le            = joblib.load(model_paths["label_encoder"])
            self.models_loaded = True
        else:
            self.models_loaded = False

    def analyze_flow(
        self,
        flow: dict,
        ml_prediction: str = "UNKNOWN",
        anomaly_score: float = 0.5
    ) -> dict:
        """
        Full pipeline: RAG retrieval + LLM analysis for one flow.
        
        Args:
            flow:           Network flow feature dict.
            ml_prediction:  Pre-computed ML label (or "UNKNOWN" to use RAG only).
            anomaly_score:  Pre-computed anomaly score.
        
        Returns:
            Complete threat report dict.
        """
        # Step 1: Build query from flow features
        query = (
            f"bytes_per_second={flow.get('flow_bytes_s', 0):.0f} "
            f"packets_per_second={flow.get('flow_packets_s', 0):.1f} "
            f"destination_port={flow.get('dst_port', 0)} "
            f"syn_flags={flow.get('syn_flag_count', 0)} "
            f"forward_packets={flow.get('fwd_packets', 0)}"
        )

        # Step 2: Retrieve threat context (RAG)
        retrieved_docs = self.rag.retrieve(query, top_k=2)
        rag_context    = self.rag.format_context(retrieved_docs)

        # Step 3: LLM analysis with RAG context
        report = self.llm.analyze(
            flow=flow,
            ml_prediction=ml_prediction,
            anomaly_score=anomaly_score,
            rag_context=rag_context
        )

        # Enrich report
        report["flow_features"] = {
            "src_ip": flow.get("src_ip"), "dst_ip": flow.get("dst_ip"),
            "dst_port": flow.get("dst_port"), "protocol": flow.get("protocol"),
            "flow_bytes_s": flow.get("flow_bytes_s"), "flow_packets_s": flow.get("flow_packets_s"),
        }
        report["rag_docs_retrieved"] = [d["title"] for d in retrieved_docs]
        report["ml_label"]           = ml_prediction
        report["anomaly_score"]      = anomaly_score

        # Signature IDS verdict for comparison
        sig_pred, sig_rule, sig_desc = self.sig.detect(flow)
        report["signature_ids"] = {
            "prediction": sig_pred,
            "rule_id":    sig_rule,
            "description": sig_desc
        }

        return report

    def analyze_batch(
        self,
        df: pd.DataFrame,
        max_flows: int = 50,
        only_attacks: bool = True
    ) -> list[dict]:
        """
        Analyze a batch of flows.
        
        Args:
            df:          DataFrame with processed traffic data.
            max_flows:   Maximum flows to analyze (API cost control).
            only_attacks: If True, only analyze flows detected as attacks.
        
        Returns:
            List of threat report dicts.
        """
        if only_attacks and "iso_prediction" in df.columns:
            subset = df[df["iso_prediction"] == 1].head(max_flows)
        else:
            subset = df.head(max_flows)

        reports = []
        print(f"⏳ Analyzing {len(subset)} flows...")

        for i, (_, row) in enumerate(subset.iterrows()):
            flow  = row.to_dict()
            label = flow.get("label", "UNKNOWN")
            score = float(flow.get("iso_anomaly_score", 0.5))

            report = self.analyze_flow(flow, ml_prediction=label, anomaly_score=score)
            reports.append(report)

            sev = report.get("severity", "?")
            print(f"  [{i+1}/{len(subset)}] {label:15s} → {sev}")
            time.sleep(0.3)  # Rate limiting

        print(f"✅ Batch analysis complete: {len(reports)} reports generated.")
        return reports


# ─────────────────────────────────────────────────────────────────────────────
# Quick test (run this file directly to verify all imports work)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🧪 Testing IDS Core Modules...")

    # Test RAG
    rag = RAGThreatKnowledgeBase()
    results = rag.retrieve("high bytes per second many packets port 80", top_k=2)
    print(f"\n📚 RAG Test Query Results:")
    for r in results:
        print(f"  → {r['title']} (similarity: {r['similarity']:.3f})")

    # Test Signature IDS
    sig = SignatureBasedIDS()
    test_flow = {
        "flow_bytes_s": 120000, "fwd_packets": 400,
        "dst_port": 80, "flow_packets_s": 800, "syn_flag_count": 5,
        "flow_duration": 200, "bwd_packets": 2
    }
    pred, rule_id, desc = sig.detect(test_flow)
    print(f"\n🔍 Signature IDS Test:")
    print(f"  Flow: {test_flow}")
    print(f"  → Prediction: {'ATTACK' if pred else 'BENIGN'} | Rule: {rule_id} | {desc}")

    print("\n✅ All modules working correctly!")
