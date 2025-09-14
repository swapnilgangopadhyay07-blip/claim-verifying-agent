import gradio as gr
import os
import json
import re
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import google.generativeai as genai
from serpapi import GoogleSearch


GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY environment variable")
if not SERPAPI_API_KEY:
    raise RuntimeError("Missing SERPAPI_API_KEY environment variable")

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

# ==========================
# Data structures
# ==========================
@dataclass
class SearchResult:
    title: str
    link: str
    snippet: str
    source: str
    date: str = ""

# ==========================
# Tools
# ==========================
def google_search(query: str, num: int = 8) -> List[Dict[str, Any]]:
    """Search the web using SerpAPI Google Search."""
    params = {
        "engine": "google",
        "q": query,
        "num": num,
        "api_key": SERPAPI_API_KEY,
    }

    search = GoogleSearch(params)
    results = search.get_dict().get("organic_results", [])
    output = []
    for r in results:
        output.append(asdict(SearchResult(
            title=r.get("title", ""),
            link=r.get("link", ""),
            snippet=r.get("snippet", ""),
            source=r.get("displayed_link", ""),
            date=r.get("date", ""),
        )))
    return output

def score_sources(claim: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Heuristic credibility scoring with fact-check signals."""
    if not results:
        return {"score": 0, "verdict": "Insufficient evidence", "top_sources": []}

    authoritative = (".gov", ".edu", "who.int", "nih.gov", "cdc.gov", "reuters.com",
                     "bbc.com", "nature.com", "science.org", "nytimes.com", "theguardian.com")
    factcheck = ("snopes.com", "politifact.com", "factcheck.org", "reuters.com/fact-check")

    def domain_quality(url):
        url = url.lower()
        if any(d in url for d in factcheck):
            return 1.0
        if any(d in url for d in authoritative):
            return 0.9
        if ".gov" in url or ".edu" in url:
            return 0.85
        return 0.6

    def recency(date_str):
        try:
            m = re.search(r"(20\d{2})", date_str)
            year = int(m.group(1)) if m else None
            if not year:
                return 0.6
            age = time.gmtime().tm_year - year
            if age <= 1:
                return 1.0
            elif age <= 3:
                return 0.8
            else:
                return 0.4
        except:
            return 0.6

    def relevance(claim, snippet, title):
        claim_terms = set(re.findall(r"\w+", claim.lower()))
        text = (snippet + " " + title).lower()
        match_count = sum(1 for word in claim_terms if word in text)
        return min(1.0, match_count / max(3, len(claim_terms) * 0.5))

    scores = []
    breakdown = []
    factcheck_flag = False
    false_flag = False

    for r in results:
        dq = domain_quality(r["link"])
        rc = recency(r.get("date", ""))
        rel = relevance(claim, r.get("snippet", ""), r.get("title", ""))

        score = 0.4 * dq + 0.2 * rc + 0.2 * rel

        if any(fc in r["link"].lower() for fc in factcheck):
            factcheck_flag = True
            score += 0.3

        if any(word in r.get("snippet", "").lower() for word in ["false", "debunked", "misleading"]):
            false_flag = True

        score = min(score, 1.0)
        scores.append(score)
        breakdown.append({**r, "score": round(score * 100, 2)})

    avg = sum(scores) / len(scores)

    if factcheck_flag and false_flag:
        avg -= 0.5
        avg = max(0, avg)

    final = int(round(avg * 100))

    if final >= 70:
        verdict = "Likely True"
    elif final <= 35:
        verdict = "Likely False"
    else:
        verdict = "Uncertain"

    return {
        "score": final,
        "verdict": verdict,
        "top_sources": breakdown[:5],
    }

def gemini_reasoning(claim: str, results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Ask Gemini to classify the claim with natural text explanation."""
    sources_text = json.dumps(results[:5], indent=2)

    prompt = f"""
    You are a claim verification assistant.
    Claim: {claim}
    Evidence sources: {sources_text}

    Based only on these sources, answer in plain English:
    - Is the claim supported, refuted, or unclear?
    - Give a short explanation (2-3 sentences).
    """

    resp = model.generate_content(prompt)
    text = resp.text.strip()
    text_lower = text.lower()

    verdict = "Uncertain"
    if any(word in text_lower for word in ["refute", "refuted", "false", "misleading", "debunked"]):
        verdict = "Likely False"
    elif any(word in text_lower for word in ["support", "supported", "true", "confirmed"]):
        verdict = "Likely True"

    return {
        "verdict": verdict,
        "explanation": text
    }

def determine_confidence(heuristic_score: int, heuristic_verdict: str, gemini_verdict: str) -> str:
    """Decide confidence level based on agreement between heuristic & Gemini."""
    if heuristic_verdict == gemini_verdict:
        if heuristic_score >= 75 or heuristic_score <= 25:
            return "High"
        else:
            return "Medium"
    else:
        return "Low"

def verify_claim(claim: str) -> Dict[str, Any]:
    """Run full verification with heuristic + Gemini reasoning."""
    if not claim.strip():
        return {
            "claim": "",
            "credibility_score": 0,
            "verdict": "No claim provided",
            "confidence": "Low",
            "rationale": "Please enter a claim to verify.",
            "top_sources": [],
        }
    
    try:
        search_results = google_search(claim, num=8)
        heuristic = score_sources(claim, search_results)
        gemini = gemini_reasoning(claim, search_results)

        final_score = heuristic["score"]
        verdict = heuristic["verdict"]

        if gemini["verdict"] == "Likely True":
            final_score = min(100, int(0.7 * heuristic["score"] + 0.3 * 100))
            verdict = "Likely True"
        elif gemini["verdict"] == "Likely False":
            final_score = max(0, int(0.7 * heuristic["score"] + 0.3 * 0))
            verdict = "Likely False"

        confidence = determine_confidence(heuristic["score"], heuristic["verdict"], gemini["verdict"])

        return {
            "claim": claim,
            "credibility_score": final_score,
            "verdict": verdict,
            "confidence": confidence,
            "rationale": gemini["explanation"],
            "top_sources": heuristic["top_sources"],
        }
    except Exception as e:
        return {
            "claim": claim,
            "credibility_score": 0,
            "verdict": "Error",
            "confidence": "Low",
            "rationale": f"An error occurred during verification: {str(e)}",
            "top_sources": [],
        }

# ==========================
# Gradio Interface
# ==========================
def format_result(claim):
    """Format the verification result for Gradio display."""
    if not claim.strip():
        return "Please enter a claim to verify.", "", "", "", ""
    
    result = verify_claim(claim)
    
    # Format main result
    score = result["credibility_score"]
    verdict = result["verdict"]
    confidence = result["confidence"]
    
    # Color coding for verdict
    if verdict == "Likely True":
        verdict_color = "🟢"
    elif verdict == "Likely False":
        verdict_color = "🔴"
    else:
        verdict_color = "🟡"
    
    main_result = f"""
    ## {verdict_color} Verification Result
    
    **Claim:** {result["claim"]}
    
    **Verdict:** {verdict}  
    **Credibility Score:** {score}/100  
    **Confidence:** {confidence}
    """
    
    # Format rationale
    rationale = f"""
    ## 🧠 AI Analysis
    
    {result["rationale"]}
    """
    
    # Format sources
    sources_text = "## 📚 Top Sources\n\n"
    for i, source in enumerate(result["top_sources"][:3], 1):
        sources_text += f"""
        **{i}. {source.get('title', 'N/A')}**  
        *Source:* {source.get('source', 'N/A')}  
        *Snippet:* {source.get('snippet', 'N/A')}  
        *Link:* [{source.get('link', 'N/A')}]({source.get('link', 'N/A')})  
        *Credibility Score:* {source.get('score', 0)}/100
        
        ---
        """
    
    return main_result, rationale, sources_text

def create_interface():
    with gr.Blocks(
        title="ClaimCheck AI - Professional Fact Verification",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            padding: 2rem 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        """
    ) as demo:
        
        gr.HTML("""
        <div class="main-header">
            <h1>🔍 ClaimCheck AI</h1>
            <p>Professional AI-powered fact verification system</p>
            <p><em>Verify claims with advanced search and AI analysis</em></p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                claim_input = gr.Textbox(
                    label="Enter Claim to Verify",
                    placeholder="e.g., 'The COVID-19 vaccine contains microchips'",
                    lines=3,
                    max_lines=5
                )
                verify_btn = gr.Button(
                    "🔍 Verify Claim", 
                    variant="primary", 
                    size="lg"
                )
                
                gr.Markdown("""
                ### How it works:
                1. **Search** - Searches authoritative sources
                2. **Analyze** - AI evaluates evidence quality  
                3. **Score** - Combines heuristic and AI analysis
                4. **Report** - Provides detailed verification results
                
                ### Tips:
                - Be specific with your claims
                - Check multiple sources for important decisions
                - Consider the confidence level in results
                """)
        
        with gr.Row():
            with gr.Column():
                result_output = gr.Markdown(label="Verification Result")
                
        with gr.Row():
            with gr.Column():
                rationale_output = gr.Markdown(label="AI Analysis")
                
        with gr.Row():
            with gr.Column():
                sources_output = gr.Markdown(label="Sources")
        
        # Examples
        gr.Examples(
            examples=[
                ["The COVID-19 vaccine contains microchips"],
                ["Climate change is caused by human activities"],
                ["The Great Wall of China is visible from space"],
                ["Drinking 8 glasses of water daily is necessary for health"],
                ["5G networks cause cancer"]
            ],
            inputs=claim_input,
            label="Try these example claims:"
        )
        
        verify_btn.click(
            fn=format_result,
            inputs=[claim_input],
            outputs=[result_output, rationale_output, sources_output]
        )
        
        claim_input.submit(
            fn=format_result,
            inputs=[claim_input],
            outputs=[result_output, rationale_output, sources_output]
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )