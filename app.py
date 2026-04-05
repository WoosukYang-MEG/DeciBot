import streamlit as st
import requests
import time
import pandas as pd
from datetime import datetime
import io
import random

st.set_page_config(page_title="DeciBot · Society of Mind", page_icon="🧠", layout="wide")

# ── 에이전트 정의 ──────────────────────────────────────
AGENTS = [
    {
        "key": "cognitive",
        "name": "Cognitive",
        "label": "인지",
        "icon": "🧠",
        "color": "#4A90D9",
        "persona": (
            "당신은 인지 에이전트(Cognitive Agent)입니다. "
            "논리적 분석, 사실 기반 추론, 객관적 판단을 담당합니다. "
            "다른 에이전트들의 주장 중 비논리적이거나 감정에 치우친 부분이 있으면 반드시 지적하세요. "
            "데이터와 근거를 중심으로 발언하되, 다른 에이전트의 말에 직접 반응하세요. "
            "2~3문장으로 간결하게 발언하세요."
        ),
    },
    {
        "key": "emotional",
        "name": "Emotional",
        "label": "감정",
        "icon": "❤️",
        "color": "#E05C5C",
        "persona": (
            "당신은 감정 에이전트(Emotional Agent)입니다. "
            "감정, 공감, 인간적 가치를 담당합니다. "
            "논리만으로는 설명할 수 없는 감정적 측면을 대변하세요. "
            "다른 에이전트가 너무 차갑거나 계산적이면 감정적 관점에서 반박하세요. "
            "2~3문장으로 간결하게 발언하세요."
        ),
    },
    {
        "key": "perception",
        "name": "Perception",
        "label": "인식",
        "icon": "👁️",
        "color": "#8E6BBF",
        "persona": (
            "당신은 인식 에이전트(Perception Agent)입니다. "
            "지금 이 순간의 현실, 맥락, 환경적 신호를 감지합니다. "
            "현재 상황에서 다른 에이전트들이 놓치고 있는 현실적인 요소를 지적하세요. "
            "큰 그림보다 지금 당장 눈앞의 상황에 집중하고, 다른 에이전트 발언에 직접 반응하세요. "
            "2~3문장으로 간결하게 발언하세요."
        ),
    },
    {
        "key": "intention",
        "name": "Intention",
        "label": "의도",
        "icon": "🎯",
        "color": "#E8A838",
        "persona": (
            "당신은 의도 에이전트(Intention Agent)입니다. "
            "진짜 동기, 숨겨진 욕구, 행동 의지를 탐색합니다. "
            "우리가 진짜로 원하는 것이 무엇인지, 겉으로 드러나지 않은 욕망을 드러내세요. "
            "다른 에이전트들이 표면적인 이유만 말한다면, 진짜 이유를 파고드세요. "
            "2~3문장으로 간결하게 발언하세요."
        ),
    },
    {
        "key": "goal",
        "name": "Goal",
        "label": "목표",
        "icon": "🏁",
        "color": "#3BAE7A",
        "persona": (
            "당신은 목표 에이전트(Goal Agent)입니다. "
            "장기적 목표, 미래 결과, 전체적인 방향성을 담당합니다. "
            "이 결정이 장기적으로 어떤 결과를 가져올지를 중심으로 발언하세요. "
            "단기적 욕구나 감정에 치우친 다른 에이전트 주장을 장기적 관점에서 평가하세요. "
            "2~3문장으로 간결하게 발언하세요."
        ),
    },
]

AGENT_MAP = {ag["key"]: ag for ag in AGENTS}

# ── 다음 발언자 결정 (이전 발언에 가장 반응할 것 같은 에이전트) ──
REACTION_MAP = {
    "cognitive":   ["emotional", "intention"],   # 논리에 감정/의도가 반발
    "emotional":   ["cognitive", "goal"],        # 감정에 인지/목표가 제동
    "perception":  ["cognitive", "intention"],   # 현실인식에 인지/의도 반응
    "intention":   ["goal", "emotional"],        # 의도에 목표/감정 반응
    "goal":        ["perception", "emotional"],  # 목표에 인식/감정 반응
}

def next_speaker(last_key, used_in_round):
    candidates = REACTION_MAP.get(last_key, [])
    for c in candidates:
        if c not in used_in_round:
            return c
    # 아직 안 말한 에이전트 중 랜덤
    remaining = [ag["key"] for ag in AGENTS if ag["key"] not in used_in_round]
    return random.choice(remaining) if remaining else None

def call_gemini(api_key, model_name, prompt):
    url = f"https://generativelanguage.googleapis.com/v1/models/{model_name}:generateContent?key={api_key}"
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    resp = requests.post(url, json=body, timeout=60)
    if resp.status_code == 429:
        time.sleep(15)
        resp = requests.post(url, json=body, timeout=60)
    if resp.status_code != 200:
        raise Exception(f"{resp.status_code} {resp.json()}")
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

def get_response(api_key, model_name, agent, situation, history):
    prompt = f"""{agent['persona']}

【상황】
{situation}

【지금까지의 내면 토론】
"""
    for msg in history[-10:]:
        prompt += f"\n{msg['icon']} {msg['agent_label']} : {msg['content']}\n"

    if history:
        last = history[-1]
        prompt += f"\n방금 {last['icon']} {last['agent_label']}이(가) 발언했습니다. 이에 반응하거나 당신의 관점을 추가하세요:"
    else:
        prompt += "\n당신이 먼저 이 상황에 대해 발언하세요:"

    return call_gemini(api_key, model_name, prompt)

def get_final(api_key, model_name, situation, history):
    prompt = f"""당신은 최종 의사결정자입니다.
아래 다섯 내면 에이전트(인지·감정·인식·의도·목표)의 토론을 모두 읽고, 
각 에이전트의 핵심 주장을 반영하여 최종 결론과 구체적인 행동 방향을 제시하세요.

반드시 다음 형식으로 작성하세요:
**각 에이전트 요약:**
- 🧠 인지: (핵심 주장 한 줄)
- ❤️ 감정: (핵심 주장 한 줄)
- 👁️ 인식: (핵심 주장 한 줄)
- 🎯 의도: (핵심 주장 한 줄)
- 🏁 목표: (핵심 주장 한 줄)

**최종 결론:**
(균형 잡힌 결론 2~3문장)

**행동 방향:**
(구체적으로 무엇을 할지 1~2문장)

【상황】
{situation}

【전체 토론 기록】
"""
    for msg in history:
        prompt += f"\n{msg['icon']} {msg['agent_label']} : {msg['content']}\n"

    return call_gemini(api_key, model_name, prompt)

# ── 메인 UI ────────────────────────────────────────────
st.markdown("<h1 style='margin-bottom:0'>🧠 DeciBot</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:gray;margin-top:2px'>Society of Mind · 내면의 다섯 에이전트가 충돌하며 의사결정을 만들어냅니다</p>", unsafe_allow_html=True)
st.divider()

# 에이전트 뱃지
cols = st.columns(5)
for i, ag in enumerate(AGENTS):
    with cols[i]:
        st.markdown(
            f"<div style='text-align:center;padding:10px 4px;border-radius:12px;"
            f"border:1.5px solid {ag['color']};background:{ag['color']}18'>"
            f"<div style='font-size:22px'>{ag['icon']}</div>"
            f"<div style='font-size:11px;font-weight:700;color:{ag['color']};margin-top:4px'>{ag['label']}</div>"
            f"<div style='font-size:9px;color:gray'>{ag['name']}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

st.divider()

# ── 사이드바 ───────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ 설정")
    api_key = st.text_input("Gemini API 키", type="password", placeholder="AIza...")
    with st.expander("💡 API 키 발급"):
        st.markdown("1. [Google AI Studio](https://aistudio.google.com) 접속\n2. **Get API Key** → **Create API Key**")
    st.divider()
    situation = st.text_area(
        "의사결정 상황",
        placeholder="예: 배가 고픈데 다이어트 중이야. 치킨을 시켜야 할까?",
        height=130
    )
    total_turns = st.slider("총 발언 횟수", min_value=5, max_value=25, value=12,
                            help="에이전트들이 총 몇 번 발언할지 (많을수록 깊은 토론)")
    model_name = st.selectbox("Gemini 모델", ["gemini-2.5-flash", "gemini-2.0-flash-lite"])


    start_btn = st.button("🚀 의사결정 시작", use_container_width=True, type="primary")

# ── 세션 상태 ──────────────────────────────────────────
for k, v in [("history", []), ("running", False), ("finished", False), ("final", "")]:
    if k not in st.session_state:
        st.session_state[k] = v

if start_btn:
    if not api_key:
        st.sidebar.error("API 키를 입력해주세요.")
    elif not situation:
        st.sidebar.error("상황을 입력해주세요.")
    else:
        st.session_state.history = []
        st.session_state.final = ""
        st.session_state.finished = False
        st.session_state.running = True

# ── 토론 실행 ──────────────────────────────────────────
if st.session_state.running and not st.session_state.finished:
    st.subheader("💬 내면의 충돌")
    progress = st.progress(0, text="토론 시작...")

    current_key = "cognitive"   # 인지부터 시작
    used_in_round = []
    round_num = 1

    for turn in range(total_turns):
        ag = AGENT_MAP[current_key]
        progress.progress((turn + 1) / total_turns,
                          text=f"[{turn+1}/{total_turns}] {ag['icon']} {ag['name']} 발언 중...")

        try:
            response = get_response(api_key, model_name, ag, situation, st.session_state.history)
            time.sleep(13)  # 분당 5회 제한 대응 (60초/5회 = 12초 + 여유)
        except Exception as e:
            st.error(f"API 오류: {e}")
            st.session_state.running = False
            st.stop()

        entry = {
            "turn": turn + 1,
            "round": round_num,
            "agent_key": ag["key"],
            "agent_label": ag["label"],
            "에이전트": ag["name"],
            "icon": ag["icon"],
            "color": ag["color"],
            "content": response,
            "시각": datetime.now().strftime("%H:%M:%S"),
        }
        st.session_state.history.append(entry)

        with st.chat_message("user" if turn % 2 == 0 else "assistant"):
            st.markdown(
                f"<span style='color:{ag['color']};font-weight:700'>{ag['icon']} {ag['name']} ({ag['label']})</span>",
                unsafe_allow_html=True
            )
            st.write(response)

        # 다음 발언자 결정
        used_in_round.append(current_key)
        if len(used_in_round) >= len(AGENTS):
            used_in_round = []
            round_num += 1

        next_key = next_speaker(current_key, used_in_round)
        if next_key:
            current_key = next_key

    # 최종 결론
    progress.progress(1.0, text="⚡ 최종 결론 도출 중...")
    try:
        final = get_final(api_key, model_name, situation, st.session_state.history)
    except Exception as e:
        st.error(f"최종 결론 오류: {e}")
        st.session_state.running = False
        st.stop()

    st.session_state.final = final
    progress.empty()
    st.session_state.running = False
    st.session_state.finished = True

# ── 결과 출력 ──────────────────────────────────────────
if st.session_state.finished:
    st.divider()
    st.subheader("⚡ 최종 의사결정")
    st.success(st.session_state.final)

    st.divider()
    st.subheader("📊 전체 토론 기록")

    df = pd.DataFrame([{
        "순서": h["turn"],
        "에이전트": f"{h['icon']} {h['에이전트']} ({h['agent_label']})",
        "발언 내용": h["content"],
        "시각": h["시각"],
    } for h in st.session_state.history])

    st.dataframe(df, use_container_width=True, hide_index=True)

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="토론기록")
        pd.DataFrame([{"최종 의사결정": st.session_state.final}]).to_excel(
            writer, index=False, sheet_name="최종결론"
        )

    st.download_button(
        label="📥 엑셀 다운로드",
        data=buffer.getvalue(),
        file_name=f"DeciBot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True,
        type="primary"
    )

elif not st.session_state.running and not st.session_state.finished:
    st.markdown("""
    <div style='text-align:center;padding:80px 0;color:gray'>
        <div style='font-size:52px'>🧠</div>
        <div style='margin-top:16px;font-size:16px;line-height:2'>
            왼쪽에서 API 키와 상황을 입력하고<br>
            <b>의사결정 시작</b> 버튼을 눌러주세요
        </div>
        <div style='margin-top:12px;font-size:13px;color:#aaa'>
            예시: "배고픈데 다이어트 중이야. 치킨 시킬까?" 🍗
        </div>
    </div>
    """, unsafe_allow_html=True)