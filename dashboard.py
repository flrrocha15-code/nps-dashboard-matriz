import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
import re
import anthropic
from groq import Groq

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard NPS – Matriz Educação",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
NPS_TARGET = 50

UNIT_FILES = {
    "Bangu":               "NPS - matriz-bangu.xlsx",
    "Campo Grande":        "NPS - matriz-campogrande.xlsx",
    "Caxias":              "NPS - matriz-caxias.xlsx",
    "Madureira":           "NPS - matriz-madureira.xlsx",
    "Nova Iguaçu":         "NPS - matriz-novaiguacu.xlsx",
    "Retiro dos Artistas": "NPS - matriz-retirodosartistas.xlsx",
    "Rocha Miranda":       "NPS - matriz-rochamiranda.xlsx",
    "São João de Meriti":  "NPS - matriz-saojoaodemeriti.xlsx",
    "Taquara":             "NPS - matriz-taquara.xlsx",
    "Tijuca":              "NPS - matriz-tijuca.xlsx",
}

ATTR_COLS = {
    "Qualidade do Ensino":         10,
    "Acolhimento Emocional":       11,
    "Recursos Pedagógicos":        12,
    "Atendimento ao Público":      13,
    "Canais de Comunicação":       14,
    "Gestão Escolar":              15,
    "Higiene e Conservação":       16,
    "Alimentação":                 17,
    "Conforto e Segurança":        18,
}

CATEGORY_MAP = {
    "Qualidade do Ensino":    "Pedagógico",
    "Acolhimento Emocional":  "Pedagógico",
    "Recursos Pedagógicos":   "Pedagógico",
    "Atendimento ao Público": "Administrativo",
    "Canais de Comunicação":  "Administrativo",
    "Gestão Escolar":         "Administrativo",
    "Higiene e Conservação":  "Infraestrutura",
    "Alimentação":            "Infraestrutura",
    "Conforto e Segurança":   "Infraestrutura",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def parse_score(val):
    """Extract numeric score from mixed text/numeric NPS values."""
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return int(val)
    m = re.match(r"^(\d+)", str(val).strip())
    return int(m.group(1)) if m else None


def classify_nps(score):
    if score is None:
        return None
    if score >= 9:
        return "Promotor"
    if score >= 7:
        return "Neutro"
    return "Detrator"


def compute_nps(scores):
    total = len(scores)
    if total == 0:
        return 0, 0, 0, 0
    promoters  = sum(1 for s in scores if s >= 9)
    detractors = sum(1 for s in scores if s <= 6)
    nps = round((promoters / total - detractors / total) * 100)
    return nps, promoters, detractors, total - promoters - detractors


@st.cache_data(show_spinner=False)
def load_all_data():
    frames = {}
    for unit, fname in UNIT_FILES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            continue
        df = pd.read_excel(path, sheet_name="CSAT 1", header=0)
        df["_unit"]     = unit
        df["_nps_raw"]  = df.iloc[:, 8].apply(parse_score)
        df["_segment"]  = df["_nps_raw"].apply(classify_nps)
        df["_feedback"] = df.iloc[:, 9].fillna("")
        df["_comments"] = df.iloc[:, 19].fillna("")
        df["_tipo"]     = df.iloc[:, 20].fillna("Não informado")
        for attr, col_idx in ATTR_COLS.items():
            df[f"_attr_{attr}"] = df.iloc[:, col_idx].apply(parse_score)
        frames[unit] = df
    return frames


def build_summary_df(frames):
    rows = []
    for unit, df in frames.items():
        scores = df["_nps_raw"].dropna().tolist()
        nps, prom, detr, neut = compute_nps(scores)
        rows.append({
            "Unidade":     unit,
            "NPS":         nps,
            "Promotores":  prom,
            "Neutros":     neut,
            "Detratores":  detr,
            "Total":       len(scores),
            "% Promotores":  round(prom / len(scores) * 100, 1) if scores else 0,
            "% Detratores":  round(detr / len(scores) * 100, 1) if scores else 0,
        })
    return pd.DataFrame(rows).sort_values("NPS", ascending=False).reset_index(drop=True)


def nps_color(nps):
    if nps >= 75:  return "#2ecc71"
    if nps >= 50:  return "#f39c12"
    if nps >= 0:   return "#e67e22"
    return "#e74c3c"


def get_feedbacks_text(df):
    texts = []
    for _, row in df.iterrows():
        fb = str(row["_feedback"]).strip()
        cm = str(row["_comments"]).strip()
        seg = row["_segment"] or "?"
        if fb or cm:
            texts.append(f"[{seg}] {fb} {cm}".strip())
    return texts


# ── AI calls ──────────────────────────────────────────────────────────────────
def _call_groq(prompt, api_key, max_tokens=2048):
    """Call Groq (free tier – Llama 3)."""
    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content


def _call_anthropic(prompt, api_key, max_tokens=2048):
    """Call Anthropic Claude."""
    client = anthropic.Anthropic(api_key=api_key)
    with client.messages.stream(
        model="claude-opus-4-6",
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        return stream.get_final_text()


def call_ai(prompt, provider, api_key, max_tokens=2048):
    if provider == "Groq – Llama 3 (Grátis)":
        return _call_groq(prompt, api_key, max_tokens)
    return _call_anthropic(prompt, api_key, max_tokens)


def ai_summarize_unit(unit, nps, feedbacks, provider, api_key):
    sample = feedbacks[:80]
    prompt = f"""Você é um analista de NPS de uma rede de colégios chamada Matriz Educação.

Unidade: {unit}
NPS atual: {nps}

A seguir estão comentários de alunos e responsáveis (formato: [Segmento] comentário):

{chr(10).join(sample)}

Faça um resumo estruturado com:
1. **Pontos Positivos** (principais elogios, máximo 5 bullets)
2. **Pontos de Melhoria** (principais críticas, máximo 5 bullets)
3. **Temas Recorrentes** (padrões que aparecem em múltiplas respostas)

Seja direto e objetivo. Use linguagem profissional em português."""

    return call_ai(prompt, provider, api_key, max_tokens=1024)


def ai_action_plan(unit, nps, summary, attr_avgs, provider, api_key):
    attr_text = "\n".join(
        f"- {k}: {v:.1f}/6" for k, v in attr_avgs.items() if v is not None
    )
    prompt = f"""Você é um consultor especializado em gestão escolar e NPS para a rede Matriz Educação.

Unidade: {unit}
NPS atual: {nps} (meta: {NPS_TARGET})
Gap para a meta: {NPS_TARGET - nps} pontos

Avaliação média por atributo (escala 1-6):
{attr_text}

Resumo das percepções dos alunos/responsáveis:
{summary}

Crie um **Plano de Ação Detalhado** para elevar o NPS de {nps} para pelo menos {NPS_TARGET} nos próximos 90 dias.

Estruture o plano em:
## Diagnóstico Rápido
(2-3 frases sobre o estado atual)

## Ações Prioritárias (Quick Wins – primeiros 30 dias)
(3 ações com maior impacto no NPS, com responsável sugerido e indicador de sucesso)

## Ações Estruturais (30-90 dias)
(3-5 ações de médio prazo que atacam as causas raiz)

## Métricas de Acompanhamento
(KPIs específicos para monitorar o progresso)

Seja específico, prático e orientado a resultados. Use linguagem profissional em português."""

    return call_ai(prompt, provider, api_key, max_tokens=2048)


# ── CSS Theme ─────────────────────────────────────────────────────────────────
MATRIZ_GREEN  = "#81d742"
MATRIZ_BLUE   = "#1c61ac"
MATRIZ_DARK   = "#0d3d6b"

def inject_css():
    st.markdown(f"""
    <style>

    /* ══════════════════════════════════════════
       SIDEBAR
    ══════════════════════════════════════════ */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, {MATRIZ_DARK} 0%, {MATRIZ_BLUE} 100%);
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff !important;
    }}
    [data-testid="stSidebar"] hr {{
        border-color: rgba(255,255,255,0.2) !important;
    }}
    [data-testid="stSidebar"] .stExpander {{
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        border-radius: 8px;
    }}
    [data-testid="stSidebar"] input[type="password"] {{
        background: rgba(255,255,255,0.12) !important;
        color: #ffffff !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 6px;
    }}
    [data-testid="stSidebar"] .stSelectbox > div > div {{
        background: rgba(255,255,255,0.12) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        color: #ffffff !important;
    }}

    /* Nav pills */
    div[data-testid="stSidebar"] .stRadio > div {{ gap: 6px; }}
    div[data-testid="stSidebar"] .stRadio > div > label {{
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.15);
        border-radius: 8px;
        padding: 12px 14px;
        width: 100%;
        transition: background 0.2s;
        font-weight: 500;
        font-size: 1rem;
    }}
    div[data-testid="stSidebar"] .stRadio > div > label:hover {{
        background: rgba(129,215,66,0.25);
        border-color: {MATRIZ_GREEN};
    }}
    div[data-testid="stSidebar"] .stRadio > div > label[data-checked="true"] {{
        background: {MATRIZ_GREEN} !important;
        border-color: {MATRIZ_GREEN} !important;
        color: #0d3d6b !important;
        font-weight: 700;
    }}

    /* ══════════════════════════════════════════
       MAIN CONTAINER
    ══════════════════════════════════════════ */
    .main .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }}
    .main-header {{
        border-left: 5px solid {MATRIZ_GREEN};
        padding-left: 14px;
        margin-bottom: 8px;
    }}
    .main-header h1 {{
        margin: 0;
        color: {MATRIZ_DARK};
        font-size: 1.6rem;
    }}

    /* ══════════════════════════════════════════
       METRIC CARDS
    ══════════════════════════════════════════ */
    [data-testid="stMetric"] {{
        background: #f8fff3;
        border: 1px solid {MATRIZ_GREEN};
        border-left: 4px solid {MATRIZ_GREEN};
        border-radius: 10px;
        padding: 10px 14px !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {MATRIZ_BLUE} !important;
        font-weight: 600;
        font-size: 0.8rem !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}
    [data-testid="stMetricValue"] {{
        color: {MATRIZ_DARK} !important;
        font-size: 1.6rem !important;
        font-weight: 700;
    }}

    /* ══════════════════════════════════════════
       TYPOGRAPHY
    ══════════════════════════════════════════ */
    h2, h3 {{ color: {MATRIZ_DARK} !important; }}
    h2::after {{
        content: "";
        display: block;
        width: 48px;
        height: 3px;
        background: {MATRIZ_GREEN};
        margin-top: 4px;
        border-radius: 2px;
    }}

    /* ══════════════════════════════════════════
       TABS
    ══════════════════════════════════════════ */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 4px;
        border-bottom: 2px solid {MATRIZ_GREEN};
        flex-wrap: wrap;
    }}
    .stTabs [data-baseweb="tab"] {{
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        font-size: 0.9rem;
        padding: 8px 12px;
    }}
    .stTabs [aria-selected="true"] {{
        background: {MATRIZ_GREEN} !important;
        color: {MATRIZ_DARK} !important;
    }}

    /* ══════════════════════════════════════════
       BUTTONS
    ══════════════════════════════════════════ */
    .stButton > button {{
        background: {MATRIZ_GREEN};
        color: {MATRIZ_DARK};
        font-weight: 700;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-size: 1rem;
        width: 100%;
        transition: opacity 0.2s;
    }}
    .stButton > button:hover {{
        opacity: 0.85;
        color: {MATRIZ_DARK};
    }}

    /* ══════════════════════════════════════════
       SIDEBAR FOOTER / LOGO
    ══════════════════════════════════════════ */
    .sidebar-footer {{
        position: fixed;
        bottom: 18px;
        font-size: 0.72rem;
        color: rgba(255,255,255,0.45) !important;
        text-align: center;
        width: 220px;
    }}
    .logo-area {{
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 20px 10px 10px 10px;
    }}
    .logo-subtitle {{
        color: {MATRIZ_GREEN};
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }}

    /* ══════════════════════════════════════════
       MOBILE  (≤ 768 px)
    ══════════════════════════════════════════ */
    @media (max-width: 768px) {{

        /* Container sem padding lateral excessivo */
        .main .block-container {{
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-top: 0.8rem !important;
        }}

        /* Empilhar todas as colunas */
        [data-testid="column"] {{
            width: 100% !important;
            flex: 1 1 100% !important;
            min-width: 100% !important;
        }}

        /* Título menor */
        .main-header h1 {{ font-size: 1.15rem !important; }}

        /* Métricas compactas */
        [data-testid="stMetricValue"] {{ font-size: 1.3rem !important; }}
        [data-testid="stMetricLabel"] {{ font-size: 0.72rem !important; }}
        [data-testid="stMetric"] {{ padding: 8px 10px !important; }}

        /* Tabs com texto menor */
        .stTabs [data-baseweb="tab"] {{
            font-size: 0.78rem !important;
            padding: 6px 8px !important;
        }}

        /* Botão full-width (já está, mantém) */
        .stButton > button {{ font-size: 0.95rem; padding: 12px 16px; }}

        /* Esconder sidebar footer fixo no mobile */
        .sidebar-footer {{ display: none; }}

        /* Dataframe ocupa largura total */
        [data-testid="stDataFrame"] {{ font-size: 0.8rem; }}

        /* Selectbox maior para toque */
        .stSelectbox > div {{ min-height: 44px; }}
    }}

    /* ══════════════════════════════════════════
       MOBILE PEQUENO  (≤ 480 px)
    ══════════════════════════════════════════ */
    @media (max-width: 480px) {{
        .main-header h1 {{ font-size: 1rem !important; }}
        [data-testid="stMetricValue"] {{ font-size: 1.1rem !important; }}
        h2 {{ font-size: 1.05rem !important; }}
        h3 {{ font-size: 0.95rem !important; }}
    }}

    </style>
    """, unsafe_allow_html=True)


# ── Main App ──────────────────────────────────────────────────────────────────
def main():
    inject_css()

    # ── Sidebar ───────────────────────────────────────────────────────────────
    with st.sidebar:
        # Logo
        st.markdown("""
        <div class="logo-area">
            <img src="https://cms.raizeducacao.com.br/matriz/wp-content/uploads/sites/13/2024/09/Matriz-Educacao-Branco.svg"
                 width="160" onerror="this.style.display='none'"/>
            <div class="logo-subtitle">Dashboard NPS</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Navegação
        st.markdown("<p style='font-size:0.72rem;letter-spacing:1.5px;color:rgba(255,255,255,0.5);margin-bottom:6px;text-transform:uppercase;'>Menu</p>", unsafe_allow_html=True)
        page = st.radio("", [
            "📊  Visão Geral",
            "🏫  Análise por Unidade",
            "🤖  IA – Resumo & Plano de Ação",
        ], label_visibility="collapsed")

        st.divider()

        # Filtro por tipo de respondente
        st.markdown("<p style='font-size:0.72rem;letter-spacing:1.5px;color:rgba(255,255,255,0.5);margin-bottom:6px;text-transform:uppercase;'>Filtrar por</p>", unsafe_allow_html=True)
        tipo_filtro = st.radio("", [
            "👥  Todos",
            "👨‍👩‍👧  Responsáveis",
            "🎓  Alunos",
        ], label_visibility="collapsed")

        st.divider()

        # Chave Groq embutida via secrets (sem exposição no UI)
        _groq_builtin = st.secrets.get("GROQ_API_KEY", "")

        # Configurações de IA – apenas Anthropic precisa de chave manual
        with st.expander("⚙️  Configurações de IA"):
            provider = st.selectbox(
                "Provedor",
                ["Groq – Llama 3 (Grátis)", "Anthropic Claude (Pago)"],
            )
            if provider == "Groq – Llama 3 (Grátis)":
                api_key = _groq_builtin
                if api_key:
                    st.success("✅ IA Groq ativa")
                else:
                    api_key = st.text_input("API Key (Groq)", type="password",
                                            help="Grátis em console.groq.com")
            else:
                api_key = st.text_input("API Key (Anthropic)", type="password")

        st.session_state["_provider"] = provider
        st.session_state["_api_key"]  = api_key

        # Footer
        st.markdown("""
        <div class="sidebar-footer">
            Matriz Educação · NPS v1.0<br>Pesquisa março 2026
        </div>
        """, unsafe_allow_html=True)

    # Recover from session_state (set inside sidebar expander)
    provider = st.session_state.get("_provider", "Groq – Llama 3 (Grátis)")
    api_key  = st.session_state.get("_api_key", "")

    # Load data
    with st.spinner("Carregando dados..."):
        frames_all = load_all_data()

    # Aplicar filtro de tipo de respondente
    tipo_map = {
        "👥  Todos":           None,
        "👨‍👩‍👧  Responsáveis":  "Responsável",
        "🎓  Alunos":          "Aluno",
    }
    tipo_sel = tipo_map[tipo_filtro]
    if tipo_sel:
        frames = {u: df[df["_tipo"].str.contains(tipo_sel, case=False, na=False)]
                  for u, df in frames_all.items()}
    else:
        frames = frames_all

    summary_df = build_summary_df(frames)

    # ── PAGE 1: Visão Geral ────────────────────────────────────────────────────
    if page == "📊  Visão Geral":
        st.markdown(f'<div class="main-header"><h1>Visão Geral · NPS Matriz Educação</h1></div>', unsafe_allow_html=True)
        st.caption(f"Meta de NPS: **{NPS_TARGET}** pontos | Pesquisa: março 2026")

        # KPI row
        total_responses = summary_df["Total"].sum()
        overall_scores = []
        for df in frames.values():
            overall_scores.extend(df["_nps_raw"].dropna().tolist())
        overall_nps, ov_prom, ov_detr, ov_neut = compute_nps(overall_scores)
        above_target = (summary_df["NPS"] >= NPS_TARGET).sum()

        c1, c2, c3 = st.columns(3)
        c1.metric("NPS Geral", f"{overall_nps}", f"Meta: {NPS_TARGET}")
        c2.metric("Respostas", f"{total_responses:,}")
        c3.metric("Acima da meta", f"{above_target}/10")
        c4, c5 = st.columns(2)
        total_scores = len(overall_scores) if len(overall_scores) > 0 else 1
        c4.metric("% Promotores", f"{round(ov_prom/total_scores*100,1)}%")
        c5.metric("% Detratores", f"{round(ov_detr/total_scores*100,1)}%")

        st.divider()

        col_left, col_right = st.columns([3, 2])

        with col_left:
            st.subheader("NPS por Unidade")
            fig_bar = go.Figure()
            colors = [nps_color(n) for n in summary_df["NPS"]]
            fig_bar.add_trace(go.Bar(
                x=summary_df["NPS"],
                y=summary_df["Unidade"],
                orientation="h",
                marker_color=colors,
                text=summary_df["NPS"],
                textposition="outside",
                hovertemplate="<b>%{y}</b><br>NPS: %{x}<extra></extra>",
            ))
            fig_bar.add_vline(x=NPS_TARGET, line_dash="dash", line_color="red",
                              annotation_text=f"Meta {NPS_TARGET}", annotation_position="top right")
            fig_bar.update_layout(
                height=340, xaxis_title="NPS", yaxis_title="",
                margin=dict(l=10, r=60, t=20, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(range=[-20, 110]),
            )
            st.plotly_chart(fig_bar, width='stretch', config={'responsive': True, 'displayModeBar': False})

        with col_right:
            st.subheader("Distribuição da Rede")
            labels = ["Promotores", "Neutros", "Detratores"]
            values = [ov_prom, ov_neut, ov_detr]
            fig_pie = go.Figure(go.Pie(
                labels=labels, values=values,
                marker_colors=[MATRIZ_GREEN, "#95a5a6", "#e74c3c"],
                hole=0.5,
                textinfo="label+percent",
            ))
            fig_pie.update_layout(
                height=260, margin=dict(l=10, r=10, t=20, b=20),
                showlegend=False,
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_pie, width='stretch', config={'responsive': True, 'displayModeBar': False})

            # Gauge geral
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=overall_nps,
                title={"text": "NPS Geral"},
                gauge={
                    "axis": {"range": [-100, 100]},
                    "bar": {"color": nps_color(overall_nps)},
                    "steps": [
                        {"range": [-100, 0],  "color": "#fadbd8"},
                        {"range": [0, 50],    "color": "#fdebd0"},
                        {"range": [50, 75],   "color": "#fef9e7"},
                        {"range": [75, 100],  "color": "#e8fad0"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "value": NPS_TARGET},
                },
            ))
            fig_gauge.update_layout(height=210, margin=dict(l=20, r=20, t=40, b=10),
                                    paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_gauge, width='stretch', config={'responsive': True, 'displayModeBar': False})

        # Tabela detalhada
        st.subheader("Tabela Resumo por Unidade")
        styled = summary_df.copy()
        styled.index = range(1, len(styled) + 1)

        def color_nps(val):
            c = nps_color(val)
            return f"background-color: {c}; color: white; font-weight: bold; border-radius: 4px;"

        st.dataframe(
            styled.style.map(color_nps, subset=["NPS"]),
            width='stretch',
            height=420,
        )

        # Comparativo Promotores vs Detratores
        st.subheader("Promotores vs Detratores por Unidade")
        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(
            name="Promotores",
            x=summary_df["Unidade"],
            y=summary_df["% Promotores"],
            marker_color=MATRIZ_GREEN,
            text=summary_df["% Promotores"].apply(lambda x: f"{x}%"),
            textposition="auto",
        ))
        fig_comp.add_trace(go.Bar(
            name="Detratores",
            x=summary_df["Unidade"],
            y=summary_df["% Detratores"],
            marker_color="#e74c3c",
            text=summary_df["% Detratores"].apply(lambda x: f"{x}%"),
            textposition="auto",
        ))
        fig_comp.update_layout(
            barmode="group", height=300,
            xaxis_tickangle=-30,
            margin=dict(l=10, r=10, t=20, b=80),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_comp, width='stretch', config={'responsive': True, 'displayModeBar': False})

    # ── PAGE 2: Análise por Unidade ────────────────────────────────────────────
    elif page == "🏫  Análise por Unidade":
        st.markdown('<div class="main-header"><h1>Análise Detalhada por Unidade</h1></div>', unsafe_allow_html=True)

        unit = st.selectbox("Selecione a Unidade", list(frames.keys()))
        df   = frames[unit]
        scores = df["_nps_raw"].dropna().tolist()
        nps, prom, detr, neut = compute_nps(scores)

        # KPIs da unidade
        delta = nps - NPS_TARGET
        c1, c2 = st.columns(2)
        c1.metric("NPS", nps, f"{delta:+d} vs meta", delta_color="normal")
        c2.metric("Total Respostas", len(scores))
        c3, c4, c5 = st.columns(3)
        c3.metric("Promotores", prom, f"{round(prom/len(scores)*100,1)}%")
        c4.metric("Neutros", neut, f"{round(neut/len(scores)*100,1)}%")
        c5.metric("Detratores", detr, f"{round(detr/len(scores)*100,1)}%")

        st.divider()
        col_l, col_r = st.columns([2, 3])

        with col_l:
            # Gauge
            fig_g = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=nps,
                delta={"reference": NPS_TARGET, "valueformat": ".0f"},
                title={"text": f"NPS – {unit}"},
                gauge={
                    "axis": {"range": [-100, 100]},
                    "bar": {"color": nps_color(nps)},
                    "steps": [
                        {"range": [-100, 0],  "color": "#fadbd8"},
                        {"range": [0, 50],    "color": "#fdebd0"},
                        {"range": [50, 75],   "color": "#fef9e7"},
                        {"range": [75, 100],  "color": "#e8fad0"},
                    ],
                    "threshold": {"line": {"color": "red", "width": 3}, "value": NPS_TARGET},
                },
            ))
            fig_g.update_layout(height=240, margin=dict(l=20, r=20, t=60, b=10),
                                paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_g, width='stretch', config={'responsive': True, 'displayModeBar': False})

            # Pizza segmentos
            fig_p = go.Figure(go.Pie(
                labels=["Promotores", "Neutros", "Detratores"],
                values=[prom, neut, detr],
                marker_colors=[MATRIZ_GREEN, "#95a5a6", "#e74c3c"],
                hole=0.45,
                textinfo="label+value+percent",
            ))
            fig_p.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10),
                                showlegend=False, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_p, width='stretch', config={'responsive': True, 'displayModeBar': False})

        with col_r:
            # Atributos – radar
            attr_avgs = {}
            for attr in ATTR_COLS:
                vals = df[f"_attr_{attr}"].dropna().tolist()
                attr_avgs[attr] = sum(vals) / len(vals) if vals else None

            valid_attrs = {k: v for k, v in attr_avgs.items() if v is not None}
            if valid_attrs:
                categories = list(valid_attrs.keys())
                values_r   = list(valid_attrs.values())
                fig_radar = go.Figure(go.Scatterpolar(
                    r=values_r + [values_r[0]],
                    theta=categories + [categories[0]],
                    fill="toself",
                    fillcolor="rgba(129,215,66,0.2)",
                    line_color="rgba(129,215,66,0.9)",
                    name=unit,
                ))
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 6])),
                    height=320,
                    margin=dict(l=40, r=40, t=40, b=40),
                    paper_bgcolor="rgba(0,0,0,0)",
                    title="Avaliação por Atributo (média 1–6)",
                )
                st.plotly_chart(fig_radar, width='stretch', config={'responsive': True, 'displayModeBar': False})

            # Distribuição das notas NPS
            score_counts = pd.Series(scores).value_counts().sort_index()
            score_colors = ["#e74c3c" if s <= 6 else ("#95a5a6" if s <= 8 else "#2ecc71")
                            for s in score_counts.index]
            fig_dist = go.Figure(go.Bar(
                x=score_counts.index.astype(str),
                y=score_counts.values,
                marker_color=score_colors,
                text=score_counts.values,
                textposition="outside",
            ))
            fig_dist.update_layout(
                title="Distribuição das Notas (0-10)",
                height=230, margin=dict(l=10, r=10, t=40, b=20),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Nota", yaxis_title="Qtd",
            )
            st.plotly_chart(fig_dist, width='stretch', config={'responsive': True, 'displayModeBar': False})

        # Atributos – barras por categoria
        st.subheader("Médias por Atributo e Categoria")
        cat_data = []
        for attr, avg in valid_attrs.items():
            cat_data.append({"Atributo": attr, "Categoria": CATEGORY_MAP[attr], "Média": round(avg, 2)})
        cat_df = pd.DataFrame(cat_data).sort_values("Média", ascending=True)

        cat_colors = {"Pedagógico": MATRIZ_GREEN, "Administrativo": MATRIZ_BLUE, "Infraestrutura": "#e67e22"}
        fig_attrs = go.Figure()
        for cat, grp in cat_df.groupby("Categoria"):
            fig_attrs.add_trace(go.Bar(
                x=grp["Média"], y=grp["Atributo"],
                orientation="h", name=cat,
                marker_color=cat_colors.get(cat, "#7f8c8d"),
                text=grp["Média"].apply(lambda x: f"{x:.1f}"),
                textposition="outside",
            ))
        fig_attrs.update_layout(
            height=300, xaxis=dict(range=[0, 6.5]), barmode="overlay",
            margin=dict(l=10, r=60, t=10, b=20),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_attrs, width='stretch', config={'responsive': True, 'displayModeBar': False})

        # Comentários por segmento
        st.subheader("Comentários por Segmento")
        tab_prom, tab_neut, tab_detr = st.tabs(["✅ Promotores", "😐 Neutros", "⚠️ Detratores"])

        for tab, seg in zip([tab_prom, tab_neut, tab_detr], ["Promotor", "Neutro", "Detrator"]):
            with tab:
                seg_df = df[df["_segment"] == seg][["_nps_raw", "_feedback", "_comments", "_tipo"]].copy()
                seg_df.columns = ["Nota", "Motivo da Avaliação", "Sugestões/Comentários", "Tipo"]
                seg_df = seg_df[seg_df["Motivo da Avaliação"].str.strip() != ""]
                if seg_df.empty:
                    st.info("Sem comentários para este segmento.")
                else:
                    st.dataframe(seg_df.reset_index(drop=True), width='stretch', height=300)

    # ── PAGE 3: IA ────────────────────────────────────────────────────────────
    elif page == "🤖  IA – Resumo & Plano de Ação":
        st.markdown('<div class="main-header"><h1>IA · Resumo & Plano de Ação</h1></div>', unsafe_allow_html=True)

        if not api_key:
            st.warning("⚠️ A IA não está configurada. Abra **⚙️ Configurações de IA** na barra lateral.")
            st.stop()

        unit = st.selectbox("Selecione a Unidade", list(frames.keys()))
        df   = frames[unit]
        scores = df["_nps_raw"].dropna().tolist()
        nps, *_ = compute_nps(scores)
        feedbacks = [t for t in get_feedbacks_text(df) if len(t.strip()) > 5]

        # Atributos
        attr_avgs = {}
        for attr in ATTR_COLS:
            vals = df[f"_attr_{attr}"].dropna().tolist()
            attr_avgs[attr] = sum(vals) / len(vals) if vals else None

        st.info(f"**Unidade:** {unit} | **NPS Atual:** {nps} | **Meta:** {NPS_TARGET} | **Total de feedbacks:** {len(feedbacks)}")

        tab_summary, tab_plan = st.tabs(["📝 Resumo das Observações", "🎯 Plano de Ação"])

        with tab_summary:
            st.markdown("Clique no botão abaixo para gerar um resumo estruturado dos feedbacks desta unidade.")
            if st.button("✨ Gerar Resumo com IA", key="btn_summary"):
                if not feedbacks:
                    st.warning("Sem feedbacks disponíveis para esta unidade.")
                else:
                    with st.spinner("Analisando feedbacks..."):
                        try:
                            result = ai_summarize_unit(unit, nps, feedbacks, provider, api_key)
                            st.session_state[f"summary_{unit}"] = result
                        except Exception as e:
                            st.error(f"Erro ao chamar a API: {e}")

            if f"summary_{unit}" in st.session_state:
                st.markdown("---")
                st.markdown(st.session_state[f"summary_{unit}"])

                st.download_button(
                    "⬇️ Baixar Resumo (.txt)",
                    data=st.session_state[f"summary_{unit}"],
                    file_name=f"resumo_nps_{unit.lower().replace(' ','_')}.txt",
                    mime="text/plain",
                )

        with tab_plan:
            st.markdown("A IA irá analisar o NPS, os atributos e os feedbacks para criar um plano de ação detalhado.")

            if nps >= NPS_TARGET:
                st.success(f"🎉 Esta unidade já atingiu a meta de NPS {NPS_TARGET}! O plano de ação irá focar em **manter e superar** a meta.")

            if st.button("🚀 Gerar Plano de Ação com IA", key="btn_plan"):
                summary_text = st.session_state.get(f"summary_{unit}", "Resumo não gerado ainda.")
                with st.spinner("Elaborando plano de ação..."):
                    try:
                        result = ai_action_plan(unit, nps, summary_text, attr_avgs, provider, api_key)
                        st.session_state[f"plan_{unit}"] = result
                    except Exception as e:
                        st.error(f"Erro ao chamar a API: {e}")

            if f"plan_{unit}" in st.session_state:
                st.markdown("---")
                st.markdown(st.session_state[f"plan_{unit}"])

                st.download_button(
                    "⬇️ Baixar Plano de Ação (.txt)",
                    data=st.session_state[f"plan_{unit}"],
                    file_name=f"plano_acao_nps_{unit.lower().replace(' ','_')}.txt",
                    mime="text/plain",
                )

        # Comparativo todas as unidades abaixo da meta
        st.divider()
        st.subheader("📋 Unidades que precisam de atenção")
        below = summary_df[summary_df["NPS"] < NPS_TARGET].copy()
        if below.empty:
            st.success("Todas as unidades já atingiram a meta!")
        else:
            below["Gap para Meta"] = NPS_TARGET - below["NPS"]
            below = below.sort_values("Gap para Meta", ascending=False)
            st.dataframe(
                below[["Unidade", "NPS", "Gap para Meta", "% Detratores", "Total"]].reset_index(drop=True),
                width='stretch',
            )


if __name__ == "__main__":
    main()
