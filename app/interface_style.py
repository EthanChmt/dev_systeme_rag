import gradio as gr


PACA_THEME = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="sky",
    neutral_hue="stone",
    radius_size="lg",
    spacing_size="lg",
    text_size="md",
).set(
    body_background_fill="#f7f1e5",
    body_background_fill_dark="#082f49",

    block_background_fill="#fffdf8",
    block_background_fill_dark="#12395b",

    block_border_color="#eadbc2",
    block_border_color_dark="#2f7ca6",

    input_background_fill="#ffffff",
    input_background_fill_dark="#082f49",

    input_border_color="#d9c8a8",
    input_border_color_focus="#d97706",

    button_primary_background_fill="#c65b32",
    button_primary_background_fill_hover="#ad4928",
    button_primary_text_color="#ffffff",

    button_secondary_background_fill="#e0f2fe",
    button_secondary_background_fill_hover="#bae6fd",
    button_secondary_text_color="#075985",

    loader_color="#d97706",
)


CUSTOM_CSS = """
.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
}

body {
    background:
        linear-gradient(
            180deg,
            #dff3fa 0%,
            #fff9ed 45%,
            #f3e4ca 100%
        ) !important;
}

.hero-paca {
    position: relative;
    overflow: hidden;
    padding: 42px;
    margin-bottom: 24px;
    border-radius: 28px;
    background:
        linear-gradient(
            135deg,
            #082f49 0%,
            #075985 55%,
            #0284c7 100%
        );
    color: white;
    box-shadow:
        0 20px 45px rgba(8, 47, 73, 0.22);
}

.hero-paca::before {
    content: "";
    position: absolute;
    top: -120px;
    right: -70px;
    width: 300px;
    height: 300px;
    border-radius: 50%;
    background: rgba(255, 255, 255, 0.10);
}

.hero-paca::after {
    content: "";
    position: absolute;
    right: 110px;
    bottom: -170px;
    width: 260px;
    height: 260px;
    border-radius: 50%;
    background: rgba(255, 211, 120, 0.13);
}

.hero-content {
    position: relative;
    z-index: 1;
}

.hero-kicker {
    display: inline-block;
    margin-bottom: 12px;
    padding: 7px 13px;
    border: 1px solid rgba(255, 255, 255, 0.23);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.10);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.hero-paca h1 {
    max-width: 780px;
    margin: 0;
    color: white;
    font-size: 42px;
    line-height: 1.08;
}

.hero-paca p {
    max-width: 730px;
    margin-top: 17px;
    color: rgba(255, 255, 255, 0.90);
    font-size: 17px;
    line-height: 1.65;
}

.hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 22px;
}

.hero-tags span {
    padding: 7px 11px;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 999px;
    background: rgba(255, 255, 255, 0.08);
    font-size: 12px;
}

.paca-card {
    padding: 22px !important;
    border: 1px solid rgba(8, 47, 73, 0.10) !important;
    border-radius: 20px !important;
    background: rgba(255, 253, 248, 0.95) !important;
    box-shadow:
        0 10px 30px rgba(8, 47, 73, 0.08) !important;
}

.section-title {
    margin-bottom: 5px;
    color: #075985;
    font-size: 22px;
    font-weight: 750;
}

.section-description {
    margin-bottom: 16px;
    color: #64748b;
    font-size: 14px;
    line-height: 1.55;
}

.answer-output textarea {
    min-height: 190px !important;
    background: #fffdf8 !important;
    font-size: 16px !important;
    line-height: 1.65 !important;
}

.notice {
    padding: 12px 15px;
    border-radius: 12px;
    font-size: 14px;
}

.notice-info {
    border: 1px solid #b6ddec;
    background: #eff9fd;
    color: #075985;
}

.notice-success {
    border: 1px solid #b5d8bc;
    background: #eef8f0;
    color: #376a40;
}

.notice-warning {
    border: 1px solid #ecd199;
    background: #fff8e6;
    color: #8a5a08;
}

.notice-error {
    border: 1px solid #e5b1aa;
    background: #fff0ed;
    color: #a13d31;
}

.status-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 11px;
}

.status-item {
    display: flex;
    align-items: center;
    gap: 9px;
    padding: 10px 13px;
    border: 1px solid rgba(8, 47, 73, 0.10);
    border-radius: 12px;
    background: white;
}

.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
}

.status-ok {
    background: #448151;
}

.status-error {
    background: #bd4437;
}

.sources-grid {
    display: grid;
    grid-template-columns:
        repeat(auto-fit, minmax(260px, 1fr));
    gap: 15px;
}

.source-card {
    padding: 18px;
    border: 1px solid rgba(8, 47, 73, 0.11);
    border-radius: 17px;
    background:
        linear-gradient(
            180deg,
            #ffffff,
            #fff8ea
        );
    box-shadow:
        0 7px 20px rgba(8, 47, 73, 0.07);
}

.source-header {
    display: flex;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 12px;
}

.source-number {
    padding: 5px 9px;
    border-radius: 999px;
    background: #e0f2fe;
    color: #075985;
    font-size: 11px;
    font-weight: 700;
}

.source-city {
    color: #c2412d;
    font-size: 12px;
    font-weight: 700;
}

.source-card h3 {
    margin: 0 0 13px;
    color: #075985;
    font-size: 17px;
}

.source-detail {
    margin-top: 8px;
    color: #334155;
    font-size: 13px;
    line-height: 1.5;
}

.source-detail strong {
    color: #66743c;
}

.empty-state {
    padding: 22px;
    border: 1px dashed rgba(8, 47, 73, 0.25);
    border-radius: 14px;
    background: rgba(224, 242, 254, 0.25);
    color: #64748b;
    text-align: center;
}

.api-links {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin-top: 14px;
}

.api-links a {
    padding: 9px 13px;
    border-radius: 10px;
    background: #e0f2fe;
    color: #075985 !important;
    font-weight: 700;
    text-decoration: none !important;
}

.footer-paca {
    margin: 25px 0 8px;
    color: #64748b;
    font-size: 12px;
    text-align: center;
}

@media (max-width: 700px) {
    .hero-paca {
        padding: 27px 21px;
    }

    .hero-paca h1 {
        font-size: 31px;
    }

    .sources-grid {
        grid-template-columns: 1fr;
    }
}
"""