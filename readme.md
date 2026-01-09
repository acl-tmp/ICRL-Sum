
icrl/
├── data/
│   ├── raw_videos/         # Raw video files
│   ├── segments/           # Segmented window information (JSON)
│   ├── window_features/    # Output from tools (ASR, Frame features, etc.)
│   ├── evidence_db/        # Persisted Evidence Database (Vector Store index)
│   ├── summaries/          # Summary JSON results from each RL iteration
│   └── logs/               # Execution and debug logs
├── segmenter/
│   └── segmenter.py        # Window slicing based on audio/visual cuts
├── tools/                  # Perception tools: ASR, Frame Extraction, Features
│   ├── asr.py
│   ├── frame_extractor.py
│   ├── audio_feature.py    # Optional: Audio features
│   └── fusion.py           # Fuses multi-source info into LLM input units
├── knowledge_base/         # Evidence Database & Retrieval Module
│   ├── vector_store.py     # Manages Vector DB (e.g., FAISS, Chroma)
│   ├── indexer.py          # Indexes multimodal evidence (text + visual desc)
│   └── retriever.py        # Semantic search logic for "Critic-Guided Retrieval"
├── summary/                # Schema-related logic
│   ├── schema.py           # Summary JSON structure definition & validation
│   ├── builder.py          # Constructs/Updates summary JSON from LLM output
│   ├── formatter.py        # JSON <-> Text conversion for LLM I/O
│   └── evaluator.py        # Metric evaluation (used by Reward)
├── rl/                     # In-Context Reinforcement Learning Core
│   ├── reward.py           # Computes rewards (Alignment, Coherence, Grounding)
│   ├── critic.py           # Diagnoses errors & triggers retrieval (The "Judge")
│   ├── incontext_rl.py     # Main Loop: Base-LLM + Feedback-LLM
│   └── trajectory_logger.py# Logs every iteration's summary, reward, and feedback
├── llm/
│   ├── prompt.py           # Prompt construction: Base generation & Feedback
│   ├── llm_client.py       # Unified LLM wrapper (Local/Remote)
│   └── response_parser.py  # Parses LLM output into JSON/Text
├── pipeline/               # Workflow Orchestration
│   ├── preprocess.py       # Video -> Windows -> Tools -> Evidence Indexing
│   └── video2summary_icrl.py # Features -> Retrieval -> ICRL Optimization -> Summary
└── scripts/                # Execution Scripts
├── run_preprocess.sh   # Runs pipeline/preprocess.py
├── run_icrl_train.sh   # Runs pipeline/video2summary_icrl.py (RL iteration)
└── run_inference.sh    # Runs inference using the optimized policy







2. In-Context RL 主循环（Figure 1 绿色 + 蓝色模块）
Procedure InContext_RL_for_Schema(schema_serial):

    # 初始化 in-context 示例（可以是人工写的 or 之前 schema 的好问题）
    I = load_in_context_examples()

    # 初始状态 s0：包含 schema 文本 + 示例，不含任何反馈
    s0.context = [I, schema_serial]
    s0.feedback_list = []

    # ---- 第一次调用 base LLM，生成初始问题和 SQL ----
    (qs0, S0) = BaseLLM_generate_question_and_SQL(s0.context)

    R0 = Compute_Reward(S0)          # 复杂度 + keyword 分布
    best_q  = qs0
    best_S  = S0
    best_R  = R0

    t = 0
    s_t = s0

    while t < MAX_ITERS and not Converged(s_t, best_R):

        # ---------- 1) Feedback-LLM 根据当前问题/SQL + reward 给出文本反馈 ----------
        feedback_t = Feedback_LLM_generate(
            context      = s_t.context,
            question     = qs_t,        # 当前问题
            SQL          = S_t,         # 当前 SQL
            reward       = R_t          # 当前 reward
        )
        # 反馈内容形如：建议增加哪些条件、聚合、连接、逻辑运算符等

        # ---------- 2) 将反馈写回到 base LLM 的上下文中（状态更新） ----------
        s_{t+1}.feedback_list = s_t.feedback_list ∪ {feedback_t}
        s_{t+1}.context = [I, schema_serial, s_{t+1}.feedback_list]

        # ---------- 3) 在新的 context 下，base LLM 重新生成问题和 SQL ----------
        (qs_{t+1}, S_{t+1}) = BaseLLM_generate_question_and_SQL(s_{t+1}.context)

        # ---------- 4) 计算新的 reward ----------
        R_{t+1} = Compute_Reward(S_{t+1})

        # ---------- 5) RL 意义上的“策略更新”是通过上下文变化隐式完成的 ----------
        # 这里不显式更新参数，而是把反馈持续写入 context，改变生成分布。

        # ---------- 6) 记录最优结果 ----------
        if R_{t+1} > best_R:
            best_q = qs_{t+1}
            best_S = S_{t+1}
            best_R = R_{t+1}

        # ---------- 7) 准备下一轮 ----------
        t   = t + 1
        qs_t = qs_{t+1}
        S_t  = S_{t+1}
        R_t  = R_{t+1}
        s_t  = s_{t+1}

    # 循环结束后，best_q 即为 qs_final
    meta = { "SQL": best_S, "reward": best_R, "iterations": t }
    return best_q, meta


3. Reward 计算：复杂度 + 关键词分布
Function Compute_Reward(S):

    # 1) 复杂度得分：鼓励“够复杂但不过度”的 SQL
    #    e.g. 使用 JOIN、GROUP BY、HAVING、嵌套子查询等
    comp_score = 0

    if contains(S, "JOIN"):           comp_score += w_join
    if contains(S, "GROUP BY"):       comp_score += w_groupby
    if contains(S, "HAVING"):         comp_score += w_having
    if contains_subquery(S):          comp_score += w_subquery
    # 也可以加长度正则化：过长减分、过短也减分
    comp_score += f_length_penalty(len_tokens(S))

    # 2) 关键词类别得分：根据各类关键字出现情况评分
    kw_score = 0
    for keyword k in KEYWORD_CATEGORY_TABLE:
        if contains(S, k.token):
            kw_score += k.weight       # e.g. SELECT:1, JOIN:2, AVG:3, etc.

    # 3) 合成最终 reward（可线性组合或非线性）
    R = α * comp_score + β * kw_score

    return R




未来优化：
segmenter.py
    阶段二（未来优化）：引入 TransNet V2。 当我们把后面的 frame_extractor 和 asr 都写好后，如果你发现切分不够准，或者想利用闲置的 NPU，我们就把 MultimediaSegmenter 类里的 calcHist 逻辑替换成一个 NPU 推理的 TransNet 模型。


