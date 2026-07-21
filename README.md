# -Multi-Agent-
RAGGuard 是一个客服回复幻觉检测系统，采用 **Claim-Level NLI（自然语言推理）验证**策略：将客服回复拆解为原子事实声明，逐条与知识库进行三分类校验（ENTAILED / CONTRADICTED / UNMENTIONED），最终输出幻觉类型、严重度和详细诊断信息。
