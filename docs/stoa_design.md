# 연구 제안서 (T1): 학습형 메모리-계층 오케스트레이터 (Learned Memory-Tier Orchestrator, STOA)

> **주제 출처**: 본 조사 `survey_agent_memory_db_index.md` §9 미래 유망 연구 주제 T1 (최우선, 세 축 Memory×DB×Index 관통).
> **작성일**: 2026-07-15. **검증**: 모든 인용 arXiv 확인. 2건(Adaptive-RAG 2403.14403, Mem-α 2509.25911)은 정식 인용 전 `[검증 필요]`.

---

## 0. 검증된 근거 문헌

### 앵커 논문 (모두 arXiv 확인)
| 키 | 논문 | arXiv | 역할 |
|---|---|---|---|
| MemOS | A Memory OS for AI System (Li et al., 2025) | 2507.03724 | 평문·활성화·파라미터 3계층 MemCube (마이그레이션 프리미티브 존재, **학습형 스케줄러 부재**) |
| Memory-R1 | Managing/Utilizing Memories via RL (Yan et al., 2025) | 2508.19828 | PPO/GRPO ADD/UPDATE/DELETE/NOOP (**내용만, 표현·계층 미선택**) |
| Memory-as-Action | Autonomous Context Curation (Zhang et al., 2025) | 2510.12635 | 문맥 편집을 RL 행동으로 (DCPO, 궤적 파열 처리) |
| Sleep-time Compute | Beyond Inference Scaling (Lin et al., 2025) | 2504.13171 | 오프라인 사전계산 (~5× 절감, **전역 모드**) |
| LightMem | Lightweight Memory-Augmented Gen (2025) | 2510.18866 | Atkinson–Shiffrin 단계 + 수면기 통합 |
| LMCache | KV Cache Layer (Cheng, Liu et al., 2025) | 2510.09665 | GPU/CPU/local/remote KV 계층화 (**의미 효용 무시**) |
| M+ | Extending MemoryLLM (Wang et al., 2025) | 2502.00592 | 잠재공간 메모리 + 공동학습 검색기 |
| RAG-Gym | Process Supervision (Xiong et al., 2025) | 2502.13957 | 단계별 프로세스 보상·전이 가능 critic |
| Big ANN | NeurIPS'23 Competition (Simhadri et al., 2024) | 2409.17424 | filtered/OOD/sparse/streaming 트랙 |

### ML-for-systems 근거 (표현×계층 결정이 학습 가능함을 입증)
| 키 | 논문 | arXiv | 역할 |
|---|---|---|---|
| MemGPT | LLMs as Operating Systems (Packer et al., 2023) | 2310.08560 | OS식 가상문맥 페이징 (휴리스틱) |
| H2O | Heavy-Hitter Oracle (Zhang et al., NeurIPS 2023) | 2306.14048 | 어텐션 점수 KV 축출 휴리스틱 |
| PARROT | Imitation Learning for Cache Replacement (Liu et al., ICML 2020) | 2006.16239 | Belady 오라클 모방 |
| Attention-Gate | In-context KV-Cache Eviction (2024) | 2410.12876 | 학습형 경량 축출 모듈 |
| Cold-RL | Cache Eviction with Offline RL (2025) | 2508.12485 | 프로덕션 오프라인-RL 축출 |
| Decima | Scheduling for Data Clusters (Mao et al., SIGCOMM 2019) | 1810.01963 | GNN + 정책경사 스케줄러 |

### 평가 벤치마크
LongMemEval (2410.10813), LoCoMo (2402.17753), MemoryAgentBench (2507.05257).

---

## 1. 문제 정의 및 시의성 (Why now)

에이전트 메모리 연구는 **하나의 근본 결정을 세 커뮤니티가 각기 한 축만 풀고 나머지 둘을 무시**하며 분열되어 있다.

- **메모리-OS 축 (어떤 표현):** MemOS·MemGPT는 평문·활성화·파라미터 위 추상(MemCube/가상문맥)과 마이그레이션·융합 프리미티브를 제공하나, *언제·마이그레이션할지*는 휴리스틱 — 정확도↔지연↔비용을 저울질하는 **학습형 계층 스케줄러가 없다.**
- **학습형 관리 축 (어떤 내용):** Memory-R1·Memory-as-Action은 RL로 ADD/UPDATE/DELETE·문맥 편집을 결정하나, **단일 표현 위 내용 연산**일 뿐 표현·물리 계층을 선택하지 않고 오프라인 통합을 스케줄하지 않는다.
- **시스템 축 (어디에 물리적으로):** LMCache는 KV를 GPU/CPU/디스크/RDMA로 계층화; PARROT·H2O·Attention-Gate·Cold-RL은 캐시 축출을, Decima는 클러스터 스케줄링을 학습한다. 그러나 이들은 **의미적 태스크 효용(정답 정확도)이 아니라 miss-rate·TTFT**만 최적화한다.

**Gap.** 각 메모리 항목에 대해, 공유 예산 하에서 **표현(평문/벡터/그래프/잠재-활성화/파라미터) × 계층(GPU/CPU/디스크/원격-RDMA) × 갱신시점(온라인 vs 오프라인 수면기)** 을 **동시에 결정**하는 시스템이 없다. Sleep-time·LightMem은 오프라인 통합의 가치(최대 5× 절감)를 보였으나 이를 **전역 모드**로만 다룬다.

**축은 결합되어 있다.** 항목을 파라미터/잠재 표현(M+)으로 승격하는 것은 비싼 오프라인 행위지만 이후 서빙이 싸고 빠르다; 디스크 평문으로 두면 쓰기는 싸지만 읽기가 느리고 토큰을 많이 쓴다. 이는 정확히 **제약 스케줄링 문제**이며, ML-for-systems(Decima·PARROT·Cold-RL)는 이런 문제가 학습 가능함을 반복 입증했다.

**왜 지금인가.** 모든 기질(MemCube 마이그레이션, RL 메모리 관리자, 계층 KV, 잠재 메모리)이 2025년 말 기준 존재하지만 **오케스트레이션되지 않은** 상태다.

---

## 2. 문제 정형화 (Formulation)

오케스트레이터를 **예산제약 MDP**로 모델링하고 학습 정책 π_θ로 푼다.

**상태 s_t** (후보 항목 i별):
- (a) 의미: 임베딩, 최신성, 접근 빈도, provenance/버전(MemCube 메타), 예측 미래접근(PARROT식 Belady 타깃)
- (b) 시스템: 현재 표현, 현재 계층, 항목 크기, GPU/CPU/디스크/RDMA별 점유·압력
- (c) 예산: 잔여 지연 여유, 세션 잔여 달러·토큰 예산
- (d) 질의 문맥: 현재/예측 질의 임베딩

**행동 공간** (항목별):
```
A = 표현 {평문, 벡터, 그래프, 잠재-활성화, 파라미터}
  × 계층 {GPU, CPU, 디스크, 원격-RDMA}
  × 시점 {온라인-즉시, 오프라인-수면기, no-op}
```
불법 조합은 마스킹. 조합폭발 완화를 위해 **공유 트렁크 위 π_θ = π_rep · π_tier · π_time 로 분해**하고, 항목별 독립이 아니라 **메모리 그래프 위 Decima식 GNN으로 배치 결정**.

**목적함수** — 예산 하 기대 태스크 효용 최대화:
```
max_θ  E[ Σ_t U(a_t) ]   s.t.  Latency ≤ L,  Cost($) ≤ C,  Tokens ≤ B
```
U = 다운스트림 태스크 보상(정답 정확도, 또는 RAG-Gym식 프로세스 감독 단계 보상). **라그랑주 완화**로 학습형 듀얼 λ_L, λ_C, λ_B 도입:
```
r_t = U − λ_L·lat − λ_C·cost − λ_B·tok + freshness_bonus
```
→ 운영자가 서빙 시 (L, C, B)를 설정하면 정책이 적응하는 **단일 제어 노브**. 정적 MemOS와 대비되는 핵심 이점.

---

## 3. 제안 방법 / 아키텍처 & 신용 할당

**STOA**는 MemOS식 저장소 + LMCache식 계층화 + M+식 잠재 경로 위의 **제어 평면(control plane)**.

- **지각(Perception):** GNN이 메모리 저장소(노드=항목, 엣지=공동접근/엔티티 링크) + 시스템 텔레메트리 벡터를 인코딩 → 항목별 임베딩.
- **정책(Policy):** 분해 actor가 항목별 (표현, 계층, 시점) 방출; critic이 제약 가치 추정; 별도 admission/consolidation 헤드가 오프라인 수면기 큐 항목 선택.
- **실행(Execution):** 온라인 행동은 MemCube migrate/fuse + LMCache put/get 호출; 오프라인 행동은 유휴 구간에 평문→잠재/파라미터 또는 그래프 요약 통합(LightMem식).

### 신용 할당 (핵심 난제)
배치의 보상은 질의 시점에 **여러 스텝 뒤** 도착하고, 메모리 편집이 궤적 프리픽스를 파열시켜(Memory-as-Action) 순진한 정책경사가 깨진다. **세 요소를 결합**:
1. **프로세스 감독(RAG-Gym):** 학습 critic으로 마이그레이션/통합 행위에 조밀한 단계별 보상 부여.
2. **Belady 오라클 모방(PARROT):** 전체 미래접근 트레이스로부터 사후최적 배치를 웜스타트 후 RL 파인튜닝.
3. **세그먼트 수준 이점(DCPO):** 편집으로 유발된 궤적 세그먼트에 걸친 advantage.

학습은 **오프라인-RL 우선(Cold-RL식, 로그 트레이스에서 안전)** → 가드된 온라인 GRPO 순.

---

## 4. 연구 질문 / 가설

- **RQ1 (통합 이득).** *H1:* 표현×계층×시점 공동 학습이 정적 MemOS·Memory-R1을 정확도–지연–비용 프론티어에서 파레토 지배하며, **표현 선택이 중요해지는 타이트한 예산에서 이득 최대.**
- **RQ2 (신용 할당).** *H2:* Belady 모방 웜스타트 + 프로세스 감독이 오프라인 오라클 배치의 5% 이내 도달, 희소보상 RL 대비 샘플효율 우위.
- **RQ3 (제어성·일반화).** *H3:* 단일 (L,C,B)-조건 정책이 재학습 없이 프론티어를 그리고 백본/데이터셋 간 전이(RAG-Gym 전이 critic과 유사).

---

## 5. 평가 계획

**베이스라인:** 정적 MemOS(고정 계층); 휴리스틱 LRU·중요도/H2O 축출; Attention-Gate 학습 축출; Memory-R1(내용만 RL); LMCache-only(시스템만 계층화); Sleep-time/LightMem(전역 오프라인); **Belady 오라클 상한.**

**지표:** 태스크 정확도; end-to-end 지연(TTFT + 총); \$/질의, 토큰/질의; **신선도**(지식 갱신 하 staleness); 통합 오버헤드; **(정확도, 지연, 비용) 파레토 hypervolume.**

**벤치마크:** LongMemEval(추출·다세션·시간·지식갱신·기권), LoCoMo(300턴 다세션 QA), MemoryAgentBench(검색·test-time학습·장기이해·충돌해결). 검색 컴포넌트는 Big ANN 스트리밍/필터 트랙으로 스트레스 테스트. **축별 ablation**으로 각 기여 분리.

---

## 6. 위험 및 완화

| 위험 | 완화 |
|---|---|
| 행동 조합폭발 | 분해 정책 + 행동 마스킹 + 배치 GNN 결정 |
| 지연/파열 신용 | Belady 웜스타트, 프로세스 감독, 세그먼트 advantage |
| Sim-to-real 시스템 격차 | LMCache 트레이스 기반 보정 시뮬레이터; 상위 정책은 실제 vLLM+LMCache 검증 |
| 보상 해킹/마이그레이션 스래싱 | 마이그레이션 비용 페널티, 히스테리시스, 윈도우당 이동 제한 |
| 통합이 사실 손상 | MemCube 버전/provenance 가드; LongMemEval 기권을 안전 지표로 |
| 약한 베이스라인이 이득 과장 | Belady 오라클 상한 + 강한 축별 베이스라인 포함 |

---

## 7. 마일스톤 로드맵 (12–18개월)

| 기간 | 작업 | 산출물 |
|---|---|---|
| M0–3 | MDP 정형화; LMCache 로그 기반 계층 시뮬레이터; MemOS+M+ 잠재경로 계측; Belady 라벨러 | 오프라인 트레이스 데이터셋 + 오라클 베이스라인 |
| M3–6 | 단일축 학습 컨트롤러(계층만, 표현만) + 휴리스틱 베이스라인 (LongMemEval/LoCoMo) | 축별 ablation (RQ1 기초) |
| M6–9 | 분해 공동 정책; 오프라인-RL + Belady 웜스타트; 프로세스 감독 critic | 최초 공동 STOA, RQ2 증거 |
| M9–12 | 온라인 GRPO + 세그먼트 advantage; 예산조건 정책; 전체 MemoryAgentBench | RQ1/RQ3 프론티어 결과; 워크숍 논문 |
| M12–15 | 실제 vLLM+LMCache 배포; sim-to-real 검증; 신선도/지식갱신 스트레스 | 배포 검증 |
| M15–18 | 스케일링, 백본 간 전이, 스래싱 강건성 분석 | 정식 논문 + 오픈소스 오케스트레이터 |

---

## 8. 기대 기여
1. Memory·DB·Index 세 축을 **단일 학습형 제어 문제**로 통합하는 최초 정형화.
2. 메모리 계층 배치에 대한 **신용 할당 방법론**(Belady 모방 + 프로세스 감독 + 세그먼트 advantage).
3. **예산조건 단일 정책**으로 정확도–지연–비용 프론티어를 서빙 시 제어.
4. 오픈소스 오케스트레이터 + 계층 시뮬레이터.

---

*본 제안은 검증된 앵커(MemOS·Memory-R1·LMCache·Sleep-time·M+·RAG-Gym) 위에 ML-for-systems 근거(PARROT·Cold-RL·Decima)를 결합해 구성했다. Adaptive-RAG(2403.14403)·Mem-α(2509.25911)는 정식 인용 전 재확인 필요.*
