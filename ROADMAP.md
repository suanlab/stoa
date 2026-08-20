# STOA 로드맵 (12–18개월)

> 설계 근거: [`docs/stoa_design.md`](docs/stoa_design.md) §7. 각 마일스톤은 검증 가능한 산출물(Deliverable)과
> 연구 질문(RQ) 연결을 가진다.

## 연구 질문 (RQ)
- **RQ1 (통합 이득)** — 표현×계층×시점 공동 학습이 정적 MemOS·Memory-R1을 정확도–지연–비용 프론티어에서 파레토 지배하는가?
- **RQ2 (신용 할당)** — Belady 모방 웜스타트 + 프로세스 감독이 오프라인 오라클의 5% 이내에 도달하는가?
- **RQ3 (제어성·일반화)** — 단일 (L,C,B)-조건 정책이 재학습 없이 프론티어를 그리고 백본 간 전이되는가?

---

## 마일스톤

### M0–3 · 기반 구축  *(진행 중)*
- [x] MDP 정형화 코드화 (`environment.py`: 상태·행동공간(마스킹 후 50)·라그랑주 보상)
- [x] 합성 트레이스 생성기 (`traces.py`) + 계층 비용·용량 모델 (`simulator.py`)
      — *부분*: 실 LMCache 트레이스 보정(지연오차 ≤15% 게이트)은 트레이스 접근 확보 후 (인프라 대기)
- [ ] MemOS + M+ 잠재경로 계측 어댑터
- [x] **Belady 오라클 라벨러** (`credit.py::belady_optimal_action`): 전체 미래접근 기반 사후최적 배치
- [x] **산출물**: 오프라인 트레이스 데이터셋 + 오라클 상한 베이스라인
      (`scripts/build_dataset.py` → `experiments/m0_oracle_baseline.json`; 재현성 게이트 통과)

### M3–6 · 단일축 컨트롤러  *(진행 중)*
- [x] 휴리스틱 베이스라인: LRU · H2O(LFU) · 정적 MemOS (`eval/online.py` 온라인 축출)
- [x] 계층-only 컨트롤러 기준선: Belady 온라인 오라클(웜스타트 타깃) — *학습 예측기는 M6–9*
      / 표현·결합 축: 정적 오라클 그리디 (`eval/oracle.py`)
- [x] **LoCoMo 파이프라인 연결** (`eval/locomo.py`, `benchmarks.load('locomo')`) — 실 데이터(10샘플),
      턴=팩트·evidence=needed 매핑. 예산 30%서 demand-aware 선택이 무작위 대비 answerable 118 vs 41
      (`scripts/run_locomo.py`). LongMemEval은 어댑터 대기.
- [x] **산출물**: 축별 ablation (RQ1 기초) — `scripts/run_ablation.py` → `experiments/m3_ablation.json`
      결과: 계층축 **Belady>H2O>LRU**(오라클 갭 3.5→6.7pp↑, 캐시↑), 표현축 −73% 토큰비용

### M6–9 · 분해 공동 정책  *(진행 중 — GNN BC 웜스타트 완료, CPU)*
- [x] Belady 웜스타트(선형): 재사용-거리 예측기 (`learn.py`, PARROT식)
- [x] **분해 정책 π_rep·π_tier·π_time + 메모리 그래프 GNN** (`gnn.py`) — 50개 합법행동 마스킹 소프트맥스,
      hand-rolled GCN(torch CPU, **GPU 불필요**; torch-geometric 미사용)
- [x] **Belady 행동 복제(BC) 웜스타트** (`train.py`) + **RQ2 증거**: 홀드아웃 모방정확도 ~87%,
      BC 배치비용이 Belady 타깃 이내(때로 하회), naive→타깃 갭 ~104% 회수
      (`scripts/train_gnn_policy.py` → `experiments/m6_gnn_policy.json`)
- [x] **오프라인 정책경사 파인튜닝**(`rl.py`): REINFORCE + per-item 신용 + value 헤드 baseline.
      정직한 결과: 정적 배치서 RL이 BC를 명확히 지배 못함 → 진단: 계층 점유상태 부재.
- [x] **순차 점유인식 MDP**(`sequential.py`) — 위 진단의 해소: 정책이 실시간 점유 관찰+마스킹으로 항상 실현가능,
      **sequential-RL이 그리디 오라클 비용 도달 + static BC(용량 위반) 지배**. (`scripts/train_sequential.py`)
- **산출물**: GNN 분해 정책 + BC 웜스타트(RQ2) + 순차 점유인식 정책(음성 결과 → 해소)

### M9–12 · 온라인 학습 & 예산 제어  *(진행 중 — torch-free 코어 완료)*
- [x] 예산조건 제어기: 토큰 예산 B → 최소비용 배치 프론티어 (`eval/budget.py`)
- [x] **RQ3 증거**: 단일 노브 B가 임의 예산에서 실현가능 배치 산출; 정적 베이스라인은 단일 점
      (`scripts/run_budget_frontier.py` → `experiments/m9_budget_frontier.json`)
- [ ] 온라인 GRPO + 세그먼트 advantage(DCPO), (L,C,B)-조건 *학습* 정책 — *torch 대기*
- [ ] 전체 MemoryAgentBench 평가 — *`datasets` 어댑터 대기*
- **산출물**: RQ3 제어성 프론티어(예비). RQ1 파레토 지배는 실 트레이스 보정 후 재검(placeholder 비용에선
      uniform-vector가 이미 효율적이라 지배 폭 작음 — 정직한 한계)

### M12–15 · 실배포 검증
- [ ] 실제 vLLM + LMCache 배포
- [ ] Sim-to-real 검증
- [ ] 신선도/지식갱신 스트레스 테스트
- **산출물**: 배포 검증 리포트

### M15–18 · 확장 & 공개
- [ ] 스케일링, 백본 간 전이, 스래싱 강건성 분석
- [ ] 오픈소스 오케스트레이터 + 계층 시뮬레이터 릴리스 (`stoa-mem`)
- **산출물**: 정식 논문 + 오픈소스

---

## 핵심 지표 (평가)
정확도 × end-to-end 지연(TTFT+총) × \$/토큰 × 신선도(staleness) × 통합 오버헤드 → **파레토 hypervolume**.
베이스라인 상한: **Belady 오라클**.

## 리스크 (요약)
행동 조합폭발 → 분해 정책·마스킹 · 지연/파열 신용 → Belady+프로세스 감독 · Sim-to-real 격차 → 보정 시뮬레이터 ·
마이그레이션 스래싱 → 비용 페널티·히스테리시스. 전체는 설계 문서 §6.
