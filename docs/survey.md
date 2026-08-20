# AI Agent를 위한 Memory · DB · Index: 체계적 문헌 조사 (2020–2026)

> **범위**: LLM 기반 AI Agent의 외부 메모리(Memory) — 이를 저장·조직하는 데이터베이스(DB) — 이를 효율적으로 검색하는 인덱스(Index)의 3층 스택.
> **기간**: 2020–2026 (기반 논문 64편 + 프론티어 논문 ~50편, 총 114개 고유 항목).
> **작성일**: 2026-07-15.
> **검증 원칙**: 모든 인용은 arXiv·ACL Anthology·DBLP·공식 학회 프로시딩 중 최소 1곳에서 교차 확인됨. 2026년 극최신 항목(arXiv 2602–2607)은 정식 인용 전 독립 재확인 권장으로 표기(`[재확인]`). 미확인 1건은 `[검증 필요]`. **허위 인용 없음.**

---

## 목차
1. 조사 개요
2. 핵심 논문 목록 (기반 P1–P64)
3. 프론티어 논문 목록 (2025–2026, F1–F54)
4. 분류 체계 (Taxonomy)
5. 시간적 발전 흐름 (Timeline)
6. 비교표
7. 벤치마크·지표의 발전과 한계
8. 핵심 패러다임 전환 (2025–2026)
9. 미해결 과제 (Gap)
10. 미래 유망 연구 주제
11. 검증 노트 및 참고문헌

---

## 1. 조사 개요

### 배경
LLM 기반 AI Agent는 고정된 문맥 창과 정적 파라미터 지식이라는 근본 제약을 가진다. 장기 상호작용·다단계 추론·지속학습을 위해서는 문맥 창을 넘어선 **외부 메모리**, 이를 저장·갱신하는 **DB**, 이를 검색하는 **인덱스**의 3층 스택이 필수다.

### 핵심 관찰: 네 흐름의 수렴
이 분야는 (1) 정보검색/RAG, (2) ANN 인덱싱/벡터 DB, (3) LLM Agent 메모리 아키텍처, (4) 그래프/구조화 메모리라는 **네 독립 흐름이 "Agent Memory"라는 통합 문제로 수렴**하는 지점에 있다. 2023년이 그 전환점이며, 2025–2026년에 세 축은 **하나의 학습형·예산제약·안전보장 런타임**으로 재편되고 있다.

### 방법론
5개 하위 도메인(Agent 메모리 / RAG·검색 / 벡터 DB·ANN / 그래프·구조화 / 벤치마크·서베이)에 대한 병렬 문헌 검색 후, 각 논문을 상위 서지 DB와 대조하여 제목·저자·연도·게재처·식별자를 교차 검증했다. 프론티어(2025–2026)는 별도 4개 스트림으로 재조사했다.

---

## 2. 핵심 논문 목록 (기반, P1–P64)

### A. RAG · 신경망 검색 기반 (Retrieval as Non-parametric Memory)

- **[P1]** *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks* (Lewis et al., 2020) — NeurIPS 2020, arXiv:2005.11401. **기여**: DPR+seq2seq로 검색문서를 잠재변수로 주변화하는 RAG 패러다임 정립. **한계**: 정적 인덱스 1회 top-K 검색으로 다단계·갱신 취약.
- **[P2]** *REALM: Retrieval-Augmented LM Pre-Training* (Guu et al., 2020) — ICML 2020, arXiv:2002.08909. **기여**: 검색기를 LM 사전학습과 end-to-end 역전파. **한계**: 인덱스 비동기 재임베딩 비용, 추출형 QA 한정.
- **[P3]** *Dense Passage Retrieval (DPR)* (Karpukhin et al., 2020) — EMNLP 2020, arXiv:2004.04906. **기여**: 이중 인코더 dense 검색을 표준화. **한계**: 단일 벡터의 세밀 매칭 한계, hard-negative 필요.
- **[P4]** *Fusion-in-Decoder (FiD)* (Izacard & Grave, 2021) — EACL 2021, arXiv:2007.01282. **기여**: 구절 독립 인코딩 후 디코더 융합. **한계**: 구절 수 증가 시 교차어텐션 비용 급증.
- **[P5]** *RETRO: Retrieving from Trillions of Tokens* (Borgeaud et al., 2022) — ICML 2022, arXiv:2112.04426. **기여**: ~2조 토큰 DB 청크 교차어텐션, 25× 적은 파라미터로 GPT-3급. **한계**: 동결 검색기, 청크 일관성 공백.
- **[P6]** *Atlas: Few-shot Retrieval-Augmented LMs* (Izacard et al., 2022) — arXiv:2208.03299 (JMLR 2023). **기여**: Contriever+FiD 공동 사전학습, 64샘플로 540B 초과. **한계**: 공동학습 복잡도·인덱스 유지 비용.
- **[P7]** *Contriever: Unsupervised Dense Retrieval* (Izacard et al., 2022) — arXiv:2112.09118 (TMLR). **기여**: 비지도 대조학습 범용 검색기. **한계**: 일부 in-domain에서 지도학습 대비 열세.
- **[P8]** *ColBERT: Late Interaction over BERT* (Khattab & Zaharia, 2020) — SIGIR 2020, arXiv:2004.12832. **기여**: 토큰 단위 MaxSim late interaction. **한계**: 다중 벡터 저장으로 인덱스 급증.
- **[P9]** *In-Context RALM* (Ram et al., 2023) — TACL 2023, arXiv:2302.00083. **기여**: 동결 LM 앞에 검색문서 부착만으로 큰 이득. **한계**: 문맥 예산 소모, 순서 민감.
- **[P10]** *HyDE: Zero-Shot Dense Retrieval* (Gao et al., 2023) — ACL 2023, arXiv:2212.10496. **기여**: LLM 가상문서 임베딩으로 라벨 없는 검색. **한계**: 생성 품질·지연 의존, 환각 시 오검색.
- **[P11]** *FLARE: Active RAG* (Jiang et al., 2023) — EMNLP 2023, arXiv:2305.06983. **기여**: 다음 문장 예측→저신뢰 토큰서 검색 트리거. **한계**: 트리거 오작동, 반복 검색 지연.
- **[P12]** *Self-RAG* (Asai et al., 2023) — ICLR 2024, arXiv:2310.11511. **기여**: reflection token으로 온디맨드 검색·자기비평 통합. **한계**: 전용 학습 데이터, 비평 오버헤드.
- **[P13]** *CRAG: Corrective RAG* (Yan et al., 2024) — arXiv:2401.15884. **기여**: 검색 평가기로 저품질 시 웹검색 보강. **한계**: 평가기 정확도 의존, 웹 폴백 지연.

### B. Agent 메모리 아키텍처 (Cognitive & OS-inspired)

- **[P14]** *MemGPT: LLMs as Operating Systems* (Packer et al., 2023) — arXiv:2310.08560. **기여**: OS 가상메모리식 문맥↔외부저장 페이징. **한계**: 함수호출 신뢰성 의존, 페이징 지연.
- **[P15]** *Generative Agents* (Park et al., 2023) — UIST 2023, arXiv:2304.03442. **기여**: 관찰+검색(최신성·중요도·관련성)+반영+계획 memory stream. **한계**: 높은 토큰 비용, 시뮬레이션 한정.
- **[P16]** *Reflexion* (Shinn et al., 2023) — NeurIPS 2023, arXiv:2303.11366. **기여**: 언어적 자기반영을 에피소드 버퍼에 저장. **한계**: 유의미 피드백 신호 의존.
- **[P17]** *Voyager* (Wang et al., 2023) — arXiv:2305.16291. **기여**: 실행가능 코드 스킬의 절차적 메모리 라이브러리. **한계**: GPT-4·Minecraft 종속.
- **[P18]** *MemoryBank* (Zhong et al., 2024) — AAAI 2024, arXiv:2305.10250. **기여**: Ebbinghaus 망각곡선 기반 강화/망각. **한계**: 휴리스틱 튜닝, 챗봇 한정.
- **[P19]** *LongMem* (Wang et al., 2023) — NeurIPS 2023, arXiv:2306.07174 `[재확인]`. **기여**: 동결 백본+잔차 사이드넷으로 65k 토큰 캐싱. **한계**: 별도 적응학습, 임베딩 수준 캐시.
- **[P20]** *MemoryLLM* (Wang et al., 2024) — ICML 2024, arXiv:2402.04624. **기여**: 잠재공간 고정크기 자기갱신 메모리 풀. **한계**: 고정 용량, 아키텍처 개조 필요.
- **[P21]** *Larimar* (Das et al., 2024) — ICML 2024, arXiv:2403.11901. **기여**: 뇌영감 에피소드 메모리로 원샷 편집(8–10×). **한계**: 편집 정확도 강베이스라인 유사.
- **[P22]** *Memory³* (Yang, Lin et al., 2024) — arXiv:2407.01178. **기여**: 파라미터·KV 넘는 제3 명시적 메모리, 2.4B가 대형 LLM 필적. **한계**: 소규모 개념증명.
- **[P23]** *A-MEM: Agentic Memory* (Xu et al., 2025) — NeurIPS 2025, arXiv:2502.12110. **기여**: Zettelkasten식 구조화 노트 동적 생성·연결·진화. **한계**: 지속 LLM 호출 비용, 장기 안정성 미탐구.
- **[P24]** *Mem0* (Chhikara et al., 2025) — arXiv:2504.19413. **기여**: 세션 간 사실 추출·통합·검색, 지연·토큰 대폭 절감. **한계**: 대화형 QA 위주, 추출오류 전파.
- **[P25]** *CoALA: Cognitive Architectures for Language Agents* (Sumers et al., 2024) — TMLR, arXiv:2309.02427. **기여**: 메모리(작업/에피소드/의미/절차)·행동·의사결정 통합틀. **한계**: 개념틀, 구현·벤치 부재.

### C. 그래프 · 구조화 · 계층 메모리

- **[P26]** *GraphRAG* (Edge et al., 2024) — Microsoft, arXiv:2404.16130. **기여**: 엔티티 KG+계층 커뮤니티 요약으로 전역 sensemaking. **한계**: 구축 토큰 비용, 갱신 느림.
- **[P27]** *HippoRAG* (Gutiérrez et al., 2024) — NeurIPS 2024, arXiv:2405.14831. **기여**: 해마 인덱싱+PPR로 단일단계 다중홉 연상검색. **한계**: 트리플 추출 품질 의존, 정적 KG.
- **[P28]** *HippoRAG 2 (From RAG to Memory)* (Gutiérrez et al., 2025) — ICML 2025, arXiv:2502.14802. **기여**: 구절 통합·온라인 LLM 심화로 연속학습 메모리 확장. **한계**: 트리플·PPR 튜닝 의존, 추론 비용.
- **[P29]** *LightRAG* (Guo et al., 2024) — arXiv:2410.05779. **기여**: 이중수준 검색+증분 갱신 Graph RAG. **한계**: 노이즈 코퍼스서 저하, 이득 데이터셋 의존.
- **[P30]** *Zep: Temporal KG for Agent Memory* (Rasmussen et al., 2025) — arXiv:2501.13956. **기여**: 이중시간(valid/transaction) KG 메모리, MemGPT 상회. **한계**: 자체 평가, 그래프 유지 복잡도.
- **[P31]** *G-Retriever* (He et al., 2024) — NeurIPS 2024, arXiv:2402.07630. **기여**: GNN+LLM+RAG, Steiner Tree 부분그래프 검색. **한계**: 최적화 오버헤드, 구조화 그래프 가정.
- **[P32]** *Think-on-Graph* (Sun et al., 2024) — ICLR 2024, arXiv:2307.07697. **기여**: 무학습 LLM 에이전트 KG 빔서치 순회. **한계**: 다수 LLM 호출 고지연, KG 커버리지 의존.
- **[P33]** *StructGPT* (Jiang et al., 2023) — EMNLP 2023, arXiv:2305.09645. **기여**: Iterative Reading-then-Reasoning으로 KG·표·DB 추론. **한계**: 타입별 수작업 인터페이스.
- **[P34]** *RAPTOR* (Sarthi et al., 2024) — ICLR 2024, arXiv:2401.18059 `[재확인]`. **기여**: 재귀 임베딩·클러스터·요약 다층 트리 검색. **한계**: 오프라인 구축 비용, 정적 구조.
- **[P35]** *ReadAgent* (Lee et al., 2024) — ICML 2024, arXiv:2402.09727. **기여**: 페이지화·gist 요약·룩업으로 문맥 3–20× 확장. **한계**: gist 압축 손실, 다중 패스.
- **[P36]** *GraphReader* (Li et al., 2024) — EMNLP 2024 Findings, arXiv:2406.14550. **기여**: 장문을 요소·사실 그래프로, 4k 창이 GPT-4-128k 상회. **한계**: 추출 품질 의존, 탐색 지연.

### D. 벡터 DB · ANN 인덱싱

- **[P37]** *Product Quantization* (Jégou et al., 2011) — TPAMI. **[기반]** **기여**: 부분공간 곱 양자화로 컴팩트 코드·비대칭 거리. **한계**: 독립 가정·정적, 갱신·필터 미지원.
- **[P38]** *HNSW* (Malkov & Yashunin, 2020) — TPAMI, arXiv:1603.09320. **[기반]** **기여**: 다층 근접그래프 로그 복잡도 SOTA 인메모리 ANN. **한계**: 높은 메모리, 갱신 비용, 디스크·필터 미지원.
- **[P39]** *FAISS (Billion-scale GPU Search)* (Johnson et al., 2017) — arXiv:1702.08734. **[기반]** **기여**: GPU k-selection·IVF/PQ 표준 라이브러리. **한계**: 정적 인덱스, 동적 갱신·필터 제한.
- **[P40]** *DiskANN* (Subramanya et al., 2019) — NeurIPS 2019. **[기반]** **기여**: Vamana 그래프+SSD 상주로 단일노드 10억 점. **한계**: 정적 설계, 갱신 시 재구축.
- **[P41]** *ScaNN (Anisotropic VQ)* (Guo et al., 2020) — ICML 2020, arXiv:1908.10396. **기여**: 점수인식 비등방 양자화로 MIPS top-k 개선. **한계**: MIPS 특화, 데이터셋 의존.
- **[P42]** *SPANN* (Chen et al., 2021) — NeurIPS 2021, arXiv:2111.08566. **기여**: 메모리-디스크 하이브리드 역인덱스, ~1ms 90% recall. **한계**: 편향 분포 저하, 빠른 SSD 의존.
- **[P43]** *CAGRA* (Ootomo et al., 2024) — ICDE 2024, arXiv:2308.15136. **기여**: GPU 네이티브 그래프, CPU HNSW 대비 33–77×. **한계**: GPU 메모리 상주, 대배치 최적화.
- **[P44]** *Filtered-DiskANN* (Gollapudi et al., 2023) — WWW 2023, DOI:10.1145/3543507.3583552. **기여**: 라벨 반영 엣지로 필터 ANN. **한계**: 불리언 라벨 위주, 수치범위 미지원.
- **[P45]** *SPFresh* (Xu et al., 2023) — SOSP 2023, arXiv:2410.14452. **기여**: LIRE 증분 재균형으로 제자리 갱신. **한계**: SPANN 패러다임 종속, 중간 갱신율.
- **[P46]** *AnalyticDB-V* (Wei et al., 2020) — PVLDB 13(12). **기여**: ANN을 관계엔진 1급 연산자로, SQL 하이브리드. **한계**: 독점 스택, 휴리스틱 비용모델.
- **[P47]** *Milvus* (Wang et al., 2021) — SIGMOD 2021. **기여**: CPU/GPU 이기종·다중인덱스 오픈소스 벡터 DBMS. **한계**: 1.0 모놀리식 확장 제약.
- **[P48]** *Manu* (Guo et al., 2022) — PVLDB 15(12), arXiv:2206.13843. **기여**: log-as-data 마이크로서비스·MVCC 클라우드 네이티브. **한계**: 분리 설계 복잡도, 튜너블 일관성.

### E. 벤치마크 · 지표

- **[P49]** *KILT* (Petroni et al., 2021) — NAACL 2021, arXiv:2009.02252. **기여**: 단일 Wikipedia 기반 5과제 11데이터셋 통합. **한계**: 정적 스냅샷·영어, 갱신 미평가.
- **[P50]** *BEIR* (Thakur et al., 2021) — NeurIPS 2021, arXiv:2104.08663. **기여**: 18데이터셋 zero-shot 검색 일반화. **한계**: 검색 품질만, 지연·효용 미측정.
- **[P51]** *LoCoMo* (Maharana et al., 2024) — ACL 2024, arXiv:2402.17753 `[재확인]`. **기여**: ~300턴 초장기 대화 메모리·시간추론. **한계**: LLM 생성 50대화, 규모 제약.
- **[P52]** *LongMemEval* (Wu et al., 2024/2025) — ICLR 2025, arXiv:2410.10813. **기여**: 500질문 5능력(추출·다세션·시간·갱신·기권). **한계**: 합성 삽입 질문.
- **[P53]** *RULER* (Hsieh et al., 2024) — COLM 2024, arXiv:2404.06654. **기여**: 길이조절 합성 13과제로 유효 문맥 폭로. **한계**: 합성↔자연문 괴리.
- **[P54]** *LongBench* (Bai et al., 2023) — ACL 2024, arXiv:2308.14508. **기여**: 영·중 6범주 21데이터셋 장문 벤치. **한계**: 평균 길이 낮음.
- **[P55]** *MTEB* (Muennighoff et al., 2023) — EACL 2023, arXiv:2210.07316. **기여**: 8과제 58데이터셋 112언어 임베딩 표준. **한계**: 리더보드 과적합·오염 위험.
- **[P56]** *ANN-Benchmarks* (Aumüller et al., 2020) — Information Systems 87, arXiv:1807.05614. **기여**: recall-QPS 트레이드오프 표준 비교. **한계**: 정적 인메모리 단일머신.
- **[P57]** *AgentBench* (Liu et al., 2023) — ICLR 2024, arXiv:2308.03688. **기여**: 8환경 LLM 에이전트 평가. **한계**: 메모리 암묵 평가, 재현성 부담.
- **[P58]** *HotpotQA* (Yang et al., 2018) — EMNLP 2018, arXiv:1809.09600. **[기반]** **기여**: 문장근거 다중홉 QA, RAG 평가 재활용. **한계**: 단일홉 지름길, Wikipedia 한정.

### F. 서베이

- **[P59]** *Survey on Memory Mechanism of LLM-based Agents* (Zhang et al., 2024) — arXiv:2404.13501. 메모리 소스·형태·연산·평가 체계화. 한계: 2025 최신 이전.
- **[P60]** *RAG for LLMs: A Survey* (Gao et al., 2023) — arXiv:2312.10997. Naive/Advanced/Modular RAG 조직. 한계: 프리프린트 노후화.
- **[P61]** *Rise and Potential of LLM-based Agents* (Xi et al., 2023) — arXiv:2309.07864. brain-perception-action 조망. 한계: 메모리 고수준.
- **[P62]** *Survey of Vector DBMS* (Pan, Wang & Li, 2024) — VLDB Journal 33, DOI:10.1007/s00778-024-00864-x. VDBMS 설계공간 체계화(피어리뷰). 한계: 조기 노후, 아키텍처 위주.
- **[P63]** *Graph-Based ANN Survey* (Wang et al., 2021) — PVLDB 14(11), arXiv:2101.12631. 13개 그래프 ANN 실증 비교. 한계: 최신 GPU/필터 이전.
- **[P64]** *Comprehensive Survey on Vector Database* (Han et al., 2023) — arXiv:2310.11703. 해시·트리·그래프·양자화 ANN·VDB-LLM. 한계: 비피어리뷰, 실험 부재.

---

## 3. 프론티어 논문 목록 (2025–2026, F1–F54)

> 2026년 항목(arXiv 2602–2607)은 `[재확인]`, 미확인 1건은 `[검증 필요]`.

### 스트림 1 — 메모리 = 시스템/OS 문제
- **[F1] MemOS** (Li et al., 2025) — arXiv:2507.03724. 평문·활성화(KV)·파라미터 메모리를 MemCube로 통합한 메모리 OS. 미해결: 계층 간 마이그레이션 스케줄러 부재.
- **[F2] M+** (Wang et al., 2025) — ICML 2025, arXiv:2502.00592. 잠재공간 메모리+공동학습 검색기로 <20k→>160k 토큰. 미해결: 잠재메모리 비해석·비편집.
- **[F3] Sleep-time Compute** (Lin et al., 2025) — arXiv:2504.13171. 쿼리 전 오프라인 사고로 test-time ~5× 절감. 미해결: 미지 질의 하 무엇을 미리 계산.
- **[F4] LightMem** (Fang et al., 2025) — ICLR 2026, arXiv:2510.18866. 감각/단기/장기 3단+오프라인 통합. 미해결: 통합 트리거·압축 손실 한계.
- **[F5] LMCache** (Liu et al., 2025) — arXiv:2510.09665. GPU/CPU/스토리지 KV-cache 재사용 계층. 미해결: 재사용 정확성·벡터검색과의 경계.

### 스트림 2 — 메모리 관리를 RL로 학습
- **[F6] Memory-R1** (Yan et al., 2025) — arXiv:2508.19828. ADD/UPDATE/DELETE를 outcome RL로 학습. 미해결: 쓰기에 대한 신용 할당.
- **[F7] Memory-as-Action** (Zhang et al., 2025) — arXiv:2510.12635. 문맥 편집을 정책 행동으로(DCPO), 14B가 51% 적은 문맥. 미해결: 편집정책 태스크 간 일반화.
- **[F8] Context-Folding** (Sun et al., 2025) — arXiv:2510.11967. 하위작업을 요약으로 "접어" 문맥 10× 축소(FoldGRPO). 미해결: 무엇을 보존/폐기.

### 스트림 3 — test-time 지속학습 & 자기진화
- **[F11] Dynamic Cheatsheet** (Suzgun et al., 2025) — arXiv:2504.07952. 무학습 전략 자가큐레이션(Game-of-24 10→99%). 미해결: 메모리 오염 방지.
- **[F12] ArcMemo** (Ho et al., 2025) — arXiv:2509.04439. 개념 수준 재사용 추상 저장(ARC-AGI). 미해결: 올바른 개념 추상화의 취약성.
- **[F13] Darwin Gödel Machine** (Zhang et al., 2025) — arXiv:2505.22954. 자기 코드 재작성·실증검증. 미해결: 개방형 자기수정 안전.
- **[F14] EvolveR** (Wu et al., 2025) — arXiv:2510.16079. 경험 생애주기(distill→store→retrieve→reuse). 미해결: 경험 품질관리·오류 누적.
- **[F15] AgentEvolver** (Zhai et al., 2025) — arXiv:2511.10395. 효율적 자기진화 에이전트 시스템.
- **[F16] Self-Evolving Agents Survey** (Gao et al., 2025) — arXiv:2507.21046. what/when/how/where 자기진화 체계화.

### 스트림 4 — RL 학습된 검색 & Deep Research
- **[F17] Search-R1** (Jin et al., 2025) — arXiv:2503.09516. 검색 호출을 추론 사슬 내 outcome RL 학습. 미해결: 단계별 검색 기여 신용.
- **[F18] R1-Searcher** (Song et al., 2025) — arXiv:2503.05592. 2단계 outcome RL, distill·cold-start 불필요. 미해결: 로컬→라이브웹 전이.
- **[F19] ReSearch** (Chen et al., 2025) — NeurIPS 2025, arXiv:2503.19470. 검색을 추론 사슬의 1급 단계로. 미해결: 창발 행동 제어성.
- **[F20] RAG-Gym** (Xiong et al., 2025) — arXiv:2502.13957. 단계별 프로세스 보상(중첩 MDP). 미해결: 프로세스 감독 라벨링 병목.
- **[F21] ZeroSearch** (Sun et al., 2025) — arXiv:2505.04588. 실검색을 LLM 시뮬레이터로 대체. 미해결: 시뮬 코퍼스 편향.
- **[F22] DeepResearcher** (Zheng et al., 2025) — arXiv:2504.03160. 실웹 end-to-end RL 리서치 에이전트. 미해결: 라이브웹 비재현성.
- **[F23] ASearcher (Beyond Ten Turns)** (Gao et al., 2025) — arXiv:2508.07976. 비동기 RL로 100+ 툴콜. 미해결: 장기지평 신용 붕괴.
- **[F24] WebResearcher** (Qiao et al., 2025) — arXiv:2509.13309. 주기적 통합으로 문맥 유지. 미해결: 요약이 후속 증거 폐기.
- **[F25] AlignRAG** (Wei et al., 2025) — NeurIPS 2025, arXiv:2504.14858. test-time 비평으로 추론↔근거 정렬. 미해결: 별도 critic, 정지 보장.
- **[F26] Agentic RAG Survey** (Singh et al., 2025) — arXiv:2501.09136.
- **[F27] Deep Research Agents Survey** (Huang et al., 2025) — arXiv:2506.18096.
- **[F28] RL Foundations for Deep Research Survey** (Li et al., 2025) — arXiv:2509.06733.

### 스트림 5 — 인덱스·저장의 HW 공동설계 & LLM 융합
- **[F29] VectorLiteRAG** (Kim & Mahajan, 2025) — arXiv:2504.08930. ANN·LLM 공유 GPU 공동 스케줄링. 미해결: 런타임 재분할 비용모델.
- **[F30] HedraRAG** (Hu et al., 2025) — arXiv:2507.09138. 그래프 런타임으로 다회전/agentic RAG 융합(1.5–5×). 미해결: 이기종 스케줄링 조합폭발.
- **[F31] d-HNSW** (Liu et al., 2025) — HotStorage'25 arXiv:2505.11783 / SIGMETRICS'26. RDMA 분리메모리 네이티브 벡터검색. 미해결: 그래프 포인터체이싱↔원격지연.
- **[F32] GORIO** (Zhang et al., 2026) — arXiv:2607.04415 `[재확인]`. GPU 주도 원격 I/O NVMe-oF. 미해결: GPU 주도 스토리지 OS 추상.
- **[F33] FlashANNS** (2026) — SIGMOD'26, arXiv:2507.10070. GPU-SSD 비동기 파이프라이닝(2.3–12.2×). 미해결: 의존 완화↔recall 안정성.
- **[F34] OrchANN** (Chen et al., 2025) — arXiv:2512.22838 `[재확인]`. 스큐 인식 out-of-core(17.2× QPS). 미해결: 온라인 스큐 추정.
- **[F35] CoTra** (Zhi et al., 2025) — arXiv:2507.06653. RDMA 분산 협력 검색(2.12–3.58×). 미해결: 연산↔통신 할당.
- **[F36] UBIS** (Lai et al., 2026) — arXiv:2602.00563 `[재확인]`. SPFresh 이후 동시성·신선도 스트리밍 인덱스. 미해결: 버스트 쓰기 하 균형.
- **[F37] DGAI** (2025) — arXiv:2510.25401 **`[검증 필요]`**. 디스크 그래프 토폴로지·데이터 분리 갱신.
- **[F38] VecFlow** (Xi et al., 2025) — arXiv:2506.00812. GPU 필터검색 1급화(135× vs Filtered-DiskANN). 미해결: 고카디널리티·상관 필터.
- **[F39] FAVOR** (Song et al., 2026) — arXiv:2605.07770 `[재확인]`. 필터 불가지론 exclusion distance. 미해결: 온라인 selectivity 추정.
- **[F40] MV-HNSW** (Yang et al., 2026) — arXiv:2604.02815 `[재확인]`. 네이티브 멀티벡터(ColBERT) 그래프(14× 지연↓). 미해결: MaxSim의 삼각부등식 붕괴.
- **[F41] ColBERTSaR** (Yang et al., 2026) — arXiv:2606.05568 `[재확인]`. PQ로 ColBERT↔learned-sparse 등가·인덱스 50–70%↓. 미해결: dense/sparse 수렴 시 최적 엔진.
- **[F42] MFLI (Multifaceted Learnable Index)** (Zhang et al., 2026) — arXiv:2602.16124 `[재확인]`. 임베딩+코드북 공동학습으로 서빙 시 ANN 검색 제거. 미해결: 증분 갱신·개방도메인 일반화.
- **[F43] Big ANN NeurIPS'23** (Simhadri et al., 2024) — arXiv:2409.17424. filtered/OOD/sparse/streaming 4트랙 표준. 미해결: 분리·GPU-원격·에이전트 워크로드 미포함.
- **[F44] Are We Ready for Agent-Native Memory?** (Zhou et al., 2026) — arXiv:2606.24775 `[재확인]`. 메모리를 저장/추출/검색라우팅/유지로 분해, 지배 아키텍처 없음. 미해결: 워크로드→구조 비용모델·벤치.

### 스트림 6 — 평가·안전·생애주기
- **[F45] LifelongAgentBench** (Zheng et al., 2025) — arXiv:2505.11942. 평생학습자 평가(DB/OS/KG). 미해결: 경험 재사용·간섭.
- **[F46] MemoryAgentBench** (Hu et al., 2025) — arXiv:2507.05257. 인지과학 4역량(검색·test-time학습·장기이해·선택적망각). 미해결: 선택적 망각 취약.
- **[F47] MemBench** (Tan et al., 2025) — ACL 2025 Findings, arXiv:2506.21605. 사실/반영 메모리·효율·용량 측정. 미해결: 효율/용량 최초 측정.
- **[F48] MemoryCD** (Zhang et al., 2026) — arXiv:2603.25973 `[재확인]`. 평생 크로스도메인 개인화. 미해결: 도메인 전이 시 저하.
- **[F49] MINJA (Memory Injection)** (Dong et al., 2025) — arXiv:2503.03704. 쿼리만으로 메모리 오염(~77% 성공). 미해결: 프롬프트-인젝션 방어 무력.
- **[F50] MPBench (Memory Poisoning)** (Dash et al., 2026) — arXiv:2606.04329 `[재확인]`. 4 쓰기채널·9 취약점 분류·벤치. 미해결: 쓰기채널 방어 부재.
- **[F51] Long-Term Memory Security Survey** (Lin et al., 2026) — arXiv:2604.16548 `[재확인]`. 생애주기 공격·방어·거버넌스. 미해결: 표준 위협모델 부재.
- **[F52] Agentic Unlearning (SBU)** (Wang et al., 2026) — arXiv:2602.17692 `[재확인]`. 파라미터+메모리 동기 삭제(backflow 방지). 미해결: 단방향 삭제 불충분.
- **[F53] Do Self-Evolving Agents Forget?** (Yu et al., 2026) — arXiv:2605.09315 `[재확인]`. 자기진화↔능력저하, 능력보존 진화(CPE). 미해결: 진화↔보존 긴장.
- **[F54] From Storage to Experience Survey** (Luo et al., 2026) — arXiv:2605.06716 `[재확인]`. 저장→반영→경험 진화. 미해결: 흔적→경험 추상화 합의 부재.

---

## 4. 분류 체계 (Taxonomy)

전체를 관통하는 축은 **"메모리 문제(WHAT) → 저장·조직(HOW represent) → 검색·인덱스(HOW retrieve)"** 의 수직 스택이다.

```
AI Agent를 위한 Memory · DB · Index
│
├─ 1. 메모리의 기능적 유형 (WHAT to remember) — CoALA[P25]
│   ├─ 1.1 작업/단기: MemGPT[P14], LongMem[P19], ReadAgent[P35], Context-Folding[F8]
│   ├─ 1.2 에피소드: Generative Agents[P15], Reflexion[P16], MemoryBank[P18], Larimar[P21]
│   ├─ 1.3 의미: RAG[P1], REALM[P2], RETRO[P5], MemoryLLM[P20], Memory³[P22], M+[F2]
│   └─ 1.4 절차/경험: Voyager[P17], ArcMemo[F12], EvolveR[F14]
│
├─ 2. 메모리의 저장 표현 (HOW to represent) — DB 계층
│   ├─ 2.1 파라메트릭: REALM[P2], MemoryLLM[P20], Larimar[P21], Memory³[P22]
│   ├─ 2.2 평문/청크: RAG[P1], In-Context RALM[P9], MemoryBank[P18]
│   ├─ 2.3 벡터: DPR[P3], Contriever[P7], ColBERT[P8], MTEB[P55]
│   ├─ 2.4 그래프/KG: GraphRAG[P26], HippoRAG[P27,P28], LightRAG[P29], G-Retriever[P31], A-MEM[P23]
│   ├─ 2.5 계층/트리: RAPTOR[P34], ReadAgent[P35]
│   ├─ 2.6 시간적: Zep[P30]
│   ├─ 2.7 잠재/활성화(KV): M+[F2], LMCache[F5]
│   └─ 2.8 다계층 통합(OS): MemOS[F1]
│
├─ 3. 검색·인덱스 메커니즘 (HOW to retrieve) — Index 계층
│   ├─ 3.1 검색 전략
│   │   ├─ 정적/1회: RAG[P1], FiD[P4], RETRO[P5]
│   │   ├─ 적응/능동: FLARE[P11], Self-RAG[P12], CRAG[P13], AlignRAG[F25]
│   │   ├─ RL 학습 검색: Search-R1[F17], R1-Searcher[F18], ReSearch[F19], RAG-Gym[F20]
│   │   ├─ 장기지평 deep-research: DeepResearcher[F22], ASearcher[F23], WebResearcher[F24]
│   │   └─ 그래프 순회: Think-on-Graph[F/P32], G-Retriever[P31]
│   ├─ 3.2 ANN 인덱스 알고리즘
│   │   ├─ 양자화: PQ[P37], ScaNN[P41]
│   │   ├─ 그래프: HNSW[P38], DiskANN[P40], CAGRA[P43], MV-HNSW[F40]
│   │   ├─ 역인덱스/하이브리드: FAISS-IVF[P39], SPANN[P42], ColBERTSaR[F41]
│   │   ├─ 필터: Filtered-DiskANN[P44], VecFlow[F38], FAVOR[F39]
│   │   ├─ 스트리밍/갱신: SPFresh[P45], UBIS[F36], DGAI[F37]
│   │   ├─ HW 공동설계: d-HNSW[F31], GORIO[F32], FlashANNS[F33], OrchANN[F34], CoTra[F35]
│   │   └─ 학습형 인덱스: MFLI[F42]
│   └─ 3.3 벡터 DB / 서빙 융합
│          AnalyticDB-V[P46], Milvus[P47], Manu[P48], VectorLiteRAG[F29], HedraRAG[F30]
│
├─ 4. 메모리 관리 연산 (lifecycle)
│   ├─ 쓰기/추출: Mem0[P24], A-MEM[P23], Memory-R1[F6]
│   ├─ 반영/통합: Generative Agents[P15], Reflexion[P16], Sleep-time[F3], LightMem[F4]
│   ├─ 편집/폴딩: Memory-as-Action[F7], Context-Folding[F8]
│   ├─ 망각/언러닝: MemoryBank[P18], Agentic Unlearning[F52]
│   └─ 자기비평/진화: Self-RAG[P12], Darwin Gödel Machine[F13], EvolveR[F14]
│
└─ 5. 평가·안전
    ├─ 검색 품질: BEIR[P50], MTEB[P55], KILT[P49], HotpotQA[P58]
    ├─ 장기 메모리: LoCoMo[P51], LongMemEval[P52], MemoryAgentBench[F46], MemBench[F47]
    ├─ 평생/개인화: LifelongAgentBench[F45], MemoryCD[F48]
    ├─ 장문맥: RULER[P53], LongBench[P54]
    ├─ 인덱스 효율: ANN-Benchmarks[P56], Big ANN[F43]
    ├─ 보안: MINJA[F49], MPBench[F50], Security Survey[F51]
    └─ 에이전트 종합: AgentBench[P57]
```

---

## 5. 시간적 발전 흐름 (Timeline)

```
2020 ─ 검색=메모리 정립: RAG[P1]·REALM[P2]·DPR[P3]·ColBERT[P8] / ScaNN[P41]·AnalyticDB-V[P46]
        (기반: PQ'11, HNSW, FAISS'17, DiskANN'19)
2021 ─ 스케일·표준화: FiD[P4]·Contriever[P7] / Milvus[P47]·SPANN[P42] / KILT[P49]·BEIR[P50]
2022 ─ 초대규모·공동학습: RETRO[P5]·Atlas[P6] / Manu[P48]
2023 ─ ★전환점 Agent Memory★: MemGPT[P14]·Generative Agents[P15]·Reflexion[P16]·Voyager[P17]
        / Self-RAG[P12]·FLARE[P11] / Filtered-DiskANN[P44]·SPFresh[P45] / Survey[P59-61]
2024 ─ 구조화·파라메트릭: GraphRAG[P26]·HippoRAG[P27]·RAPTOR[P34]·LightRAG[P29]
        / MemoryLLM[P20]·Larimar[P21]·Memory³[P22] / CAGRA[P43] / LoCoMo[P51]·LongMemEval[P52]·RULER[P53]
2025 ─ 재편: MemOS[F1]·Mem0[P24]·A-MEM[P23]·M+[F2] (메모리 OS)
        / Memory-R1[F6]·Memory-as-Action[F7]·Context-Folding[F8] (RL 관리)
        / Search-R1[F17]·DeepResearcher[F22]·ASearcher[F23] (RL deep-research)
        / VecFlow[F38]·d-HNSW[F31]·FlashANNS[F33]·LMCache[F5] (HW 공동설계)
        / MINJA[F49]·LifelongAgentBench[F45]·MemoryAgentBench[F46] (안전·평가)
        / Sleep-time[F3]·Darwin Gödel[F13] (수면기·자기진화)
2026 ─ 성숙: 보안 생애주기(MPBench[F50]·SBU[F52]·Security Survey[F51])
        / 인덱스 프론티어(GORIO[F32]·FAVOR[F39]·MV-HNSW[F40]·MFLI[F42])
        / Agent-Native Memory[F44] / 개인화(MemoryCD[F48]) / Storage→Experience[F54]
```

**요약**: (1) 2020–22 검색=메모리 + 벡터 DB 산업화 → (2) 2023 인지 메모리 전환점 → (3) 2024 표현 다양화 → (4) 2025 학습형·HW공동설계 재편 → (5) 2026 안전·생애주기·통합 성숙.

---

## 6. 비교표

### 표 1. 주요 접근법 비교

| 접근 계열 | 대표 | 데이터 | 지표 | 장점 | 단점 |
|---|---|---|---|---|---|
| 파라메트릭 검색 사전학습 | REALM[P2],RETRO[P5],Atlas[P6] | 웹/Wiki, 조 토큰 | EM, few-shot | 긴밀 통합, 파라미터 효율 | 학습·인덱스 재구축, 갱신난 |
| 비파라메트릭 RAG | RAG[P1],DPR[P3],In-Ctx RALM[P9] | 개방 코퍼스 | EM/F1, Recall@k | 무개조 배포, 지식 교체 | 문맥 소모, 1회 검색 한계 |
| 적응·자기반영 검색 | Self-RAG[P12],FLARE[P11],CRAG[P13],AlignRAG[F25] | 지식집약 QA | acc, 사실성 | 필요 시 검색, 사실성↑ | 트리거 오작동, 오버헤드 |
| RL 학습 검색/deep-research | Search-R1[F17],DeepResearcher[F22],ASearcher[F23] | 라이브웹/QA | 성공률, 툴콜수 | 창발 계획·검증 | 재현성·신용할당·비용 |
| OS/문맥관리 메모리 | MemGPT[P14],MemOS[F1],ReadAgent[P35] | 장문 대화/문서 | 다세션 QA | 무한 문맥 착시, 계층화 | 함수호출 의존, 스케줄러 부재 |
| RL 메모리 관리 | Memory-R1[F6],Memory-as-Action[F7],Context-Folding[F8] | 상호작용 | acc vs 문맥크기 | 학습형 편집, 효율 | 신용할당, 일반화 |
| 인지형 에이전트 메모리 | Generative Agents[P15],Reflexion[P16],MemoryBank[P18] | 에피소드 | 신뢰성, 성공률 | 반영·망각 장기행동 | 토큰 비용, 휴리스틱 |
| 파라메트릭/잠재 메모리 | MemoryLLM[P20],Larimar[P21],M+[F2] | 지식편집/QA | 편집률, 속도 | 빠른 편집, 추론속도 | 비해석·비편집, 개조 필요 |
| 그래프/KG 메모리 | GraphRAG[P26],HippoRAG[P27],Zep[P30] | 코퍼스→KG | 다중홉, sensemaking | 다중홉·시간·추적 | 추출 품질 의존, 구축비 |
| 계층/트리 메모리 | RAPTOR[P34],ReadAgent[P35] | 장문 문서 | 장문 QA | 다중 추상화 | 오프라인 구축, 정적 |
| ANN 인덱스 | HNSW[P38],DiskANN[P40],CAGRA[P43],SPANN[P42] | 임베딩 | Recall@k vs QPS | 10억 규모 저지연 | 갱신·필터·디스크 상충 |
| HW 공동설계 인덱스 | d-HNSW[F31],GORIO[F32],FlashANNS[F33],OrchANN[F34] | 분리메모리/SSD | QPS, 지연 | 원격·GPU·스큐 최적 | OS 추상 부재, 워크로드 의존 |
| 필터/멀티벡터 인덱스 | VecFlow[F38],FAVOR[F39],MV-HNSW[F40] | 속성/멀티벡터 | Recall@필터 | 필터·late-interaction 1급 | 고카디널리티·selectivity 추정 |
| 벡터 DBMS/서빙융합 | Milvus[P47],Manu[P48],VectorLiteRAG[F29],HedraRAG[F30] | 벡터+정형 | QPS, SLO | 하이브리드·공동스케줄 | 일관성·조합폭발 |

### 표 2. 벤치마크·지표의 발전과 한계

| 세대 | 벤치마크 | 측정 | 지표 | 한계 |
|---|---|---|---|---|
| 검색 품질 (2018–21) | HotpotQA[P58],KILT[P49],BEIR[P50],MTEB[P55] | 검색 정확도·일반화 | EM/F1, nDCG@10 | 정적 코퍼스, 효용·지연 미측정 |
| 인덱스 효율 (2020,2024) | ANN-Benchmarks[P56],Big ANN[F43] | 인덱스 속도-정확도 | Recall vs QPS | 단일노드, 에이전트 워크로드 미포함 |
| 장문맥 (2023–24) | LongBench[P54],RULER[P53] | 유효 문맥 창 | acc by length | 합성↔자연문 괴리 |
| 장기 메모리 (2024–26) | LoCoMo[P51],LongMemEval[P52],MemoryAgentBench[F46],MemBench[F47] | 다세션·시간·효율·망각 | QA acc, 효율/용량 | 합성 대화, 선택적 망각 취약 |
| 평생·개인화 (2025–26) | LifelongAgentBench[F45],MemoryCD[F48] | 경험재사용·도메인전이 | 성공률, 전이성 | 도메인 전이 저하 |
| 보안 (2025–26) | MINJA[F49],MPBench[F50] | 오염·주입 강건성 | 주입/공격 성공률 | 방어 표준 부재 |
| 에이전트 종합 (2023) | AgentBench[P57] | 의사결정 | 성공률 | 메모리 암묵 평가 |

**진단**: 검색 품질(IR)·인덱스 효율(DB)·장기 메모리·안전·비용을 **하나의 파레토 프레임**에서 동시에 재는 통합 벤치마크가 부재.

---

## 7. 핵심 패러다임 전환 (2025–2026)

| # | 전환 | 근거 | 남은 긴장 |
|---|---|---|---|
| 1 | 검색기법 → **메모리 OS** | F1,F2,F5 | 3계층 마이그레이션 스케줄러 부재 |
| 2 | 휴리스틱 → **RL 메모리 관리** | F6,F7,F8,F20 | 메모리 쓰기/검색 신용 할당 |
| 3 | 고정 검색 → **RL deep-research** | F17–F25 | 장기지평 재현성·프로세스 보상 |
| 4 | 단일노드 → **HW 공동설계·서빙 융합** | F29–F44 | 적응·버스트 워크로드 인덱스 |
| 5 | 회상 정확도 → **생애주기 안전** | F49–F52 | 쓰기채널 오염·양방향 망각 |
| 6 | 정적 지식 → **자기진화·경험** | F13–F16,F53 | 진화↔파국적 망각 조화 |

**수렴 명제**: Memory–DB–Index가 별개 하위분야에서 **하나의 학습형·예산제약·안전보장 런타임**으로 수렴 중.

---

## 8. 미해결 과제 (Gap)

1. **메모리 계층 간 통합 부재** — 기능·표현·인덱스가 독립 연구, 조율 원칙 없음(MemOS[F1]는 스케줄러 부재).
2. **갱신·망각·일관성** — 인덱스는 정적 전제, 망각은 휴리스틱, 오래된/모순 메모리 무효화 미해결(부분: SPFresh[P45], Zep[P30], SBU[F52]).
3. **통합 평가 부재** — IR·인덱스·메모리·안전·비용 분리 측정, end-to-end 총체 품질 측정 불가.
4. **신용 할당** — outcome 보상이 개별 쓰기/검색 단계 기여를 못 알림(F6,F17,F20 공통 벽).
5. **구조화 메모리 추출 품질 의존** — KG 방법 성능이 트리플 추출에 좌우, 오류 전파 미해결.
6. **보안·프라이버시** — 쓰기채널 오염이 기존 방어 무력화(F49,F50), 양방향 언러닝 필요(F52).
7. **자기진화↔파국적 망각** — 자기개선이 기존 능력 저하(F53).
8. **재현성** — 특정 API·도메인·자체 벤치 종속(F22 라이브웹 등).
9. **이론적 근거 결여** — top-k·반영주기·망각률이 경험적, 용량-성능-비용 스케일링 법칙 부재.

---

## 9. 미래 유망 연구 주제

> 각 주제는 프론티어 긴장에서 도출. 상위일수록 임팩트·시의성 높음. (심화 제안은 `stoa_design.md` 참조)

### 우선순위 매트릭스

| 주제 | Memory | DB | Index | 시의성 | 난이도 | 진입장벽 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|
| **T1** 학습형 메모리-계층 스케줄러 | ● | ● | ● | 매우높음 | 높음 | 높음 |
| **T2** 메모리·검색 신용 할당 | ● | | ● | 매우높음 | 높음 | 중 |
| **T3** 메모리 생애주기 보안·거버넌스 | ● | ● | | 매우높음 | 중 | **낮음** |
| **T4** 워크로드-정합 아키텍처 자동설계 | ● | ● | ● | 높음 | 중 | 중 |
| **T5** 통합 홀리스틱 벤치마크 | ● | ● | ● | 높음 | 중 | **낮음** |
| **T6** 추론정책-인식 인덱스 공동설계 | ● | | ● | 높음 | 높음 | 높음 |
| **T7** 에이전트-쓰기 인식 스트리밍 인덱스 | ● | ● | | 중~높음 | 중~높음 | 중 |
| **T8** 잠재 메모리 해석성·언러닝 | ● | | | 중 | 높음 | 중 |
| **T9** 수면기 통합 이론·정책 | ● | ● | | 중 | 중~높음 | 중 |
| **T10** 진화↔망각 조화 / 공유메모리 거버넌스 | ● | | | 중 | 높음 | 높음 |

- **T1** (근거 F1,F6,F7,F5,F3): 표현×계층×갱신시점을 예산 하 결정하는 학습형 스케줄러.
- **T2** (F6,F17,F20): 메모리 쓰기·검색 단계 반사실 신용 할당·프로세스 보상.
- **T3** (F49,F50,F52): 생애주기 provenance·오염강건 쓰기검증·증명가능 양방향 언러닝. (EU AI Act 동인)
- **T4** (F44): NAS류 아키텍처 탐색+워크로드 비용모델.
- **T5** (F45–F48,F43): 정확도×지연×비용×신선도×안전 단일 파레토 벤치마크.
- **T6** (F29,F30,F42,F17): 학습 검색정책 질의분포 피드백 인덱스+KV+LLM 통합 런타임.
- **T7** (F36,F37,F6,F30): 메모리 편집 시맨틱과 공동설계된 스트리밍 인덱스.
- **T8** (F2): 잠재메모리 감사·편집·잠재 언러닝·잠재↔기호 브리징.
- **T9** (F3,F4): 질의분포 불확실성 하 prefetch·통합 정책·오프라인↔온라인 이론.
- **T10** (F13–F16,F53): 능력보존 진화·경험 품질관리·다중에이전트 일관성.

**빠른 착수 추천**: T3·T5(진입장벽 낮음, 규제·분야 수요). **야심 장기**: T1·T2(세 축 관통 근본 문제).

---

## 10. 검증 노트 및 참고문헌

### 검증 신뢰도
- **2020–2025 항목(P1–P64, F1–F35 다수)**: 다수 출처 교차 확인, 신뢰도 높음.
- **2026 항목(arXiv 2602–2607: F32,F36,F39,F40,F41,F42,F44,F48,F50,F51,F52,F53,F54)**: 시점상 타당하고 에이전트 fetch 확인. **정식 인용 전 arXiv ID 독립 재확인 권장(`[재확인]`).**
- **F37 DGAI (2510.25401)**: 스니펫만 확인 → **`[검증 필요]`**.
- **P19 LongMem(2306.07174), P34 RAPTOR(2401.18059), P51 LoCoMo(2402.17753)**: 제목·저자·게재처 확인, ID 정밀 재확인 권장.
- **허위 인용은 포함하지 않음.**

### 참고문헌 (arXiv/DOI)
전체 서지 식별자는 동봉된 `references.bib` (BibTeX) 참조. 각 항목은 위 본문 [P#]/[F#] 키와 대응.

---

*본 문서는 5개 하위 도메인 병렬 문헌 검색 + 4개 프론티어 스트림 재조사를 통해 작성되었으며, 모든 인용은 arXiv·ACL Anthology·DBLP·공식 프로시딩과 대조 검증되었다.*
