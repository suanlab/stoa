# 프로젝트명 선정 기록 — 왜 "STOA"인가

> 작성일 2026-07-15. 방법: 각 후보를 arXiv(cs.AI/CL/LG/DB/DC)·GitHub·PyPI/npm·주요 제품/컨퍼런스에 대조하는
> 병렬 웹 조사(도메인 우선순위: ① AI 에이전트/메모리/LLM/DB → ② GitHub/제품 → ③ 트레이드마크).
> 이 조사는 "사용 중 여부(usage-in-the-wild)" 확인이며, 정식 상표 등록 여부(USPTO/EUIPO)는 별도 확인 필요.

## 최종 판정 (8종)

| 후보 | AI-메모리 도메인 판정 | 결정적 충돌 | 종합 |
|---|---|---|---|
| **STOA** | 🟢 **클리어** | 동명 시스템·논문 없음 | ✅ **채택** |
| MNEME | 🟡 학술 무충돌 | Mneme HQ / Mneme AI (상업) | 차선 |
| STRATA | 🟠 근접 충돌 | arXiv "Strata: Hierarchical Context Caching for LLM Serving" (T1 KV계층화와 동일 영역) · Klavis "Strata" MCP · O'Reilly Strata 컨퍼런스 | 위험 |
| WARDEN | 🟠 인접 충돌 | ACL 2024 "WARDEN" 논문 · AI 에이전트 거버넌스 repo 5+ · Warden Protocol(블록체인 "AI agent OS") | 위험 |
| MAESTRO | 🔴 심각 | arXiv 9편+·GitHub 7개+ 에이전트 오케스트레이션 · Netflix/IBM/Google Magenta | 탈락 |
| ENGRAM | 🔴 심각 | arXiv "ENGRAM"(대화 에이전트 메모리, 동일 스코프) · DeepSeek "Engram" · $98M 스타트업 · repo 10+ | 탈락 |
| MNEMOSYNE | 🔴 심각 | arXiv 2편(엣지 LLM 메모리·초장문 서빙) · AI 메모리 repo 5+ · 20년 된 동명 프로젝트 | 탈락 |
| TESSERA | 🔴 직접 충돌 | MCP 네이티브 에이전트 메모리 툴 2+(동일 카테고리) · CVPR 2026 "TESSERA" · 반도체/바이오 등록상표 | 탈락 |
| CAIRN | 🔴 충돌 | 동명 메모리 툴 클러스터(cairn-mcp·agentcairn·smcady/Cairn·cairn.ink) · arXiv 2607.06534 · cairn.info | 탈락 |

## STOA 채택 근거
- AI 에이전트-메모리/RAG/벡터DB 도메인에서 **동명 시스템·논문 없음** (조사 8종 중 유일).
- 의미 정합: 주랑(기억의 궁전 건축) + 스토아 어원, 백로님 *Storage & Tiered Orchestration for Agents*.

## 유의점 (STOA)
1. **인접 소프트 충돌** — `EdanToledo/Stoa` (JAX RL 환경 인터페이스, 2025–26 활발). 우리 니치 아님, 검색 시 노출.
2. **네임스페이스 선점** — 맨 `stoa`는 npm/PyPI 구형 패키지 존재 → 배포는 `stoa-mem` 등 한정 네임스페이스 권장.
3. **상업 브랜드 노이즈** — withstoa.com(AI 미팅툴), stoa.ae(HR), EU 의회 STOA 기구(1987~). 직접 경쟁자는 아님.
4. **상용화 시** USPTO/EUIPO Class 9/42 정식 상표조사 필요.

## 교훈
짧고 매력적인 사전/신화 단어(Mnemosyne·Engram·Maestro·Tessera·Cairn)는 2025–26년 MCP·에이전트-메모리 툴
물결에 급속 선점되고 있다. 완전한 백지가 필요하면 조어(portmanteau)가 사실상 유일하게 안전하다.
