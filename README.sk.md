# 15 · P7: RAG aplikácia nad klinickými štúdiami (FastAPI + React)

🇬🇧 [English version](README.md)

> **Stav: rozpracované.** Kód, testy, UI aj eval sú hotové. Zatiaľ sú namerané iba čísla vyhľadávania cez BM25
> (cloudové CPU). Dense, hybridné vyhľadávanie, reranking a všetky čísla o kvalite odpovedí ešte čakajú na beh na
> Macu, lebo cloudová session nevedela stiahnuť embedding modely. Dovtedy tu nie je doplnené žiadne číslo, ktoré
> nebolo naozaj namerané.

## TL;DR
- Full-stack RAG aplikácia: FastAPI backend, ktorý streamuje odpovede cez Server-Sent Events, React + TypeScript UI
  s klikateľnými citáciami a panel zdrojov, ktorý ukazuje skóre každej fázy pre každý vyhľadaný chunk.
- Korpus: 396 ukončených intervenčných štúdií z 8 terapeutických oblastí (ClinicalTrials.gov API v2), 694 tisíc slov.
- Eval: 156 otázok zo štruktúrovaných polí (126 zodpovedateľných s presnou gold hodnotou, 30 nezodpovedateľných).
- Zatiaľ namerané (BM25, cloudové CPU): **chunky podľa sekcií dostanú chunk s odpoveďou do top 5 pri 90,5 % otázok,
  naivné chunky pevnej dĺžky pri 77,0 %.**
- Jedno rozhranie pre poskytovateľov LLM: Ollama (predvolená, 0 €), lokálny Hugging Face model, AWS Bedrock alebo
  Azure OpenAI, volí sa v `.env`. Bedrock a Azure sú pripravené pre kroky 18 a 19, tu sa nespúšťajú.

## Problém
Tímy v klinickom R&D sa pýtajú napríklad „koľko pacientov zaradila štúdia X?“ alebo „aké zaslepenie použila?“.
Samotné LLM to buď nevie, alebo si sebavedome vymyslí odpoveď. RAG najprv vyhľadá relevantné záznamy a model
odpovedá z nich, s citáciou, ktorú vie človek overiť. Tento krok odpovedá číslami na dve otázky:
1. ktoré nastavenie vyhľadávania naozaj nájde pasáž, v ktorej je odpoveď,
2. o koľko vyhľadávanie zlepší odpovede oproti samotnému LLM, vrátane toho, či povie „neviem“, keď má.

## Setup
- **Dáta:** ClinicalTrials.gov API v2 (zadarmo, bez kľúča). Najviac 50 ukončených intervenčných štúdií so
  zverejnenými výsledkami pre každú z oblastí: diabetes 2. typu, rakovina prsníka, astma, Alzheimerova choroba,
  reumatoidná artritída, hypertenzia, depresívna porucha a psoriáza, spolu 396 unikátnych štúdií. Zoznam ID je
  zmrazený v `eval/study_ids.txt`, takže `make data` stiahne ten istý korpus znova.
- **Eval set** (`eval/questions.jsonl`, seed 0): 7 typov otázok × 18 (fáza, počet účastníkov, hlavný sponzor, rok
  začiatku, zaslepenie, intervencia, minimálny vek), plus 15 otázok na fakt, ktorý žiadny záznam nemá (rozpočet),
  a 15 otázok na neexistujúce štúdie. Otázky pomenúvajú štúdiu názvom, nikdy nie NCT ID.
- **Modely:** embeddingy `BAAI/bge-small-en-v1.5`, reranker `cross-encoder/ms-marco-MiniLM-L-6-v2`, LLM `qwen3:4b`
  v Ollame. Všetky sa nastavujú v `.env`.
- **Hardvér:** čísla pre BM25 sú z cloudového Linux CPU. Všetko ostatné pobeží na Macu (MPS).

## Ako to funguje
```
ClinicalTrials.gov ─► rag/ingest.py ─► rag/chunking.py ─► rag/index.py ─► BM25 (pamäť) + Qdrant (vektory)
                                                                              │
React UI ◄── SSE ── api/main.py ◄── rag/answer.py (prompt, citácie) ◄── rag/retrieve.py (4 režimy)
```
- **Záznamy sa menia na text so sekciami** (`rag/ingest.py`). Hodnoty z API ako `PHASE2` / `DOUBLE` sa menia na
  „Phase 2“ / „Double“, pretože BM25 aj LLM čítajú slová, nie kódy.
- **Dva spôsoby chunkingu** (`rag/chunking.py`). `fixed` spojí záznam a reže každých 220 slov. `section` reže podľa
  sekcií záznamu a každý chunk začína hlavičkou `[NCT ID] názov | sekcia`. Prečo: chunk „Enrollment: 240
  participants“ je bezcenný, ak v ňom nie je, ku ktorej štúdii patrí.
- **Štyri režimy vyhľadávania** (`rag/retrieve.py`):
  - `bm25`: vlastná implementácia na 40 riadkov s Lucene idf. Klasické idf v rank_bm25 je záporné pre slová,
    ktoré sú vo viac ako polovici korpusu.
  - `dense`: normalizované bge vektory v Qdrante, takže cosine sa rovná skalárnemu súčinu (krok 10).
  - `hybrid`: Reciprocal Rank Fusion, `Σ 1/(60 + poradie)`. Používa iba poradie, takže skóre BM25 a cosine
    podobnosti netreba prevádzať na jednu škálu.
  - `rerank`: cross-encoder prečíta top 20 kandidátov z hybridu *spolu s otázkou*. Na celý korpus je príliš pomalý,
    na krátkom zozname je presný. To je klasický dvojstupňový dizajn.
- **Prompt postavený na zdrojoch** (`rag/answer.py`). Každý zdroj je označený ID štúdie, model musí za každým
  faktom citovať a odpovedať presne „I don't know“, keď odpoveď v zdrojoch nie je. Po generovaní sa každé citované
  ID, ktoré nebolo medzi zdrojmi, označí ako vymyslená citácia.
- **API** (`api/main.py`): `POST /api/ask` streamuje `sources → token… → done` ako SSE. SSE a nie WebSockety, lebo
  stream ide jedným smerom, je to obyčajné HTTP a prejde cez proxy aj load balancery (kroky 17–18). Pydantic
  validuje vstup (otázka 3–500 znakov, k 1–10).
- **UI** (`frontend/`): React 19 + TypeScript + Vite. SSE stream z `fetch` parsuje ručne, lebo `EventSource` nevie
  POST. Citácie otvoria celý zdrojový záznam a panel zdrojov ukazuje pruhy skóre pre každú fázu, takže vidno,
  *prečo* odpoveď vyšla tak, ako vyšla. Toto UI je vizuálny príklad kroku. Autíčka z Racing Labu som nepoužil,
  lebo vyhľadávanie a citácie nemajú prirodzenú analógiu v jazdení.
- **Tajomstvá:** všetko sa číta z prostredia cez `pydantic-settings`. `.env` je v `.gitignore`, v gite je iba
  `.env.example`. AWS používa svoj štandardný reťazec prihlasovacích údajov (SSO profil), nikdy nie kľúče v súbore.

## Výsledky
### Vyhľadávanie
„evidence@k“ znamená, že v top k je chunk, ktorý **obsahuje gold fakt** (napr. presný riadok `Enrollment: 240
participants`). „trial@k“ sa pýta iba na hocijaký chunk správnej štúdie. 141 otázok (otázky na neexistujúce štúdie
nemajú gold záznam); cloudové CPU.

| Chunking | Režim | evidence@1 | evidence@5 | evidence@10 | MRR@10 | trial@5 | latencia p50 |
|---|---|---|---|---|---|---|---|
| fixed (220 slov) | bm25 | 0,532 | 0,770 | 0,810 | 0,627 | 1,000 | 0,2 ms |
| section + hlavička | bm25 | 0,421 | **0,905** | 0,984 | 0,613 | 1,000 | 0,2 ms |
| oba | dense / hybrid / rerank | *čaká na beh na Macu* | | | | | |

![Vyhľadávanie](assets/retrieval.png)

### Odpovede
*Čaká na beh na Macu* (`make eval-generation`): presnosť, presnosť podložená zdrojom (správne **a** cituje správnu
štúdiu), odmietnutie pri nezodpovedateľných otázkach, zbytočné odmietnutia, vymyslené citácie a latencia, pre
samotné LLM aj pre každý režim vyhľadávania.

## Analýza chýb
- **trial@5 = 1,0 pre BM25 je varovanie, nie úspech.** Otázky citujú presný názov štúdie, takže na nájdenie štúdie
  stačí zhoda slov. Preto je hlavnou metrikou evidence@k: ťažké je nájsť *pasáž s odpoveďou*.
- **Chunky podľa sekcií prehrávajú pri @1, ale vyhrávajú pri @5 (0,421 vs 0,532, potom 0,905 vs 0,770).**
  Pravdepodobná príčina: každý chunk sekcie opakuje názov v hlavičke, takže všetky chunky správnej štúdie sa s
  názvom zhodujú rovnako. Poradie potom určí pár zvyšných slov otázky a penalizácia dĺžky v BM25, ktorá posúva
  krátke chunky nahor. Správna štúdia aj tak zaplní top 5. Presne na toto je reranker, ktorý číta otázku a chunk
  spolu. Treba to overiť behom s rerankom.

## Čo som sa naučil
- Metrika vyhľadávania musí kontrolovať *odpoveď*, nie *dokument*. Bez toho BM25 vyzeralo dokonale.
- RRF spojí poradia bez kalibrovania skóre a celá metóda je jeden riadok.
- Streamovanie potrebuje explicitný protokol udalostí (`sources`, `token`, `done`, `error`), aby UI vedelo ukázať
  zdroje pred prvým tokenom a nahlásiť nefunkčné LLM namiesto zaseknutia.

## Spustenie
```bash
cp .env.example .env              # potom uprav; .env nikdy necommituj
make sync                         # uv sync + npm ci
make data                         # zmrazený zoznam štúdií → data/studies.jsonl, eval/questions.jsonl
make index                        # chunking + embeddingy pre oba spôsoby (embedded Qdrant v data/qdrant)
ollama pull qwen3:4b
make eval-retrieval               # → results.json
make eval-generation              # → results.json, eval/runs/*.jsonl
make plot                         # → assets/*.png
make api                          # http://localhost:8000/docs
make web                          # http://localhost:5173
make test lint
```
Voliteľne: `docker compose up -d qdrant` a `QDRANT_URL=http://localhost:6333` pre skutočný Qdrant server.
Embedded Qdrant dovolí naraz iba jeden proces, preto pred evalom zastav API.

## Obmedzenia
- Otázky sú generované zo šablón a citujú presné názvy, čo zvýhodňuje vyhľadávanie podľa kľúčových slov.
  Parafrázované otázky (napísané LLM a potom ručne skontrolované) sú ďalší krok pre férovejšie porovnanie dense
  a BM25.
- Odpovede sa kontrolujú pravidlami (je tam číslo, sú tam slová názvu). Je to transparentné, ale nevie to posúdiť
  dlhé voľné odpovede, preto sa eval pýta iba na krátke fakty.
- Iba otázky na jednu štúdiu. Otázky naprieč štúdiami („ktoré štúdie psoriázy testovali X?“) sa nemerajú.
- Poskytovatelia Bedrock a Azure sú napísaní, ale nespustení (stoja peniaze; plánované na kroky 18 a 19).
