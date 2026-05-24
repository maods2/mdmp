# Individual Structure (IS) aggregation — logic diagrams

## Statistical Interpretation (reference)

Before reading the diagrams, keep three distinctions in mind:

| Claim | What the code actually computes |
|---|---|
| "global effect" of edge p→c | **Conditional** mean E\[θ\|edge=1\]: only subjects expressing the edge contribute; absent subjects are excluded from both numerator and divisor |
| "posterior credible interval" | MC uncertainty from **independent** per-subject DLM posteriors; **not** a joint hierarchical posterior; no shrinkage, no between-subject covariance |
| "structural uncertainty" | G\* is **fixed** before any MC; inference is p(θ\|G\*); structural uncertainty is **not** propagated |

**Pooling semantics.**
`pooling='conditional_mean_among_edge_subjects'` (legacy alias `'mean_with_edge'`):

```
θ̄_t^(b) = (1/A) Σ_{i ∈ 𝒜} θ_{i,t}^(b)
```

where 𝒜 = subjects whose individual DAG contained edge p→c, A = |𝒜|.
This estimates E\[θ_{pc,t} | edge_{pc} = 1\], **not** the unconditional population mean (1/S) Σ_i θ_{i,t}^(b).

**Smoothed draws.**
When `mc_posterior='smoothed'`, the code uses smoothed moments (smt, sCt) together with filtered (nt, dt) at the same time index — a pragmatic approximation, not the exact full smoothing posterior.

Diagramas em [Mermaid](https://mermaid.js.org/). Pré-visualize no GitHub, VS Code (extensão Mermaid), ou [mermaid.live](https://mermaid.live).

## 1. Pipeline (`inds.pipeline.run_full`)

```mermaid
flowchart TD
  V[validate] --> C[coerce]
  C --> VC[validate_coerced]
  VC --> Vote[vote_edge_frequencies]
  Vote --> Repair[repair_dag_to_acyclic]
  Repair --> Refit{refit on G*?}
  Refit -->|mc_refit_global_structure| R[refit_on_consensus]
  Refit -->|skip| MCgate{n_draws > 0?}
  R --> MCgate
  MCgate -->|yes| MC[run_inds_global_beta_mc]
  MCgate -->|no| Asm[assemble_view]
  MC --> Asm
```

Split entry points: `vote_individual_structures`, `refit_on_consensus`,
`run_inds_global_beta_mc`, `pool_conditional_filtered_states`, `as_inds_mdm_view`.

## 2. Visão geral: `aggregate_individual_structures`

```mermaid
flowchart LR
  subgraph in["Entrada"]
    M["Lista: adjacências OU MDMs ajustados"]
  end

  subgraph norm["1. Normalizar"]
    N1["Um único MDM → [MDM]"]
    N2["Matriz 2D única → [adj]"]
  end

  subgraph coerce["2. Coerção MDM"]
    C1{Todos MDM-like?}
    C2["Extrair adj: (adj_mat > 0)"]
    C3["Opcional: Filt de cada m"]
    C4["Opcional: plot_data = média dos data"]
  end

  subgraph val["3. Validar"]
    V["Adj binárias N×N; node_names; plot_data (T,N)"]
  end

  subgraph vote["4. Voto + DAG"]
    Freq["Contar arestas por sujeito → frequência"]
    Thr{"threshold_mode"}
    S["strict: freq > τ"]
    I["inclusive: freq ≥ τ"]
    Cyc["Greedy: enquanto houver ciclo, remover do ciclo a aresta de menor freq"]
  end

  subgraph out["5. Saída base"]
    R["ISAggregatedMDMView: adj_mat global, metadata"]
  end

  subgraph opt["Opcional"]
    PF["pool_filt / plot_filt → Filt global para plots"]
    RF["mc_refit_global_structure → refit MDM na DAG global"]
    MC["n_draws > 0 → global_beta_mc"]
  end

  M --> norm --> coerce
  coerce --> val --> vote --> R
  Freq --> Thr
  Thr --> S
  Thr --> I
  S --> Cyc
  I --> Cyc
  Cyc --> R
  R --> PF
  R --> RF
  R --> MC
```

## 2. Voto por aresta e reparo de ciclos (greedy)

O reparo **não** calcula um conjunto mínimo global de arcos de feedback (minimum FAS): em cada iteração encontra-se **um** ciclo dirigido, remove-se a aresta desse ciclo com menor **frequência empírica** entre os sujeitos, e repete-se até obter um DAG.

```mermaid
flowchart TD
  S["Empilhar matrizes I×N×N dos sujeitos"]
  C["edge_counts[i,j] = quantos sujeitos têm i→j"]
  P["edge_freq[i,j] = contagem / I"]
  T{"threshold_mode"}
  Ts["strict: aresta candidata se freq > τ"]
  Ti["inclusive: candidata se freq ≥ τ"]
  G["Grafo binário candidato"]

  loop["Loop greedy"]
  DAC{"É DAG?"}
  FC["Achar um ciclo dirigido (ex.: NetworkX)"]
  RM["Remover do ciclo a aresta com menor freq"]
  OK["DAG final = adj_mat global"]

  S --> C --> P --> T
  T --> Ts --> G
  T --> Ti --> G
  G --> loop
  loop --> DAC
  DAC -->|sim| OK
  DAC -->|não| FC --> RM --> loop
```

## 3. Monte Carlo nos coeficientes das arestas globais

Interpretação em dois eixos:

- **Posterior por sujeito:** por defeito usa-se o **filtro** individual sob a DAG **individual**; com `mc_refit_global_structure`, primeiro corre-se o mesmo pipeline que o MDM (`refit_mdm_on_structure`) com a DAG global fixa, obtendo posteriores **à estrutura global**.
- **Fonte temporal:** `mc_posterior='filtered'` amostra `(mt, Ct, nt, dt)`; `'smoothed'` usa `(smt, sCt)` do smooth com `nt, dt` do filtro no mesmo instante (reutilização da rotina de amostragem tipo *t*).

```mermaid
flowchart TD
  subgraph prep["Preparação"]
    E["Arestas globais ordenadas parent→child"]
    PL["Listas de pais por nó/sujeito (design matrix na DAG usada no MC: individual ou global após refit)"]
  end

  subgraph rep["Para cada réplica b = 1…B"]
    subgraph samp["Amostragem por sujeito"]
      Si["Para cada sujeito i"]
      Nj["Para cada nó filho c"]
      Draw["Amostrar estado em t (filtro ou smooth + nt/dt)"]
    end

    subgraph pool["Agregar por aresta global e ∈ E"]
      Ind["mc_contributors=individual_edge: só quem tem a aresta na DAG individual"]
      All["mc_contributors=all_subjects: todos após refit global"]
      Align["Extrair coeficiente do pai global no vetor local"]
      Mean["mean_with_edge / sum_with_edge"]
    end
  end

  subgraph post["Depois das B réplicas"]
    BD["beta_draws: eixo 0 = b"]
    MV["beta_mean, beta_var nan-aware no eixo b"]
    T1["Um t: forma B × |E|"]
    Tk["time_indices: forma B × |E| × |T|"]
    Q["mc_quantiles → beta_quantiles ao longo do eixo b"]
  end

  prep --> rep
  samp --> pool
  rep --> post
```

## 4. DAG individual × DAG global (MC)

```mermaid
flowchart LR
  subgraph ind["Sujeito i"]
    Gi["DAG_i (voto) ou DAG global (após refit)"]
    Ti["Amostra θ_i^(b) no nó c"]
  end

  subgraph glob["DAG global agregada"]
    Gg["Aresta p→c"]
  end

  Gi --> Align["Alinhar: coef. de p no bloco local de c"]
  Ti --> Align
  Gg --> Align
  Align --> M["Contribui conforme mc_contributors"]
```

## Referência no código

- Orquestração: `aggregate_individual_structures` em `aggregation.py`
- Voto + ciclos: `_vote_threshold_and_repair_cycles`, `_remove_lowest_freq_cycle_edge`
- Refit estrutura fixa: `mdmp.model.refit_mdm_on_structure`
- MC: `_monte_carlo_global_edge_beta`, `_monte_carlo_beta_draws_at_time`, `_sample_dlm_state_posterior`
- Filt agregado para plots: `build_plot_filt_from_subjects`
