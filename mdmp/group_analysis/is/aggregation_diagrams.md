# Individual Structure (IS) aggregation — logic diagrams

Diagramas em [Mermaid](https://mermaid.js.org/). Pré-visualize no GitHub, VS Code (extensão Mermaid), ou [mermaid.live](https://mermaid.live).

## 1. Visão geral: `aggregate_individual_structures`

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
    Thr["Manter aresta se freq > τ"]
    Cyc["Enquanto houver ciclo: remover aresta do ciclo com menor freq"]
  end

  subgraph out["5. Saída base"]
    R["ISAggregatedMDMView: adj_mat global, metadata"]
  end

  subgraph opt["Opcional"]
    PF["pool_filt / plot_filt → Filt global para plots"]
    MC["n_draws > 0 → global_beta_mc"]
  end

  M --> norm --> coerce
  coerce --> val --> vote --> R
  R --> PF
  R --> MC
```

## 2. Voto por aresta e reparo de ciclos

```mermaid
flowchart TD
  S["Empilhar matrizes I×N×N dos sujeitos"]
  C["edge_counts[i,j] = quantos sujeitos têm i→j"]
  P["edge_freq[i,j] = contagem / I"]
  T["Aresta candidata se freq[i,j] > τ"]
  G["Grafo binário candidato"]

  loop["Loop"]
  DAC{"É DAG?"}
  FC["Achar um ciclo dirigido"]
  RM["Remover do ciclo a aresta com menor freq"]
  OK["DAG final = adj_mat global"]

  S --> C --> P --> T --> G --> loop
  loop --> DAC
  DAC -->|sim| OK
  DAC -->|não| FC --> RM --> loop
```

## 3. Monte Carlo nos coeficientes das arestas globais

```mermaid
flowchart TD
  subgraph prep["Preparação"]
    E["Arestas globais ordenadas parent→child"]
    A["Por sujeito: quais arestas existem no DAG individual"]
    PL["Listas de pais locais por nó e sujeito (alinhamento)"]
  end

  subgraph rep["Para cada réplica b = 1…B"]
    subgraph samp["Amostragem por sujeito"]
      Si["Para cada sujeito i"]
      Nj["Para cada nó filho c"]
      Draw["Amostrar vetor de estado do filtro em t<br/>mistura Gamma–Normal → t multivariada em (m_t, C_t, n_t, d_t)"]
    end

    subgraph pool["Agregar por aresta global e ∈ E"]
      Cont["Só sujeitos com essa aresta no grafo individual"]
      Align["Extrair coeficiente do pai global no vetor local"]
      Mean["mean_with_edge: θ̄^(b) = (1/A) Σ valores"]
      Sum["sum_with_edge: soma em vez de média"]
    end
  end

  subgraph post["Depois das B réplicas"]
    BD["beta_draws: eixo 0 = b"]
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
    Gi["DAG_i com pais locais de c"]
    Ti["Amostra θ_i^(b) no nó c"]
  end

  subgraph glob["DAG global agregado"]
    Gg["Aresta p→c"]
  end

  Gi --> Align["Alinhar: coef. de p no bloco local de c"]
  Ti --> Align
  Gg --> Align
  Align --> M["Incluir na média/soma só se p→c ∈ DAG_i"]
```

## Referência no código

- Orquestração: `aggregate_individual_structures` em `aggregation.py`
- Voto + ciclos: `_vote_threshold_and_repair_cycles`, `_remove_lowest_freq_cycle_edge`
- MC: `_monte_carlo_global_edge_beta`, `_monte_carlo_beta_draws_at_time`, `_sample_dlm_state_posterior`
- Filt agregado para plots: `build_plot_filt_from_subjects`
