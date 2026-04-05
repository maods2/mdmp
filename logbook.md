# 📘 Research & Development Diary – Repository Evolution

## 🔹 Purpose

This diary is meant to document the evolution of the repository across different branches and methodological phases, enabling future recall of decisions, ideas, and implementations without needing to revisit the codebase in detail.

---

## 🧩 Branch: `group-analysis-v1`

### 📌 Method Implemented

**VST (Virtual T Structure)**

### 🧠 Core Idea

The main idea behind this approach is to build a **global structure across multiple individuals** by leveraging temporal information.

Two main strategies were explored:

* **Recomputing temporal windows** for each individual
* **Averaging temporal representations** across individuals

From these aggregated temporal representations, a **global DAG structure** is estimated for the dataset.

### 🎯 Goal

Capture a **shared structural pattern** across individuals while still respecting temporal dynamics.

### ⚠️ Notes / Limitations

* Assumes some level of alignment across individuals
* Temporal aggregation may smooth out individual-specific patterns
* Structure may be sensitive to how temporal windows are defined

---

## 🧩 Branch: `group-analysis-v2`

### 📌 Method Implemented

**IS (Individual Structure) Aggregation**

### 🧠 Core Idea

Instead of aggregating data, this approach aggregates **structures**:

1. Learn one DAG per individual
2. Analyze edge occurrence across individuals
3. Build a **global DAG based on edge frequency**

### ⚙️ Method

* For each edge:

  * Keep it only if it appears in **more than 50% of individuals**
* This acts as a **filtering mechanism** over individual graphs

### 🎯 Goal

Capture **robust and consistent relationships** that generalize across individuals.
