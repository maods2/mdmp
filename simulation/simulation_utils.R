# Simulation utility functions for MDM DAG structure evaluation
# Helper functions for building adjacency matrices and computing metrics

#' Build connection matrix from adjacency matrix
#'
#' Creates a symmetric matrix representing undirected connections
#' from a directed adjacency matrix.
#'
#' @param adj_mat Square adjacency matrix (n x n) where 1 indicates an edge
#' @return Symmetric connection matrix (n x n) where 1 indicates any connection
build_connection_matrix <- function(adj_mat) {
  n_n <- ncol(adj_mat)
  k <- diag(rep(0, n_n))
  lower_ind <- which(lower.tri(k), arr.ind = TRUE)
  upper_ind <- t(combn(ncol(k), 2))
  m_con <- adj_mat[lower_ind] + adj_mat[upper_ind]
  k[lower_ind] <- k[upper_ind] <- m_con == 1
  return(list(
    connection_matrix = k,
    connection_vector = m_con,
    lower_ind = lower_ind,
    upper_ind = upper_ind
  ))
}

#' Compute evaluation metrics for DAG structure learning
#'
#' Computes accuracy, sensitivity, specificity, PPV, NPV, and directional accuracy
#' by comparing true and estimated adjacency matrices.
#'
#' @param true_adj True adjacency matrix (n x n)
#' @param estimated_adj Estimated adjacency matrix (n x n)
#' @return List containing all evaluation metrics
compute_metrics <- function(true_adj, estimated_adj) {
  n_n <- ncol(true_adj)
  
  # Build connection matrices (undirected representation)
  true_con <- build_connection_matrix(true_adj)
  est_con <- build_connection_matrix(estimated_adj)
  
  m_con_true <- true_con$connection_vector
  m_con_est <- est_con$connection_vector
  
  # Connection accuracy
  accuracy <- mean(m_con_est == m_con_true)
  
  # Sensitivity (true positive rate)
  sensitivity <- mean(m_con_est[m_con_true == 1] == 1)
  if (is.nan(sensitivity)) sensitivity <- NA
  
  # Specificity (true negative rate)
  specificity <- mean(m_con_est[m_con_true == 0] == 0)
  if (is.nan(specificity)) specificity <- NA
  
  # Positive Predictive Value (PPV)
  ppv <- mean(m_con_true[m_con_est == 1] == 1)
  if (is.nan(ppv)) ppv <- NA
  
  # Negative Predictive Value (NPV)
  npv <- mean(m_con_true[m_con_est == 0] == 0)
  if (is.nan(npv)) npv <- NA
  
  # Directional accuracy
  k <- true_con$connection_matrix
  upper_ind <- true_con$upper_ind
  lower_ind <- true_con$lower_ind
  
  # Only evaluate directions where connections exist
  d_accuracy <- mean(
    (true_adj[upper_ind][k[upper_ind] == 1] == estimated_adj[upper_ind][k[upper_ind] == 1]) &
    (true_adj[lower_ind][k[lower_ind] == 1] == estimated_adj[lower_ind][k[lower_ind] == 1])
  )
  if (is.nan(d_accuracy)) d_accuracy <- NA
  
  return(list(
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    ppv = ppv,
    npv = npv,
    directional_accuracy = d_accuracy
  ))
}

#' Convert adjacency matrix to data frame with row/column names
#'
#' @param adj_mat Adjacency matrix
#' @param node_names Vector of node names (optional)
#' @return Data frame with labeled rows and columns
adj_matrix_to_df <- function(adj_mat, node_names = NULL) {
  if (is.null(node_names)) {
    node_names <- paste0("Y", 1:ncol(adj_mat))
  }
  
  result <- data.frame(adj_mat)
  rownames(result) <- colnames(result) <- node_names
  return(result)
}
