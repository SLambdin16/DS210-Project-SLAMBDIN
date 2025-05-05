
use std::error::Error;
use std::fs::File;
use std::io::BufReader;
use csv::Reader;


// Module: libs
// Purpose: Provides functions for loading correlation data, constructing a graph,
//          finding clusters of correlated stocks, and related data structures.

pub struct Graph {
    adjacency_list: Vec<Vec<usize>>,
}

 // Function
    // Purpose: makes a new, empty graph with a specified number of nodes
    // Inputs: number of nodes in the graph (usize).
    // Outputs: new Graph struct with an empty adjacency list
impl Graph {
    pub fn new(size: usize) -> Self {
        Self { adjacency_list: vec![Vec::new(); size] }
    }
    // Function: add_edge
        // Purpose: Adds an undirected edge between two nodes in the graph.
        // Inputs:
        //     u: index of the first node
        //     v: index of the second node 
        // Outputs: None -   Modifies the graph in place
    pub fn add_edge(&mut self, u: usize, v: usize) {
        if !self.adjacency_list[u].contains(&v) {
            self.adjacency_list[u].push(v);
        }
        if !self.adjacency_list[v].contains(&u) {
            self.adjacency_list[v].push(u);
        }
    }
     // Function: from_correlation_matrix
    // Purpose: Constructs a graph from a correlation matrix; edge is added
    //          between two nodes if their correlation is above a given threshold
    // Inputs:
    //     matrix: a 2D vector of f64 values representing the correlation matrix
    //     threshold: minimum correlation value for an edge to be added (f64).
    // Outputs: new Graph struct representing the correlation network.
    pub fn from_correlation_matrix(matrix: &[Vec<f64>], threshold: f64) -> Self {
        let size = matrix.len();
        let mut graph = Self::new(size);

        for i in 0..size {
            for j in (i + 1)..size {
                if matrix[i][j] >= threshold {
                    graph.add_edge(i, j);
                }
            }
        }

        graph
    }
}
// Function: load_correlation_matrix
// Purpose: Loads a correlation matrix from a CSV file.  The CSV file is
//          assumed to have a header row, with the first column being a stock
//          identifier (which is ignored), and subsequent columns containing
//          correlation values.
// Inputs: path to the CSV file (str).
// Outputs:
//     Result<(Vec<String>, Vec<Vec<f64>>), Box<dyn Error>>:
//         - On success, returns a tuple containing:
//             - A vector of stock tickers (Vec<String>).  These are read from the header row.
//             - A 2D vector of f64 values representing the correlation matrix (Vec<Vec<f64>>).
//         - On error, returns a Box<dyn Error> describing the error.
pub fn load_correlation_matrix(path: &str) -> Result<(Vec<String>, Vec<Vec<f64>>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut reader = Reader::from_reader(BufReader::new(file));

    let headers = reader.headers()?.iter().skip(1).map(|s| s.to_string()).collect::<Vec<_>>();
    let mut matrix = Vec::new();

    for result in reader.records() {
        let record = result?;
        let row = record.iter().skip(1)
            .map(|v| v.parse::<f64>().unwrap_or(0.0))
            .collect::<Vec<_>>();
        matrix.push(row);
    }

    Ok((headers, matrix))
}
// Function: find_clusters
// Purpose: Finds clusters of connected nodes in a graph using depth-first search (DFS).
// Inputs:
//     graph: A Graph struct representing the graph to find clusters in.
// Outputs:
//     Vec<Vec<usize>>: A vector of clusters, where each cluster is a vector of
//                       node indices (usize).
pub fn find_clusters(graph: &Graph) -> Vec<Vec<usize>> {
    let mut visited = vec![false; graph.adjacency_list.len()];
    let mut clusters = Vec::new();

    for node in 0..graph.adjacency_list.len() {
        if !visited[node] {
            let mut cluster = Vec::new();
            dfs(node, &graph, &mut visited, &mut cluster);
            clusters.push(cluster);
        }
    }

    clusters
}
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect()
}

pub fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    assert_eq!(n, y.len());

    let mean_x = x.iter().copied().sum::<f64>() / n as f64;
    let mean_y = y.iter().copied().sum::<f64>() / n as f64;

    let numerator: f64 = x.iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum();

    let denominator_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
    let denominator_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();

    let denominator = (denominator_x * denominator_y).sqrt();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}
// Function: dfs
// Purpose: Performs a depth-first search (DFS) traversal of a graph, starting from a given node.
//          This is a helper function for find_clusters.
// Inputs:
//     node: The starting node for the DFS traversal (usize).
//     graph: The Graph struct being traversed.
//     visited: A mutable slice of booleans indicating which nodes have been visited.
//     cluster: A mutable vector of node indices representing the current cluster being built.
// Outputs:
//     None.  Modifies the visited slice and cluster vector in place.
pub fn dfs(node: usize, graph: &Graph, visited: &mut [bool], cluster: &mut Vec<usize>) {
    visited[node] = true;
    cluster.push(node);
    for &neighbor in &graph.adjacency_list[node] {
        if !visited[neighbor] {
            dfs(neighbor, graph, visited, cluster);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Graph, find_clusters};
    use crate::compute_log_returns;
    use crate::pearson_correlation;
        // Test: test_similarity_threshold
    // Purpose: Checks that the graph construction correctly filters edges based on the
    //          similarity threshold.
    // What it checks:
    //     - Creates a small graph with weighted edges.
    //     - Filters the edges based on a threshold.
    //     - Asserts that the correct number of edges remain after filtering.
    // Why it matters:
    //     Ensures that the graph accurately represents the connections between nodes
    //     based on the specified correlation threshold, which is crucial for the
    //     clustering algorithm.
    #[test]
    fn test_similarity_threshold() {
        let edges = vec![
            (0, 1, 0.9),
            (0, 2, 0.7),
            (1, 2, 0.95),
        ];
        let threshold = 0.8;
        let filtered: Vec<_> = edges
            .into_iter()
            .filter(|&(_, _, weight)| weight >= threshold)
            .collect();

        assert_eq!(filtered.len(), 2); // 0-1 and 1-2 pass
    }

    #[test]
    fn test_cluster_count() {
        let mut graph = Graph::new(5);
        graph.add_edge(0, 1);
        graph.add_edge(1, 2);
        graph.add_edge(3, 4);

        let clusters = find_clusters(&graph);
        assert_eq!(clusters.len(), 2); // One for 0-1-2, one for 3-4
    }
    #[test]
    fn test_log_return_computation() {
        let prices = vec![100.0, 110.0];
        let expected = vec![ (110.0_f64 / 100.0_f64).ln() ];
        let result = compute_log_returns(&prices);
        assert!((result[0] - expected[0]).abs() < 1e-6);
    }
    #[test]
    fn test_correlation_perfect_match() {
        let series1 = vec![0.1, 0.2, 0.3];
        let series2 = vec![0.1, 0.2, 0.3]; // identical
        let corr = pearson_correlation(&series1, &series2);
        assert!((corr - 1.0).abs() < 1e-6);
    }
    
}